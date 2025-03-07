import os
import zipfile

import PIL.Image
import cv2
import numpy as np
import torch
import pandas as pd
import json
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from natsort import natsorted
import warnings
from torch.utils.data import Dataset

from src import video_transforms as video_transforms
from src import volume_transforms as volume_transforms
from src.sequencing import RegularSequencer, UnsafeOverlapSequencer, RegularSequencerWithStart
from src.data_utils import smooth_labels, compute_time_vector, tensor_normalize, preprocess_fixation_map


DRIVE_FIXATIONS_TRANSFORMS = [
    "Rotate",
    "ShearX",
    "ShearY",
]


class FrameClsDataset_DADA(Dataset):
    """Load your own video classification dataset."""
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]

    def __init__(self, anno_path, data_path, mode='train',
                 view_len=8, target_fps=30, orig_fps=30, view_step=10,
                 crop_size=224, short_side_size=320, video_ext=".png",
                 keep_aspect_ratio=True, loss_name="crossentropy", test_empty_text=False):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.view_len = view_len
        self.target_fps = target_fps
        self.orig_fps = orig_fps
        self.view_step = view_step
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.video_ext = video_ext
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ttc_TT = 2.
        self.ttc_TA = 1.
        self.fix_size = 64
        self.aug = False
        self.test_empty_text = test_empty_text
        if self.mode in ['train']:
            self.aug = True

        self._read_anno()
        self._prepare_views()
        assert len(self.dataset_samples) > 0
        assert len(self._label_array) > 0

        if loss_name in ("2bce",):
            self.label_array = self._smoothed_label_array
        else:
            self.label_array = self._label_array

        count_safe = self._label_array.count(0)
        count_risk = self._label_array.count(1)
        print(f"\n\n===\n[{mode}] | COUNT safe: {count_safe}\nCOUNT risk: {count_risk}\n===")

        if (mode == 'train'):
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.data_transform_fixation = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.data_transform_fixation = transforms.Compose([
                transforms.ToTensor(),
            ])

        if mode == 'test':
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_toa = []
        clip_ttc = []
        clip_acc = []
        clip_text = []
        clip_smoothed_labels = []

        errors = []

        with open(os.path.join(self.data_path, self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]

        df = pd.read_csv(os.path.join(self.data_path, "annotation", "full_anno.csv"))

        for clip in tqdm(clip_names, "Part 1/2. Reading and checking clips"):
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            if len(row) != 1:
                errors.append(info)
            row = row.iloc[0]
            text = row["texts"]
            with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1] == self.video_ext])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            if_acc_video = int(row["whether an accident occurred (1/0)"])
            st = int(row["abnormal start frame"])
            en = int(row["abnormal end frame"])
            if st > -1 and en > -1:
                binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
            else:
                binary_labels = [0 for t in timesteps]
            cat_labels = [l * clip_type for l in binary_labels]
            if_ego = clip_type in self.ego_categories
            toa = int(row["accident frame"])
            ttc = compute_time_vector(binary_labels, fps=self.orig_fps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc,
                                            before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_toa.append(toa)
            clip_ttc.append(ttc)
            clip_text.append(text)
            clip_acc.append(if_acc_video)
            clip_smoothed_labels.append(smoothed_labels)

        for line in errors:
            print(line)
        if len(errors) > 0:
            print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
            exit(0)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_toa = clip_toa
        self.clip_ttc = clip_ttc
        self.clip_text = clip_text
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in tqdm(range(N), desc="Part 2/2. Preparing views"):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.orig_fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            smoothed_label_array.extend([self.clip_smoothed_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([self.clip_ttc[i][seq[-1]] for seq in sequences])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.ttc = ttc
        self._smoothed_label_array = smoothed_label_array

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            clip_id, seq_timesteps = sample
            buffer, _, __ = self.load_images_zip(sample, final_resize=False, resize_scale=1.)  # T H W C
            fixation, _, __ = self.load_fixations_zip(sample, final_resize=False, resize_scale=1.)  # T H W C

            # for i, (b, f) in enumerate(zip(buffer, fixation)):
            #     if np.max(f) > 0:
            #         clip_name = self.clip_names[clip_id].replace("/", "-")
            #         fname = self.clip_timesteps[clip_id][seq_timesteps[i]]
            #         img1 = cv2.resize(f, (64, 64))
            #         cv2.imwrite(f"out/deb_{clip_name}_{fname}_f.jpg", f)
            #         cv2.imwrite(f"out/deb_{clip_name}_{fname}_f1.jpg", img1)
            #         cv2.imwrite(f"out/deb_{clip_name}_{fname}_b.jpg", b)
            #         exit(0)

            if len(buffer) == 0 or len(fixation) == 0:
                while len(buffer) == 0 or len(fixation) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_zip(sample, final_resize=False, resize_scale=1.)
                    fixation, _, __ = self.load_fixations_zip(sample, final_resize=False, resize_scale=1.)

            buffer, fixation = self._aug_frame(buffer, fixation)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index], "index": index}
            labels = np.array([self.clip_bin_labels[clip_id][ts] for ts in seq_timesteps]).astype(int)
            text = self.clip_text[clip_id]
            return buffer, fixation, labels, extra_info, text

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            clip_id, seq_timesteps = sample
            buffer, _, __ = self.load_images_zip(sample, final_resize=True, apply_tf=True)
            fixation, _, __ = self.load_fixations_zip(sample, final_resize=True, apply_tf=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_zip(sample, final_resize=True, apply_tf=True)
                    fixation, _, __ = self.load_fixations_zip(sample, final_resize=True, apply_tf=True)
            buffer = torch.stack(buffer, dim=0)
            fixation = torch.stack(fixation, dim=0)/ 255.
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index], "index": index}
            labels = np.array([self.clip_bin_labels[clip_id][ts] for ts in seq_timesteps]).astype(int)
            if self.test_empty_text:
                text = "a video frame of { }"
            else:
                text = self.clip_text[clip_id]
            return buffer, fixation, labels, extra_info, text

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            clip_id, seq_timesteps = sample
            buffer, clip_name, frame_name = self.load_images_zip(sample, final_resize=True, apply_tf=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images_zip(sample, final_resize=True, apply_tf=True)
            buffer = torch.stack(buffer, dim=0)
            extra_info = {"ttc": self.ttc[index], "clip": clip_name, "frame": frame_name,
                          "smoothed_labels": self._smoothed_label_array[index]}
            labels = np.array([self.clip_bin_labels[clip_id][ts] for ts in seq_timesteps]).astype(int)
            if self.test_empty_text:
                text = "a video frame of { }"
            else:
                text = self.clip_text[clip_id]
            return buffer, None, labels, extra_info, text
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
            self,
            buffer,
            fixation,
    ):
        h, w, _ = buffer[0].shape
        fh, fw = fixation[0].shape
        assert fh == h and fw == w
        # Perform data augmentation - vertical padding and horizontal flip
        # add padding
        do_pad = video_transforms.pad_wide_clips(h, w, self.crop_size)
        buffer = [do_pad(img) for img in buffer]
        fixation = [do_pad(img) for img in fixation]

        aug_double_transform = video_transforms.create_random_double_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment='rand-m6-n3-mstd0.5-inc1',
            interpolation='bicubic',
            do_transforms=video_transforms.DRIVE_TRANSFORMS,
            ok_geometric_transforms=DRIVE_FIXATIONS_TRANSFORMS
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        fixation = [transforms.ToPILImage()(frame) for frame in fixation]
        buffer, fixation = aug_double_transform([buffer, fixation])

        # buffer[0].save("out/deb_d.jpg")
        # fixation[0].save("out/deb_f.jpg")
        # fixation[0].resize((64,64), resample=PIL.Image.Resampling.BILINEAR).save("out/deb_f2.jpg")
        # print("fix max min: ", fixation[0].getextrema())
        # exit(0)

        buffer = [self.data_transform(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W

        fixation = [self.data_transform_fixation(img.resize((64,64), resample=PIL.Image.Resampling.BILINEAR)) for img in fixation]
        fixation = torch.stack(fixation)  # T 1 H W
        fixation /= 255

        return buffer, fixation

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        subclip = clip_name.split("/")[1]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{subclip}_frame_{ts}{self.video_ext}" for ts in timesteps]
        view = []
        for fname in filenames:
            img = cv2.imread(os.path.join(self.data_path, "frames", clip_name, fname))
            if img is None:
                print("Image doesn't exist! ", fname)
                exit(1)
            if final_resize:
                img = cv2.resize(img, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            elif resize_scale is not None:
                short_side = min(min(img.shape[:2]), self.short_side_size)
                target_side = self.crop_size * resize_scale
                k = target_side / short_side
                img = cv2.resize(img, dsize=(0, 0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            view.append(img)
        # view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def load_images_zip(self, dataset_sample, final_resize=False, resize_scale=None, apply_tf=False):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(4)}{self.video_ext}" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                if final_resize:
                    img = cv2.resize(img, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
                elif resize_scale is not None:
                    short_side = min(img.shape[:2])
                    target_side = self.crop_size * resize_scale
                    k = target_side / short_side
                    img = cv2.resize(img, dsize=(0, 0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                else:
                    raise ValueError
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                if apply_tf:
                    img = self.data_transform(img)
                view.append(img)
        # view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def load_fixations_zip(self, dataset_sample, final_resize=False, resize_scale=None, apply_tf=False):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(4)}{self.video_ext}" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "fixation.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) * 255
                    img = preprocess_fixation_map(img, new_size=(img.shape[1], img.shape[0]))
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                if final_resize:
                    img = cv2.resize(img, dsize=(self.fix_size, self.fix_size), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
                elif resize_scale is not None:
                    short_side = min(img.shape[:2])
                    target_side = self.crop_size * resize_scale
                    k = target_side / short_side
                    img = cv2.resize(img, dsize=(0, 0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
                else:
                    raise ValueError
                if apply_tf:
                    img = self.data_transform_fixation(img)
                view.append(img)
        # view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=1,
        motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


if __name__ == "__main__":

    exit(0)
