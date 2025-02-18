import os
import zipfile
import cv2
import numpy as np
import torch
import pandas as pd
import json
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset

from src.random_erasing import RandomErasing
import src.video_transforms as video_transforms
import src.volume_transforms as volume_transforms

from src.sequencing import RegularSequencer, RegularSequencerWithStart
from src.data_utils import smooth_labels, compute_time_vector, tensor_normalize


class FrameClsDataset_DoTA(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train',
                 view_len=8, target_fps=10, orig_fps=10, view_step=10,
                 crop_size=224, short_side_size=320,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=1, test_num_crop=1, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.view_len = view_len
        self.target_fps = target_fps
        self.orig_fps = orig_fps
        self.view_step = view_step
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        #self.new_height = new_height
        #self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.ttc_TT = args.ttc_TT if hasattr(args, "ttc_TT") else 2.
        self.ttc_TA = args.ttc_TA if hasattr(args, "ttc_TA") else 1.
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self._read_anno()
        self._prepare_views()
        assert len(self.dataset_samples) > 0
        assert len(self._label_array) > 0

        if self.args.loss in ("2bce",):
            self.label_array = self._smoothed_label_array
        else:
            self.label_array = self._label_array

        count_safe = self._label_array.count(0)
        count_risk = self._label_array.count(1)
        print(f"\n\n===\n[{mode}] | COUNT safe: {count_safe}\nCOUNT risk: {count_risk}\n===")

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.crop_size, self.crop_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])
            ])
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_names = None
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_ttc = []
        clip_smoothed_labels = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc = compute_time_vector(binary_labels, fps=self.orig_fps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc, before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_ttc.append(ttc)
            clip_smoothed_labels.append(smoothed_labels)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
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
            args = self.args
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                smoothed_label_list = []
                index_list = []
                ttc_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self._label_array[index]
                    smoothed_label = self._smoothed_label_array[index]
                    ttc = self.ttc[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    smoothed_label_list.append(smoothed_label)
                    index_list.append(index)
                    ttc_list.append(ttc)
                extra_info = [{"ttc": ttc_item, "smoothed_labels": slab_item} for ttc_item, slab_item in zip(ttc_list, smoothed_label_list)]
                return frame_list, label_list, index_list, extra_info
            else:
                buffer = self._aug_frame(buffer, args)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer,self._label_array[index], index, extra_info

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "clip": clip_name, "frame": frame_name, "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        # Perform data augmentation - vertical padding and horizontal flip
        # add padding
        do_pad = video_transforms.pad_wide_clips(h, w, self.crop_size)
        buffer = [do_pad(img) for img in buffer]

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            do_transforms=video_transforms.DRIVE_TRANSFORMS
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C
        # T H W C
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                max_area=0.1,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
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
                    short_side = min(min(img.shape[:2]), self.short_side_size)
                    target_side = self.crop_size * resize_scale
                    k = target_side / short_side
                    img = cv2.resize(img, dsize=(0,0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                else:
                    raise ValueError
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)