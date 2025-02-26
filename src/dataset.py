import os
import json
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange, repeat, reduce
import glob
from io import BytesIO
import zipfile
from natsort import natsorted
from PIL import Image


# before 1/(1+exp(-6*(x+1))), after 1/(1+exp(-12*(-x+0.5)))
def compute_time_vector(labels, fps, TT=2, TA=1):
    """
    Compute time vector reflecting time in seconds before or after anomaly range.

    Parameters:
        labels (list or np.ndarray): Binary vector of frame labels (1 for anomalous, 0 otherwise).
        fps (int): Frames per second of the video.
        TT (float): Time-to-anomalous range in seconds (priority).
        TA (float): Time-after-anomalous range in seconds.

    Returns:
        np.ndarray: Time vector for each frame.
    """
    num_frames = len(labels)
    labels = np.array(labels)
    default_value = max(TT, TA) * 2
    time_vector = torch.zeros(num_frames, dtype=float)

    # Get anomaly start and end indices
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) == 0:
        return time_vector  # No anomalies, return all zeros

    # Define maximum frame thresholds for TT and TA
    TT_frames = int(TT * fps)
    TA_frames = int(TA * fps)

    # Iterate through each frame
    for i in range(num_frames):
        if labels[i] == 1:
            time_vector[i] = 0  # Anomalous frame, set to 0
        else:
            # Find distances to the start and end of anomaly ranges
            distances_to_anomalies = anomaly_indices - i

            # Time-to-closest-anomaly-range (TT priority)
            closest_to_anomaly = distances_to_anomalies[distances_to_anomalies > 0]  # After the frame
            if len(closest_to_anomaly) > 0 and closest_to_anomaly[0] <= TT_frames:
                time_vector[i] = -closest_to_anomaly[0] / fps
                continue

            # Time-after-anomaly-range (TA range)
            closest_after_anomaly = distances_to_anomalies[distances_to_anomalies < 0]  # Before the frame
            if len(closest_after_anomaly) > 0 and abs(closest_after_anomaly[-1]) <= TA_frames:
                time_vector[i] = -closest_after_anomaly[-1] / fps
                continue

            # Outside both TT and TA
            time_vector[i] = -100.

    return time_vector


class DADA(Dataset):
    def __init__(self, root_path, phase, interval,transform,
                  data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms= transform
        self.data_aug = data_aug
        self.fps = 30
        self.num_classes = 2
        self.data_list, self.labels, self.clips, self.toas ,self.texts= self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(self.root_path, self.phase, self.phase + '.txt')
        assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
        fileIDs, labels, clips, toas,texts= [], [], [], [],[]
        samples_visited, visit_rows = [], []
        skipped_folders = []
        with open(list_file, 'r',encoding='utf-8') as f:
            # for ids, line in enumerate(f.readlines()):
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split(',')
                # print(sample )
                sample1=sample[0].strip().split(' ')
                word = sample[1].replace('\xa0', ' ')
                word.strip()
                # Check if we have this video clip folder, otherwise skip line
                clip_folder_path = os.path.join(self.root_path, self.phase, 'rgb_videos', sample1[0])
                if not os.path.exists(clip_folder_path):
                    skipped_folders.append(clip_folder_path)
                    continue
                # Adding info
                fileIDs.append(sample1[0])  # 1/002
                labels.append(int(sample1[1]))  # 1: positive, 0: negative
                clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
                toas.append(int(sample1[4]))  # time-of-accident (toa)
                texts.append(word.strip())
                sample_id = sample1[0] + '_' + sample1[1]
                if sample_id not in samples_visited:
                    samples_visited.append(sample_id)
                    visit_rows.append(ids)
        # if not self.data_aug:
        #     fileIDs = [fileIDs[i] for i in visit_rows]
        #     labels = [labels[i] for i in visit_rows]
        #     clips = [clips[i] for i in visit_rows]
        #     toas = [toas[i] for i in visit_rows]
        # print(fileIDs,labels,clips,toas,texts )
        if len(skipped_folders) > 0:
            print(f"\nSKIPPED FOLDERS:")
            print("\n".join(skipped_folders))
        return fileIDs, labels, clips, toas, texts

    def __len__(self):
        return len(self.data_list)

    def read_rgbvideo(self, video_file, start, end):
        """Read video frames
        """
        #assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        video_datas = []
        for fid in range(start, end+1, self.interval):
            try:
                _ = os.path.exists(video_file[fid])
            except Exception as e:
                print(f"Skip frame {fid} in {video_file}. Exception {e}")
                return None
            video_data=video_file[fid]
            video_data=Image.open(video_data)
            if self.transforms:
                video_data = self.transforms(video_data)
                video_data= np.asarray(video_data, np.float32)
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32) # 4D tensor
        return video_data


    def read_foucsvideo(self, video_file, start, end):
        video_datas = []
        for fid in range(start, end + 1, self.interval):
            try:
                _ = os.path.exists(video_file[fid])
            except Exception as e:
                print(f"Skip frame {fid} in {video_file}. Exception {e}")
                return None
            video_data = video_file[fid]
            video_data = Image.open(video_data)
            video_data = np.asarray(video_data, np.float32)/255
            video_data = video_data[None, ...]
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        return video_data

    def gather_info(self, index):
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        texts=self.texts[index]
        # toa info
        start, end = self.clips[index]
        if self.labels[index] > 0: # positive sample
            self.labels[index]= 0,1
            assert self.toas[index] >= start and self.toas[index] <= end, "sample id: %s" % (self.data_list[index])
            toa = int((self.toas[index] - start) / self.interval)
        else:
            self.labels[index] = 1, 0
            toa = int(self.toas[index])  # negative sample

        data_info = np.array([accident_id, video_id, start, end,toa], dtype=np.int32)
        y=torch.tensor(self.labels[index], dtype=torch.float32)
        data_info=torch.tensor(data_info)
        return data_info,y,texts


    def __getitem__(self, index):
        # clip start and ending
        start, end = self.clips[index]
        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path, self.phase, 'rgb_videos', self.data_list[index])
        video_path=glob.glob(video_path+'/'+"*.jpg")
        video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        video_data = self.read_rgbvideo(video_path, start, end)
        #read focus video
        focus_path = os.path.join(self.root_path, self.phase, 'focus_videos', self.data_list[index])
        focus_path = glob.glob(focus_path + '/' + "*.jpg")
        focus_path = sorted(focus_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        focus_data = self.read_foucsvideo(focus_path, start, end)
        data_info,y,texts= self.gather_info(index)
        return  video_data, focus_data, data_info,y,texts


class DADA2K(Dataset):
    def __init__(self, root_path, phase, interval,transform,
                  data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms= transform
        self.data_aug = data_aug
        self.fps = 30
        self.num_classes = 2
        self.data_list, self.labels, self.clips, self.toas ,self.texts = self.get_data_list()

    def get_data_list(self):
        with open(os.path.join(self.root_path, "DADA2K_my_split", self.phase + '.txt'), 'r') as file:
            clip_names = [line.rstrip() for line in file]

        df = pd.read_csv(os.path.join(self.root_path, "annotation", "full_anno.csv"))

        fileIDs = clip_names
        labels = []
        clips = []
        toas = []
        texts = ["a video frame of { }" for _ in range(len(clip_names))]

        # TODO DEB
        clip_names = clip_names[:2]

        for clip in clip_names:
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            row = row.iloc[0]

            with zipfile.ZipFile(os.path.join(self.root_path, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1]==".png"])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            if_acc_video = int(row["whether an accident occurred (1/0)"])
            st = int(row["abnormal start frame"])
            en = int(row["abnormal end frame"])
            if st > -1 and en > -1:
                binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
            else:
                binary_labels = [0 for _ in timesteps]
            labels.append(binary_labels)
            clips.append([timesteps[0], timesteps[-1]])
            toas.append(int(row["accident frame"]))

        return fileIDs, labels, clips, toas, texts

    def __len__(self):
        return len(self.clips)


    def read_rgbvideo(self, video_file, start, end):
        """Read video frames
        """
        #assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        video_datas = []
        for fid in range(start, end+1, self.interval):
            try:
                _ = os.path.exists(video_file[fid])
            except Exception as e:
                print(f"Skip frame {fid} in {video_file}. Exception {e}")
                return None
            video_data=video_file[fid]
            video_data=Image.open(video_data)
            if self.transforms:
                video_data = self.transforms(video_data)
                video_data= np.asarray(video_data, np.float32)
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32) # 4D tensor
        return video_data

    def read_zip(self, video_path, start, end):
        filenames = [f"{str(ts).zfill(4)}.png" for ts in range(start, end+1)]
        video_datas = []
        with zipfile.ZipFile(os.path.join(video_path, "images.zip"), 'r') as zipf:
            for fname in filenames:
                data = zipf.read(fname)
                image_file = BytesIO(data)
                video_data = Image.open(image_file)
                if self.transforms:
                    video_data = self.transforms(video_data)
                    video_data = np.asarray(video_data, np.float32)
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32) # 4D tensor
        return video_data


    def read_foucsvideo_raw(self, video_file, start, end):
        filenames = [f"{str(ts).zfill(4)}.png" for ts in range(start, end + 1)]
        video_datas = []
        with zipfile.ZipFile(os.path.join(os.path.join(video_file, "fixation.zip")), 'r') as zipf:
            for fname in filenames:
                data = zipf.read(fname)
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                video_data = img.astype(np.float32)
                video_data = video_data[None, ...]
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        return video_data

    def gather_info(self, index):
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        # toa info
        start, end = self.clips[index]
        toa = int(self.toas[index])  # negative sample
        data_info = np.array([accident_id, video_id, start, end, toa], dtype=np.int32)
        texts = self.texts[index]
        y=torch.tensor(self.labels[index], dtype=torch.float32)
        data_info=torch.tensor(data_info)
        # my ttc
        ttc = compute_time_vector(y, fps=self.fps, TT=2., TA=1.)
        return data_info,y,texts,ttc


    def __getitem__(self, index):
        # clip start and ending
        start, end = self.clips[index]
        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path, "frames", self.data_list[index])
        video_data = self.read_zip(video_path, start, end)
        #read focus video
        focus_path = os.path.join(self.root_path, "frames", self.data_list[index])
        focus_data = self.read_foucsvideo_raw(focus_path, start, end)
        data_info, y, texts, ttc = self.gather_info(index)
        extra_info = {"data_info": data_info, "ttc": ttc}
        return video_data, focus_data, extra_info, y, texts


class DoTA(Dataset):
    def __init__(self, root_path, phase, interval,transform,
                  data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms= transform
        self.data_aug = data_aug
        self.fps = 10
        self.num_classes = 2
        self.data_list, self.labels, self.clips, self.ttcs , self.texts, self.acc_ids = self.get_data_list()

    def get_data_list(self):
        split_file = "val_split" if self.phase in ("testing", "validation") else "train_split"
        with open(os.path.join(self.root_path, "dataset", split_file + '.txt'), 'r') as file:
            clip_names = [line.rstrip() for line in file]

        fileIDs = clip_names
        labels = []
        clips = []
        ttcs = []
        texts = ["a video frame of { }" for _ in range(len(clip_names))]
        acc_ids = []

        for clip in clip_names:
            clip_anno_path = os.path.join(self.root_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted(
                    [int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                     in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
                acc_id = int(anno["accident_id"])
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc = compute_time_vector(binary_labels, fps=self.fps, TT=2., TA=1.)

            labels.append(binary_labels)
            clips.append([timesteps[0], timesteps[-1]])
            ttcs.append(ttc)
            acc_ids.append(acc_id)

        return fileIDs, labels, clips, ttcs, texts, acc_ids

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # clip start and ending
        start, end = self.clips[index]
        # read RGB video (trimmed)
        video_data = self.read_zip(self.data_list[index], start, end)
        #read focus video
        focus_data = self.empty_foucsvideo(nb=video_data.shape[0])
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        texts = self.texts[index]
        ttc = self.ttcs[index]
        data_info = torch.tensor(np.array([self.acc_ids[index], index, start, end, -1], dtype=np.int32))
        extra_info = {"data_info": data_info, "ttc": ttc}
        return video_data, focus_data, extra_info, y, texts

    def read_zip(self, clip_name, start, end):
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in range(start, end+1)]
        video_datas = []
        with zipfile.ZipFile(os.path.join(self.root_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                data = zipf.read(fname)
                image_file = BytesIO(data)
                video_data = Image.open(image_file)
                if self.transforms:
                    video_data = self.transforms(video_data)
                    video_data = np.asarray(video_data, np.float32)
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        #view = np.stack(view, axis=0)
        return video_data

    def empty_foucsvideo(self, nb):
        video_data = np.zeros((nb, 1, 64, 64), dtype=np.float32)  # 4D tensor
        return video_data


if __name__=="__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    # device = torch.device('cuda:0')
    num_epochs = 50
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # learning_rate = 0.0001
    batch_size = 1
    shuffle = True
    pin_memory = True
    num_workers = 1
    rootpath = r'G:\full-test'
    frame_interval = 1
    input_shape = [224, 224]
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # transform_dict = {'image': transforms.Compose([ProcessImages(input_shape)])}
    train_data = DADA(rootpath, 'training', interval=1,transform=transform )

    # val_data = DADA2KS(rootpath, 'testing', interval=1,transform=transforms

    traindata_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False ,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    for video_data, focus_data, data_info,y,texts in traindata_loader:
        if video_data.shape[1]==150:
            print(video_data.shape,data_info[0:2],texts )
        else:
            print("True")
            print(data_info[0:2],texts)


