import torch
import random
import torch.nn.functional as F
import pytorchvideo.data
from typing import List
from torch.utils.data import Dataset, DataLoader
from glob import glob
import  pandas as pd
from .dataframe_constructor import label_df_constructor
import numpy as np
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

class SequentialRandomSampler:
    def __init__(self, n=4, segments=4):
        self.segments = segments
        self.n = n
    
    def sample_indices(self, num_frames:int) -> List: 
        segment_size = num_frames // self.segments
        segments = []
        
        for i in range(self.segments):
            start = i * segment_size
            end = start + segment_size
            segment = list(range(start,end))
            if segment:
                random_index = random.randint(0, len(segment) - self.n)
                selected_items = segment[random_index:random_index + self.n]
                segments.extend(selected_items)
        return segments
    
    def __call__(self, video:torch.Tensor):
        num_frames = video.shape[1]
        if num_frames > (self.n * self.segments):
            samples = self.sample_indices(num_frames)
            return video[:,samples,...]
        return video
class SequentialRandomSampler:
    def __init__(self, n=4, segments=4):
        self.segments = segments
        self.n = n
    
    def sample_indices(self, num_frames:int) -> List: 
        segment_size = num_frames // self.segments
        segments = []
        
        for i in range(self.segments):
            start = i * segment_size
            end = start + segment_size
            segment = list(range(start,end))
            if segment:
                random_index = random.randint(0, len(segment) - self.n)
                selected_items = segment[random_index:random_index + self.n]
                segments.extend(selected_items)
        return segments
    
    def __call__(self, video:torch.Tensor):
        num_frames = video.shape[1]
        if num_frames > (self.n * self.segments):
            samples = self.sample_indices(num_frames)
            return video[:,samples,...]
        return video

class dataset(Dataset):
    def __init__(self,
                 datasets:list=["b4c","hdd", "oxford"],
                 split_type="train",
                 resize_to = (224,224),
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 fps=30,
                 b4c_fold=0,
                 oxford_fold=0,
                 multi_modal=False,
                 action_rec=False,
                 multi_labels=False,
                 overlapping_labels=True,
                 minvid_length=16,
                 label2id=None,
                 id2label=None,
                 num_classes=None,
                ):
        
        self.split_type = split_type
        self.included_datasets = datasets
        self.multi_modal = multi_modal
        self.fps = fps
        self.df = label_df_constructor(datasets=datasets,
                                       b4c_fold=b4c_fold, oxford_fold=oxford_fold,
                                       multimodal=multi_modal, action_rec=action_rec,
                                       multi_labels=multi_labels, minvid_length=minvid_length,
                                       overlapping_labels=overlapping_labels, # overlapping labels between datasets
                                       split_type=split_type)
        self.b4c_fold = b4c_fold
        self.oxford_fold = oxford_fold
        self.action_rec = action_rec
        self.dataset_list = self.df.exterior_video_file.tolist()
        
        self.id2label = id2label 
        self.label2id = {i: label for label, i in self.id2label.items()}
        self.num_classes = num_classes

        self.transform = Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        SequentialRandomSampler(),     
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        RandomShortSideScale(min_size=256, max_size=320),
                                        RandomCrop(resize_to),
                                    ]),),])
        if split_type == "test":
            self.transform = Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        UniformTemporalSubsample(16),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        Resize(resize_to),
                                    ]),),])
        
        self.flip_transform = RandomHorizontalFlip(p=1.0)
    
    def __len__(self):
        return len(self.dataset_list)
    
    def load_video(self, video_path, model_frame_length=16):
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        frames = self.fps * video.duration.numerator 
        end_sec = frames/self.fps + 10
        video= video.get_clip(start_sec=0, end_sec=end_sec)
        frames = video["video"].shape[1]
        if frames < model_frame_length:
            required_frames = model_frame_length - frames
            padding_frames = torch.stack([torch.zeros_like(video["video"][:,:1,...]) 
                                          for _ in range(required_frames)], dim=1)
            if len(padding_frames.shape) ==5:
                padding_frames = padding_frames.squeeze()
            if len(padding_frames.shape) ==3:
                padding_frames = padding_frames.unsqueeze(dim=1)
            video["video"] = torch.concat([video["video"], padding_frames], dim=1)
        video = self.transform(video)
        return video["video"]
    
    def load_target(self, video_path):
        if not(self.action_rec): 
            label =  self.df.loc[self.df['exterior_video_file'] ==video_path].intent_labels.values[0]
        else:
            label = self.df.loc[self.df['exterior_video_file'] ==video_path].action_labels.values[0]
        label =  self.label2id[label]
        one_hot_label = F.one_hot(torch.tensor(label), self.num_classes).float()
        return one_hot_label
    
    def flip_video(self, video):
        return torch.stack([self.flip_transform(frame) for frame in video])
        
    def label_flip(self, one_hot_tensor):
        original_label = one_hot_tensor.argmax().item()
        new_label = self.label_mapper(original_label)
        return F.one_hot(torch.tensor(new_label), self.num_classes).float() 

    def label_mapper(self, label):
        flip_dict = {
            "left_lane_change": "right_lane_change",
            "right_lane_change":"left_lane_change",
            "left_turn": "right_turn",
            "right_turn": "left_turn",
            "right_lane_branch":"left_lane_branch",
            "left_lane_branch":"right_lane_branch",
            "move_right":"move_left",
            "move_left":"move_right",
        }
        original_label = self.id2label[label]
        
        if original_label in list(flip_dict.keys()):
            flipped_label = flip_dict[original_label]
            new_label = self.label2id[flipped_label]
            return new_label
        else:
            return label

    
    def horizontal_flip(self, data_dict):
        if random.random() > 0.5:
            try: 
                data_dict["video"] = self.flip_video(data_dict["video"]) 
            except KeyError:
                pass
            try: 
                data_dict["incabin_video"] = self.flip_video(data_dict["incabin_video"]) 
            except KeyError:
                pass
            data_dict["label"] = self.label_flip(data_dict["label"])
        return data_dict
    
    def __getitem__(self, index):
        video_path = self.dataset_list[index]
        data_dict = {
            "video": self.load_video(video_path),
            "label": self.load_target(video_path)
        }
        
        if self.multi_modal and "incabin_video_file" in self.df.columns:
            try:
                incabin_video_path = self.df.set_index("exterior_video_file").loc[video_path].incabin_video_file
                data_dict["incabin_video"] = self.load_video(f"./{incabin_video_path}")
            except FileNotFoundError:
                data_dict["incabin_video"] = torch.zeros(3, 16, 224, 224)
        
        
        if self.multi_modal and ("canbus_signal_files" in self.df.columns):
            canbus_path = self.df.set_index("exterior_video_file").loc[video_path].canbus_signal_files
#             print(canbus_path)
            try:
                tensor =  torch.from_numpy(np.load(canbus_path))

                if self.split_type == "train":
                    indices = torch.randint(high=tensor.size(0), size=(16,))
                    indices = indices.sort()[0]
                    data_dict["canbus"] = tensor[indices]
                else:

                    # Use np.linspace to generate evenly spaced indices
                    indices = np.linspace(0, tensor.size(0)-1, 16)
                    # Convert the indices to integers
                    indices = np.round(indices).astype(int)
                    # Use slicing with the step size to get the subsample
                    data_dict["canbus"] = tensor[indices]
            except TypeError:
                data_dict["canbus"] = torch.zeros(16,8)

        if self.split_type == "train":
            data_dict = self.horizontal_flip(data_dict)
        
        return data_dict
