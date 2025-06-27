import os
import pandas as pd
import random
from collections import defaultdict, OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from .utils import MultiLabelDatum, MultiLabelDatasetBase, read_json, write_json, build_data_loader, listdir_nohidden

labels = []

class_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

templates = {
    "Grasper" : "I use grasper tor cautery forcep to grasp it",
    "Bipolar" : "I use bipolar to coagulate and clean the bleeding",
    "Hook" : "I use hook to dissect it",
    "Scissors" : "I use scissors",
    "Clipper" : "I use clipper to clip it",
    "Irrigator" : "I use irrigator to suck it",
    "SpecimenBag" : "I use specimenbag to wrap it",
}

class Cholec80(MultiLabelDatasetBase):
    def __init__(self, config):
        self.dataset_dir = config["dataset_root"]
        self.image_dir = os.path.join(self.dataset_dir, "frames")
        self.tool_dir = os.path.join(self.dataset_dir, "tool_annotations")
        train_video = [f"video{video_idx:02d}" for video_idx in range(1, 41)]

        test_video = [f"video{video_idx:02d}" for video_idx in range(61, 71)]

        self.templates = templates
        self.class_names = class_names

        print("Preparing training dataset")
        train, val = self.read_data(train_video, "train")
         
        print("Preparing testing dataset")
        test,_ = self.read_data(test_video, "test")

        super().__init__(train_x=train, val=val, test=test, class_names=class_names, num_classes=len(class_names))

    def read_data(self, video_idx, split):
        items = []
        folders = [os.path.join(self.image_dir, video_dir) for video_dir in video_idx]
        data_count = 0

        for folder in folders:
            impaths = listdir_nohidden(folder)
            video_dir = folder.split("/")[-1]
            tool_filepath = f"{video_dir}-tool.txt"
            labels_df = pd.read_csv(os.path.join(self.tool_dir, tool_filepath), sep = '\t')
            for impath in impaths:
                frame_idx = int(impath.split(".")[-2].split("_")[-1]) - 1
                labels = []
                classnames = []
                for idx, col in enumerate(labels_df.columns):
                    if labels_df.loc[frame_idx, col ] == 1:
                        labels.append(idx-1)
                        classnames.append(col)
                item = MultiLabelDatum(impath=os.path.join(folder, impath), labels=labels, classnames=classnames)
                items.append(item)
                data_count += 1
        
        print(f"{split} has {data_count} pieces of data")
        
        if split == "train":
            random.shuffle(items)
            print(f"{len(items[:int(data_count/2)])} for training")
            print(f"{len(items[:int(data_count/2)])} for validation")
            return items[:int(data_count/2)], items[int(data_count/2):]
        
        if split == "test":
            return items, items

    def generate_fewshot_dataset_(self, num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
    
        return few_shot_data