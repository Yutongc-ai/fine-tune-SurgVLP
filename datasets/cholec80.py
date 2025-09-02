import os
import pandas as pd
import random
import numpy as np
from .utils import MultiLabelDatum, MultiLabelDatasetBase, read_json, write_json, build_data_loader, listdir_nohidden

labels = []

class_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
class_names_extend = ["grasper to cautery forcep to grasp it",
                      "bipolar to coagulate and clean the bleeding",
                      "hook to dissect it",
                      "scissors",
                      "clipper to clip it",
                      "irrigator to suck it",
                      "specimenbag to wrap it"]

templates = {
    "Grasper" : "I use grasper to cautery forcep to grasp it",
    "Bipolar" : "I use bipolar to coagulate and clean the bleeding",
    "Hook" : "I use hook to dissect it",
    "Scissors" : "I use scissors",
    "Clipper" : "I use clipper to clip it",
    "Irrigator" : "I use irrigator to suck it",
    "SpecimenBag" : "I use specimenbag to wrap it",
}

phase_templates = {
    "Preparation": "In preparation phase I insert trocars to patient abdomen cavity",
    "CalotTriangleDissection": "In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic triangle area and cystic duct and cystic artery",
    "ClippingCutting": "In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them",
    "GallbladderDissection": "In dissection phase I use the hook to dissect the onnective tissue between gallbladder and liver",
    "GallbladderPacking": "In packaging phase I put the gallbladder into the specimen bag",
    "CleaningCoagulation": "In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels",
    "GallbladderRetraction": "In retraction phase I grasp the specimen bag and remove it from trocar",
}

all_labels_set = set([0, 1, 2, 3, 4, 5, 6])

negated_templates = {
    "Grasper" : "I did not use grasper",
    "Bipolar" : "I did not use bipolar",
    "Hook" : "I did not use hook",
    "Scissors" : "I did not use scissors",
    "Clipper" : "I did not use clipper",
    "Irrigator" : "I did not use irrigator",
    "SpecimenBag" : "I did not use specimenbag",
}

negated_templates_1 = {
    "Grasper" : "I use grasper to coagulate and clean the bleeding",
    "Bipolar" : "I use bipolar to dissect it",
    "Hook" : "I use hook to clip it",
    "Scissors" : "I use scissors to clip it",
    "Clipper" : "I use clipper to suck it",
    "Irrigator" : "I use irrigator to wrap it",
    "SpecimenBag" : "I use specimenbag to cautery forcep to grasp it",
}

negated_templates_2 = {
    "Grasper" : "I use bipolar to coagulate and clean the bleeding but I don't use grasper to cautery forcep to grasp it",
    "Bipolar" : "I use hook to dissect it but I don't use bipolar to coagulate and clean the bleeding",
    "Hook" : "I use scissors but I don't use hook to dissect it",
    "Scissors" : "I use clipper to clip it but I don't use scissors",
    "Clipper" : "I use irrigator to suck it but I don't use clipper to clip it",
    "Irrigator" : "I use specimenbag to wrap it but I don't use irrigator to suck it",
    "SpecimenBag" : "I use grasper to cautery forcep to grasp it but I don't use specimenbag to wrap it",
}

negated_templates_3 = "I use {} but I don't use {}"

class Cholec80(MultiLabelDatasetBase):
    def __init__(self, config):
        self.dataset_dir = config["dataset_root"]
        # self.sample_negated_num = config["sample_negated_num"]

        self.image_dir = os.path.join(self.dataset_dir, "frames")
        self.tool_dir = os.path.join(self.dataset_dir, "tool_annotations")
        self.phase_dir = os.path.join(self.dataset_dir, "phase_annotations")

        train_video = [f"video{video_idx:02d}" for video_idx in range(1, 41)]
        val_video = [f"video{video_idx:02d}" for video_idx in range(41, 61)]

        test_video = [f"video{video_idx:02d}" for video_idx in range(61, 81)]

        self.templates = list(templates.values())
        self.negated_templates = list(negated_templates.values())
        self.negated_templates_1 = list(negated_templates_1.values())
        self.negated_templates_2 = list(negated_templates_2.values())
        self.phase_templates = list(phase_templates.values())

        self.class_names = class_names
        self.class_names_extend = class_names_extend

        print("Preparing training dataset")
        train = self.read_data(train_video, "train")

        print("Preparing validation dataset")
        val = self.read_data(val_video, "val")
         
        print("Preparing testing dataset")
        test = self.read_data(test_video, "test")

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
            phase_filepath = f"{video_dir}-phase-cls.txt"
            phase_labels_df = pd.read_csv(os.path.join(self.phase_dir, phase_filepath), sep = '\t')
            for impath in impaths:
                frame_idx = int(impath.split(".")[-2].split("_")[-1]) - 1
                labels = []
                phase_labels = []
                classnames = []
                for idx, col in enumerate(labels_df.columns):
                    if labels_df.loc[frame_idx, col ] == 1:
                        labels.append(idx-1)
                        classnames.append(col)
                
                for idx, col in enumerate(phase_labels_df.columns):
                    if phase_labels_df.loc[frame_idx*25, col ] == 2:
                        phase_labels.append(idx)
                        break
                
                # sample negated labels
                # all_values = np.arange(len(self.class_names))
                # mask = np.isin(all_values, labels, invert=True)
                # available_values = all_values[mask]
                # negated_labels = np.random.choice(available_values, size=self.sample_negated_num, replace=False)
                negated_labels = all_labels_set - set(labels)
                item = MultiLabelDatum(impath=os.path.join(folder, impath), labels=labels, phase_labels=phase_labels, negated_labels=list(negated_labels), classnames=classnames)
                items.append(item)
                data_count += 1
        
        print(f"{split} has {data_count} pieces of data")
        
        if split == "train":
            random.shuffle(items)
            # print(f"{len(items[:int(data_count/2)])} for training")
            # print(f"{len(items[:int(data_count/2)])} for validation")
            # return items[:int(data_count/2)], items[int(data_count/2):]
            return items
    
        return items

    def generate_fewshot_dataset_(self, num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
    
        return few_shot_data

class CholecPhase(MultiLabelDatasetBase):
    def __init__(self, config):
        self.dataset_dir = config["dataset_root"]
        # self.sample_negated_num = config["sample_negated_num"]

        self.image_dir = os.path.join(self.dataset_dir, "frames")
        self.tool_dir = os.path.join(self.dataset_dir, "tool_annotations")
        train_video = [f"video{video_idx:02d}" for video_idx in range(1, 41)]
        val_video = [f"video{video_idx:02d}" for video_idx in range(41, 45)]

        test_video = [f"video{video_idx:02d}" for video_idx in range(61, 71)]

        self.templates = list(templates.values())
        self.negated_templates = list(negated_templates.values())
        self.negated_templates_1 = list(negated_templates_1.values())
        self.negated_templates_2 = negated_templates_2
        self.negated_templates_3 = negated_templates_3

        self.class_names = class_names

        print("Preparing training dataset")
        train = self.read_data(train_video, "train")

        print("Preparing validation dataset")
        val = self.read_data(val_video, "val")
         
        print("Preparing testing dataset")
        test = self.read_data(test_video, "test")

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
                
                # sample negated labels
                # all_values = np.arange(len(self.class_names))
                # mask = np.isin(all_values, labels, invert=True)
                # available_values = all_values[mask]
                # negated_labels = np.random.choice(available_values, size=self.sample_negated_num, replace=False)
                negated_labels = all_labels_set - set(labels)
                item = MultiLabelDatum(impath=os.path.join(folder, impath), labels=labels, negated_labels=list(negated_labels), classnames=classnames)
                items.append(item)
                data_count += 1
        
        print(f"{split} has {data_count} pieces of data")
        
        if split == "train":
            random.shuffle(items)
            # print(f"{len(items[:int(data_count/2)])} for training")
            # print(f"{len(items[:int(data_count/2)])} for validation")
            # return items[:int(data_count/2)], items[int(data_count/2):]
            return items
    
        return items

    def generate_fewshot_dataset_(self, num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
    
        return few_shot_data
    
class NegationCholec80(MultiLabelDatasetBase):
    def __init__(self, config):
        self.dataset_dir = config["dataset_root"]
        
        if "sample_negated_num" in config:
            self.sample_negated_num = config["sample_negated_num"]
        else:
            self.sample_negated_num = None

        self.image_dir = os.path.join(self.dataset_dir, "frames")
        self.tool_dir = os.path.join(self.dataset_dir, "tool_annotations")
        train_video = [f"video{video_idx:02d}" for video_idx in range(1, 41)]
        val_video = [f"video{video_idx:02d}" for video_idx in range(41, 45)]

        test_video = [f"video{video_idx:02d}" for video_idx in range(61, 71)]

        self.templates = list(templates.values())
        self.negated_templates = list(negated_templates.values())

        self.class_names = class_names

        print("Preparing training dataset")
        train = self.read_data(train_video, "train")

        print("Preparing validation dataset")
        val = self.read_data(val_video, "val")
         
        print("Preparing testing dataset")
        test = self.read_data(test_video, "test")

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
                
                # sample negated labels
                assert(self.sample_negated_num is not None)

                all_values = np.arange(len(self.class_names))
                mask = np.isin(all_values, labels, invert=True)
                available_values = all_values[mask]
                negated_labels = np.random.choice(available_values, size=self.sample_negated_num, replace=False)

                item = MultiLabelDatum(impath=os.path.join(folder, impath), labels=labels, classnames=classnames, negated_labels=negated_labels)
                items.append(item)
                data_count += 1
        
        print(f"{split} has {data_count} pieces of data")
        
        if split == "train":
            random.shuffle(items)
            # print(f"{len(items[:int(data_count/2)])} for training")
            # print(f"{len(items[:int(data_count/2)])} for validation")
            # return items[:int(data_count/2)], items[int(data_count/2):]
            return items
    
        return items

    def generate_fewshot_dataset_(self, num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
    
        return few_shot_data