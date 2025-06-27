# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import os
import random
import torch
import torch.utils.data
import numpy as np
import json
import pickle as pkl
import re
from PIL import Image
import math
import copy
import pandas as pd
from . import utils
from ..registry import DATASETS
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

labellist = {
    'Preparation' : 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderRetraction': 4,
    'CleaningCoagulation': 5,
    'GallbladderPackaging': 6,
}

# def label2id(label) :
#     return labellist[label]

@DATASETS.register_module(name='Recognition_frame')
class CholecDataset(Dataset):
    def __init__(self, csv_root, vid, video_root, transforms=None, loader=pil_loader):
        csv_name = os.path.join(csv_root, vid)
        print(csv_name)
        df = pd.read_csv(csv_name, sep='\s+')
        print(df.columns.tolist())
        self.vid = vid
        self.video_root = video_root

        self.file_list = df['Frame'].tolist()
        self.label_list = df['Phase'].tolist()
        assert len(self.file_list) == len(self.label_list)
        self.transform = transforms
        self.loader = loader


    def __getitem__(self, index):
        f_id = self.file_list[index]
        v_id = self.vid.split('.txt')[0]
        # v_id, f_id = img_names.split('.png')[0].split('_')
        f_id = int(int(f_id-1)/25) + 1
        img_names = os.path.join(self.video_root, v_id, v_id+'_'+f'{f_id:06d}'+'.png')
        imgs = self.loader(img_names)

        labels_phase = self.label_list[index]

        # print(imgs.size)
        if self.transform is not None:
            imgs = self.transform(imgs)

        final_dict = {'video':imgs, 'label': labellist[labels_phase]} # the label id starts from 1
        return final_dict

    def __len__(self):
        return len(self.file_list)