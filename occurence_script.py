import torch
import surgvlp
from mmengine.config import Config
import random
from datasets.cholec80 import Cholec80, NegationCholec80
from datasets.utils import *
import argparse
import datetime
from methods.utils import *
import wandb

import numpy as np
import pandas as pd
from collections import defaultdict

dataset_config = dict(
    dataset_root = "/home/yongxuan/datasets/cholec80",
    num_classes = 7,
)

model_config = dict(
    type='PeskaVLP',
    backbone_img = dict(
        type='img_backbones/ImageEncoder',
        num_classes=768,
        pretrained='imagenet',
        backbone_name='resnet_50',
        img_norm=False
    ),
    backbone_text= dict(
        type='text_backbones/BertEncoder',
        text_bert_type='emilyalsentzer/Bio_ClinicalBERT',
        text_last_n_layers=4,
        text_aggregate_method='sum',
        text_norm=False,
        text_embedding_dim=768,
        text_freeze_bert=False,
        text_agg_tokens=True
    )
)
model_path = '/home/yongxuan/SurgVLP/checkpoints/PeskaVLP.pth'

dataset = Cholec80(config=dataset_config)

surgvlp_model, preprocess = surgvlp.load(model_config, device='cuda', pretrain=model_path)

train_loader = build_data_loader(data_source=dataset.train_x, batch_size = 1, tfm = preprocess, is_train=True, 
                                 num_classes = dataset.num_classes)

# 假设你已经有了所有的工具名称列表
# all_tool_names = sorted(list(set(tool for video_frames in video_data.values() for frame_tools in video_frames.values() for tool in frame_tools)))
num_classes = 7
# tool_to_idx = {name: i for i, name in enumerate(all_tool_names)}

# 初始化共现计数矩阵
co_occurrence_counts = np.zeros((num_classes, num_classes), dtype=int)
total_frames = len(train_loader)

for i, (images, target, negated_target, _) in enumerate(tqdm(train_loader)):
    target = target.squeeze(0)
    for tool_idx in range(7):
        if target[tool_idx].item() == False:
            continue
        co_occurrence_counts[tool_idx, tool_idx] += 1
        for idx2 in range(tool_idx + 1, 7):
            # 由于共现是对称的，只填充一半即可
            if target[idx2].item() == False:
                continue
            co_occurrence_counts[tool_idx, idx2] += 1

co_occurrence_probabilities = co_occurrence_counts / total_frames

print(co_occurrence_counts)
print(co_occurrence_probabilities)

# 遍历所有视频和所有帧
# for video_id, frames_data in video_data.items():
#     for frame_id, tools_in_frame in frames_data.items():
#         total_frames += 1
#         # 将当前帧的工具列表转换为索引列表
#         current_frame_tool_indices = [tool_to_idx[tool] for tool in tools_in_frame]

#         # 遍历当前帧中所有工具对
#         for i in range(len(current_frame_tool_indices)):
#             idx1 = current_frame_tool_indices[i]
#             # 对角线上的元素表示该工具自身出现的次数
#             co_occurrence_counts[idx1, idx1] += 1
#             for j in range(i + 1, len(current_frame_tool_indices)):
#                 idx2 = current_frame_tool_indices[j]
#                 # 由于共现是对称的，只填充一半即可
#                 co_occurrence_counts[idx1, idx2] += 1
#                 co_occurrence_counts[idx2, idx1] += 1

# 可选：将计数转换为概率
# co_occurrence_probabilities = co_occurrence_counts / total_frames
