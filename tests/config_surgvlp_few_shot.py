"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torchvision.transforms as transforms

config = {
    "zero_shot_attn_pooling" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = True,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 1,
    ),
    "zero_shot_average_pooling": dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
                text_freeze_bert=True,
                text_agg_tokens=True
            )
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 1,
    ),
    "attn_pooling_lp" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = True,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 5,
        num_shots = 1,
        lr = 0.001,
        epochs = 100,
        unfreeze = True,
        unfreeze_layer = "last",
    ),
    "avg_pooling_lp" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 5,
        num_shots = 16,
        lr = 0.001,
        epochs = 100,
        unfreeze = True,
        unfreeze_layer = "last",
    ),
    "attn_pooling_lpplus" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = True,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 5,
        num_shots = 1,
        lr = 0.001,
        epochs = 200,
        unfreeze = True,
        unfreeze_layer = "last",
        init_alpha = 0.2,
    ),
    "avg_pooling_lpplus" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 3,
        num_shots = 4,
        lr = 0.001,
        epochs = 300,
        unfreeze = True,
        unfreeze_layer = "last",
        init_alpha = 0.2,
    ),
    "zoom_in" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
        model_config = dict(
            type='SurgVLP',
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 5,
        num_shots = 256,
        lr = 0.001,
        epochs = 30,
        unfreeze = True,
        unfreeze_layer = "last",
    ),
}