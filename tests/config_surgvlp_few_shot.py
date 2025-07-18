"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""

config = {
    "zero_shot" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = True,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 1,
    ),
    "linear_probe" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = True,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 5,
        num_shots = 1,
        lr = 0.001,
        epochs = 30,
        unfreeze = False,
        unfreeze_layer = "last",
        csv_path = "results.csv",
    ),
    "linear_probe++" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 1,
        num_shots = 1,
        lr = 0.001,
        epochs = 1,
        unfreeze = False,
        unfreeze_layer = "last",
        init_alpha = 0.2,
    ),
    "zoom_in" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 5,
        num_shots = 256,
        lr = 0.001,
        epochs = 30,
        unfreeze = True,
        unfreeze_layer = "last",
    ),
    "bi_cross_attn" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True, # change to True after first run
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 5,
        num_shots = 64,
        lr = 0.001,
        epochs = 30,
        unfreeze = True,
        unfreeze_layer = "last",
        csv_path = "results.csv",
        checkpoint_path = "checkpoints/cross_attn.pth",
        train_mode = "training", # inference
    ),
    "residual_bi_cross_attn" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True, # change to True after first run
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 3,
        num_shots = 1,
        lr = 0.001,
        epochs = 30,
        unfreeze = True,
        unfreeze_layer = "last",
        csv_path = "results.csv",
    ),
    "negation" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 3,
        num_shots = 128,
        lr = 0.001,
        epochs = 100,
        unfreeze_vision = True,
        unfreeze_vision_layer = "last",
        unfreeze_text = True,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation.pth",
    ),
    "negation_nce" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 40,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.00001,
        annealling = True,
        tasks = 1,
        num_shots = 64,
        epochs = 30,
        unfreeze_vision = True,
        # unfreeze_vision_layer = "last",
        unfreeze_text = True,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation_nce.pth",
    ),
    "mixture" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.00001,
        annealling = True,
        tasks = 1,
        num_shots = 64,
        epochs = 30,
        unfreeze_vision = True,
        unfreeze_text = True,
        csv_path = "results_mixture.csv",
        checkpoint_path = "checkpoints/mixture.pth",
    ),
    "aggre_negation" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
            sample_negated_num = 2,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        learning_rate = 0.001,
        tasks = 3,
        num_shots = 1,
        lr = 0.001,
        epochs = 30,
        unfreeze_vision = True,
        unfreeze_vision_layer = "last",
        unfreeze_text = True,
        csv_path = "results_negation.csv",
    ),
    "simple" : dict(
        dataset_config = dict(
            dataset_root = "/home/yongxuan/datasets/cholec80",
            num_classes = 7,
        ),
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
        ),
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 3,
        num_shots = 2,
        lr = 0.001,
        epochs = 10,
        unfreeze_vision = True,
        unfreeze_vision_layer = "last",
        unfreeze_text = True,
        csv_path = "results_negation.csv",
    ),
}