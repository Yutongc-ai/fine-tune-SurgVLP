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
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        tasks = 1,
        num_shots = -1,
        csv_path = "results_zero_shot.csv",
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
        attention_pooling = False,
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        batch_size = 64,
        accumulate_step = 1,
        learning_rate = 0.01,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        annealling = True,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_lp.csv",
        checkpoint_path = "checkpoints/linear_probe.pth",
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
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        tasks = 3,
        num_shots = 5000,
        lr = 0.001,
        epochs = 50,
        annealling = True,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_lp++.csv",
        checkpoint_path = "checkpoints/linear_probe++.pth",
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
        accumulate_step = 1,
        learning_rate = 0.0001,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        early_stop = True,
        patience = 10,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_cross_attn.csv",
        checkpoint_path = "checkpoints/cross_attn.pth",
    ),
    "ours" : dict(
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
        tasks = 3,
        num_shots = 256,
        epochs = 10,
        unfreeze_vision = False,
        unfreeze_text = True,
        eval_interval = 2,
        avg_freq = 1,
        csv_path = "results_ours.csv",
        checkpoint_path = "checkpoints/ours.pth",
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        alpha = 0.2,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation.pth",
    ),
    "normal_finetune" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        alpha = 0.2,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation.pth",
    ),
    "negation_mul" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        alpha = 0.2,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation_nce.pth",
    ),
    "negation_maf" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        alpha = 0.2,
        csv_path = "results_negation_maf.csv",
        checkpoint_path = "checkpoints/negation_maf.pth",
    ),
    "negation_nce_dir" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 10,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01,
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        alpha = 0.2,
        csv_path = "results_negation_dir.csv",
        checkpoint_path = "checkpoints/negation_nce_dir.pth",
    ),
    "weighted_negation_nce" : dict(
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
        tasks = 3,
        num_shots = 256,
        epochs = 10,
        unfreeze_vision = False,
        unfreeze_text = True,
        csv_path = "results_negation.csv",
        checkpoint_path = "checkpoints/negation_nce.pth",
    ),
    "negation_nce_all" : dict(
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
        tasks = 3,
        num_shots = -1,
        epochs = 20,
        unfreeze_vision = False,
        unfreeze_text = True,
        csv_path = "results_negation_all.csv",
        checkpoint_path = "checkpoints/negation_nce_all.pth",
    ),
    "finetune" : dict(
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
        tasks = 3,
        num_shots = -1,
        epochs = 20,
        unfreeze_vision = False,
        unfreeze_text = True,
        csv_path = "results_finetune.csv",
        checkpoint_path = "checkpoints/finetune.pth",
    ),
    "part_negation_nce" : dict(
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
        tasks = 3,
        num_shots = 256,
        epochs = 10,
        unfreeze_vision = False,
        unfreeze_text = True,
        csv_path = "results_part_negation.csv",
        checkpoint_path = "checkpoints/part_negation_nce.pth",
    ),
    "clip_adapter" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 100,
        early_stop = True,
        batch_size = 64,
        accumulate_step = 1,
        learning_rate = 0.01, # follow few-shot-medVLM
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        search_alpha_ca = False,
        alpha_ca = 0.5,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_clip_adapter.csv",
        checkpoint_path = "checkpoints/clip_adapter.pth",
    ),
    "tip_adapter" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 100,
        early_stop = True,
        batch_size = 64,
        accumulate_step = 1,
        learning_rate = 0.01, # follow few-shot-medVLM
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        alpha = 1,
        beta = 1,
        unfreeze_vision = False,
        unfreeze_text = False,
        finetune = False,
        search_hp = True,
        search_scale = [100, 100],
        search_step = [10, 10],
        csv_path = "results_tip_adapter.csv",
        checkpoint_path = "checkpoints/tip_adapter.pth",
    ),
    "coop" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 100,
        early_stop = True,
        batch_size = 64,
        accumulate_step = 1,
        learning_rate = 0.01, # follow few-shot-medVLM
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_coop.csv",
        checkpoint_path = "checkpoints/coop.pth",
    ),
    "cocoop" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 100,
        early_stop = True,
        batch_size = 8,
        accumulate_step = 8,
        learning_rate = 0.01, # follow few-shot-medVLM
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_cocoop.csv",
        checkpoint_path = "checkpoints/cocoop.pth",
    ),
    "dual_coop" : dict(
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
        preload_local_features = True,
        cache_dir = "/home/yongxuan/SurgVLP/cache",
        patience = 100,
        early_stop = True,
        batch_size = 64,
        accumulate_step = 1,
        learning_rate = 0.01, # follow few-shot-medVLM
        annealling = True,
        tasks = 3,
        num_shots = 5000,
        epochs = 50,
        unfreeze_vision = False,
        unfreeze_text = False,
        csv_path = "results_dual_coop.csv",
        checkpoint_path = "checkpoints/dual_coop.pth",
    ),
}
