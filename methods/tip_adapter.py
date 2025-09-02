import time
from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features, Cholec80FeaturesVal
from methods.utils import multilabel_metrics, search_hp_tip, ConstantWarmupScheduler
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

class TIPAdapter(nn.Module):
    '''
    TIP Adapter and Tip-Adapter-F methods
    '''
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs

        # =====================================Training Settings======================================
        self.lr = configs.learning_rate
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.accumulate_step = configs.accumulate_step
        self.patience = configs.patience
        self.checkpoint_path =  "checkpoints/best_tipAdapter_" + str(self.configs.num_shots) + "shots.pt"
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        self.annealling = configs.annealling

        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes

        self.finetune = configs['finetune']
        self.init_alpha = configs['alpha']
        self.init_beta = configs['beta']
        self.scale_logits = 100

        # ======================================Training Parameters=====================================
        # metrcis
        self.criterion = torch.nn.BCEWithLogitsLoss()

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False

    def get_metrics(self, split):
        # total_loss = 0.0
        all_probs = []
        all_logits = []
        all_labels = []

        if split == "val":
            feature_loader = self.val_feature
        elif split == "test":
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")

        with torch.no_grad():

            for _ in range(len(feature_loader)):
                global_image_features, _, label, _ = feature_loader[_]
                global_image_features = global_image_features.to(device)
                global_image_features = global_image_features / global_image_features.norm(dim = -1, keepdim = True)

                logits = self.scale_logits * global_image_features @ self.feats_templates.T 
                prob = logits.sigmoid()

                all_probs.append(prob)
                all_logits.append(logits)
                all_labels.append(label)

        final_logits = torch.cat(all_logits, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics, final_logits, final_labels
    
    def get_metrics_finetune(self, split, alpha, beta):
        all_probs = []
        # all_logits = []
        all_labels = []

        if split == "val":
            feature_loader = self.val_feature
        elif split == "test":
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")

        with torch.no_grad():

            for _ in range(len(feature_loader)):
                global_image_features, _, label = feature_loader[_]
                global_image_features = global_image_features.to(device)
                global_image_features = global_image_features / global_image_features.norm(dim = -1, keepdim = True)
                
                affinity = self.adapter(global_image_features)
                # print("cache values")
                # print(self.cache_values)
                # print("===========================")
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values.to(affinity.dtype)

                clip_logits = self.scale_logits * global_image_features @ self.feats_templates.T
                # print(clip_logits)
                # print("=======================================")
                # print(cache_logits)
                # assert(0)
                logits = clip_logits + cache_logits * alpha

                prob = logits.sigmoid()

                all_probs.append(prob)
                # all_logits.append(logits)
                all_labels.append(label)

        # final_logits = torch.cat(all_logits, dim=0).to('cpu')
        final_labels = torch.cat(all_labels, dim=0).to('cpu')
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def get_affinity(self, split):
        if split == 'val':
            feature_loader = self.val_feature
        elif split == 'test':
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")
        
        with torch.no_grad():
            affinity_list = []
            for idx in range(len(feature_loader)):
                global_image_features, _, label, _ = feature_loader[idx]
                global_image_features = global_image_features.to(device)
                global_image_features = global_image_features / global_image_features.norm(dim = -1, keepdim = True)

                # self.cache_key: [768, num_class * num_shots]
                affinity_list.append(global_image_features @ self.cache_keys)
            
            affinity = torch.cat(affinity_list)
        
        return affinity

    def forward(self,
                dataset: MultiLabelDatasetBase):
        
        wandb.init(
            project=f"tip-adapter-{self.configs.model_config.type}",
            name=f"batchsize{self.batch_size * self.accumulate_step}_lr{self.lr}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
        )

        # ===========================generate text weights=============================
        templates = dataset.templates

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        input_ids = self.templates['input_ids']
        token_type_ids = self.templates['token_type_ids']
        attention_masks = self.templates['attention_mask']
        
        with torch.no_grad():
            _, self.feats_templates, _ = self.model.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            self.feats_templates = self.feats_templates.to(device)

        # ===========================build dataloader=============================
        # Creating dataset loader
        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)
        
        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80FeaturesVal(self.configs, "val")

        print(len(self.test_feature))
        print(len(self.val_feature))

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")

        train_loader = build_data_loader(data_source=train_data, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)
        
        # ==========================build cache model============================

        self.cache_keys, self.cache_values = build_cache_model(self.configs, self.model, train_loader)
        # print(cache_keys.shape) [768, num_shot*num_class]
        # print(cache_values.shape) [num_shot*num_class, 7(num_classes:one hot)]
        beta, alpha = self.init_beta, self.init_alpha

        start_time = time.time()
        # ===========validation dataset================
        # Zero-shot CLIP
        zero_shot_val_metrics, val_clip_logits, val_labels = self.get_metrics("val")
        zero_shot_test_metrics, test_clip_logits, test_labels = self.get_metrics("test")

        print("\n**** Zero-shot CLIP's val mAP: {:.4f}. ****\n".format(zero_shot_val_metrics["mAP"]))
        print("\n**** Zero-shot CLIP's test mAP: {:.4f}. ****\n".format(zero_shot_test_metrics['mAP']))
        
        # Tip-Adapter
        val_affinity = self.get_affinity("val") # affinity: [val_len, num_class * num_shots]
        cache_logits = ((-1) * (beta - beta * val_affinity)).exp() @ self.cache_values.to(val_affinity.dtype)
        # cache_logits = beta * affinity @ cache_values
        
        tip_logits = (val_clip_logits + cache_logits * alpha) / (1 + alpha)
        tip_prob = tip_logits.sigmoid().cpu()
        val_metric = multilabel_metrics(val_labels, tip_prob)
        print("**** Tip-Adapter's val mAP: {:.4f}. ****\n".format(val_metric["mAP"]))

        # Search Hyperparameters
        best_beta, best_alpha = search_hp_tip(self.configs, val_affinity, val_clip_logits, self.cache_values, val_labels)
        
        self.configs["alpha"] = best_alpha
        self.configs["beta"] = best_beta

        if not self.finetune:
            # ===========test dataset=====================
            # Tip-Adapter
            test_affinity = self.get_affinity("test")
            cache_logits = ((-1) * (best_beta - best_beta * test_affinity)).exp() @ self.cache_values.to(test_affinity.dtype)
            
            tip_logits = (test_clip_logits + cache_logits * best_alpha) / (1 + alpha)
            tip_prob = tip_logits.sigmoid().cpu()
            test_metric = multilabel_metrics(test_labels, tip_prob)
            print("**** Tip-Adapter's test mAP: {:.4f}. ****\n".format(test_metric["mAP"]))
            
            return test_metric
        
        # Enable the cached keys to be learnable
        self.adapter = nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).cuda()
        self.adapter.weight = nn.Parameter(self.cache_keys.t())
        # print(self.adapter.weight.dtype)
        
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.lr, eps=1e-4)
        if self.annealling:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs * len(train_loader) / self.accumulate_step, eta_min=1e-5)    
            self.scheduler = ConstantWarmupScheduler(self.optimizer, self.scheduler, 5, 1e-4)
        
        # Test metric before training
        test_metrics = self.get_metrics_finetune("test", best_alpha, best_beta)
        test_map = test_metrics["mAP"]
        print(f"test mAP from best clip adapter model {test_map}")
        wandb.log({"test_map": test_map, "epoch": 0})
        wandb.log({"test_f1": test_metrics["f1"], "epoch": 0})
        wandb.log({"test_precision": test_metrics["precision"], "epoch": 0})
        wandb.log({"test_recall": test_metrics["recall"], "epoch": 0})

        # Training Prodecure
        print("**** Start Training **** \n")
        best_val_map, best_epoch = 0.0, 0
        for epoch in range(self.epochs):
            # Train
            self.adapter.train()
            epoch_loss = 0 
            batch_count = 0
            print('Train Epoch: {:} / {:}'.format(epoch, self.epochs))
            self.optimizer.zero_grad()

            for i, (images, target, _, _) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                # print("Extract features")
                with torch.no_grad():
                    image_features, __ = self.model.extract_feat_img(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                affinity = self.adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values.to(affinity.dtype)
                clip_logits = self.scale_logits * image_features @ self.feats_templates.T
                tip_logits = clip_logits + cache_logits * alpha

                loss = self.criterion(tip_logits, target) / self.accumulate_step
                epoch_loss += loss.item()
                batch_count += 1

                loss.backward()

                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader): # last iteration need to update grad
                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                
            avg_epoch_loss = epoch_loss / batch_count
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            current_lr = self.scheduler.get_last_lr()[0]
            print('Epoch {} LR: {:.6f}, Loss: {:.4f}'.format(epoch+1, current_lr, avg_epoch_loss))
            self.adapter.eval()

            val_metrics = self.get_metrics_finetune("val", best_alpha, best_beta)
            cur_val_map = val_metrics["mAP"]
            wandb.log({"val_map": cur_val_map, "epoch": epoch+1})
            wandb.log({"val_f1": val_metrics["f1"], "epoch": epoch+1})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch+1})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch+1})

            save_model = self.early_stopping(cur_val_map)
            if save_model:
                torch.save(self.state_dict(), self.checkpoint_path)
                # print(f'Validation map decreased ({self.val_map_max:.6f} --> {val_map:.6f}). Saving model...')
                print(f'Validation map increased. Saving model...')

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if cur_val_map > best_val_map:
                best_val_map = cur_val_map
                test_metrics = self.get_metrics_finetune("test", best_alpha, best_beta)
                test_map = test_metrics["mAP"]
                print(f"test mAP from best clip adapter model {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch+1})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})

        wandb.finish()
        return test_metrics