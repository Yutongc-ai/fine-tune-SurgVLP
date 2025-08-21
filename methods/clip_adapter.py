import copy

from .utils import *

from torch.optim.lr_scheduler import _LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipAdapter(nn.Module):
    '''
    CLIP Adapter method
        @article{gao2021clip,
            title={CLIP-Adapter: Better Vision-Language Models with Feature Adapters},
            author={Gao, Peng and Geng, Shijie and Zhang, Renrui and Ma, Teli and Fang, Rongyao and Zhang, Yongfeng and Li, Hongsheng and Qiao, Yu},
            journal={arXiv preprint arXiv:2110.04544},
            year={2021}
        }
    
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
        self.checkpoint_path =  "checkpoints/best_clipAdapter_" + str(self.configs.num_shots) + "shots.pt"
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        self.annealling = configs.annealling

        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes
        self.alpha = configs['alpha_ca']
        self.search_alpha_ca = configs['search_alpha_ca']

        # ======================================Training Parameters=====================================
        # metrcis
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # ======================================Trained Adapter=========================================
        print('Building custom CLIP')
        self.clip_ad_model = CustomCLIP(self.model)
        self.clip_ad_model_val = copy.deepcopy(self.clip_ad_model)

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
    
    def get_metrics(self, split, clip_ad_model_val, alpha):
        # total_loss = 0.0
        all_probs = []
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

                logits = clip_ad_model_val(global_image_features, self.feats_templates, alpha)

                prob = logits.sigmoid()

                all_probs.append(prob)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):
        wandb.init(
            project=f"clip-adapter-{self.configs.model_config.type}",
            name=f"batchsize{self.batch_size * self.accumulate_step}_lr{self.lr}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
        )

        self.templates = self.tokenizer(dataset.templates, device = device)
        
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']

        with torch.no_grad():
            _, self.feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
            self.feats_templates = self.feats_templates.to(device)
        
        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.clip_ad_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                
        for name, param in self.clip_ad_model_val.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        
        self.clip_ad_model.to(device)
        self.clip_ad_model_val.to(device)

        # Creating dataset loader
        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)
        
        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80Features(self.configs, "val")

        print(len(self.test_feature))
        print(len(self.val_feature))

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")

        train_loader = build_data_loader(data_source=train_data, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)

        # alpha initialization
        self.clip_ad_model.eval()
        if self.search_alpha_ca:
            best_mAP = 0.0
            print("**** Searching for best initialization of alpha **** \n")
            alpha_list = [0.5] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for init_alpha in alpha_list:
                self.clip_ad_model_val.adapter = self.search_init_hp(init_alpha, train_loader, self.clip_ad_model, self.clip_ad_model_val, self.feats_templates)
                search_map = self.get_metrics("val", self.clip_ad_model_val, init_alpha)["mAP"]                

                print(f"Search init alpha {init_alpha}, get validation map {search_map}")
                if search_map > best_mAP:
                    best_mAP = search_map
                    self.alpha = init_alpha

        print(f"Use alpha: {self.alpha}")

        print("--- ID Check Before Optimizer ---")
        adapter_params_before = list(self.clip_ad_model.adapter.parameters())
        print(f"Adapter's first param ID: {id(adapter_params_before[0])}")


        self.optimizer = torch.optim.SGD(self.clip_ad_model.adapter.parameters(), self.lr)

        if self.annealling:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs * len(train_loader) / self.accumulate_step, eta_min=1e-5)    
            # warm start increase to 3?
            self.scheduler = ConstantWarmupScheduler(self.optimizer, self.scheduler, 5, 1e-4)

        # Precompute metrics before training
        test_metrics = self.get_metrics("test", self.clip_ad_model, self.alpha)
        test_map = test_metrics["mAP"]
        print(f"Initial test mAP {test_map}")
        wandb.log({"test_map": test_map, "epoch": 0})
        wandb.log({"test_f1": test_metrics["f1"], "epoch": 0})
        wandb.log({"test_precision": test_metrics["precision"], "epoch": 0})
        wandb.log({"test_recall": test_metrics["recall"], "epoch": 0})

        val_metrics = self.get_metrics("val", self.clip_ad_model, self.alpha)
        val_map = val_metrics["mAP"]
        print(f"Initial val mAP {val_map}")
        wandb.log({"val_map": val_map, "epoch": 0})
        wandb.log({"val_f1": val_metrics["f1"], "epoch": 0})
        wandb.log({"val_precision": val_metrics["precision"], "epoch": 0})
        wandb.log({"val_recall": val_metrics["recall"], "epoch": 0})

        print(f"Initial val mAP {val_map} test mAP {test_map}")
        # Train
        print('\nStart Training procedure')

        best_val_map = 0.0

        for epoch in range(self.epochs):
            # Train
            self.clip_ad_model.adapter.train()
            epoch_loss = 0 
            batch_count = 0
            print('Train Epoch: {:} / {:}'.format(epoch, self.epochs))

            for i, (images, target, _) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features, _ = self.model.extract_feat_img(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = self.clip_ad_model(image_features, self.feats_templates, self.alpha)

                loss = self.criterion(logits, target) / self.accumulate_step
                epoch_loss += loss.item()
                batch_count += 1

                loss.backward()
                
                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader): # last iteration need to update grad
                    param_to_check = next(self.clip_ad_model.adapter.parameters())
                
                    # 使用 float64 来保存，避免精度问题
                    value_before_step = param_to_check.data.clone().detach().to(torch.float64)

                    current_lr = self.optimizer.param_groups[0]['lr']
                    # print(f"\n[Epoch {epoch}, Step {i}] Current LR: {current_lr:.10f}")

                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    value_after_step = param_to_check.data.to(torch.float64)
                    change = (value_after_step - value_before_step).abs().sum()

                    # print("\n--- Update Check ---")
                    # print(f"Parameter change (sum of absolute differences): {change.item():.15f}")
                    # if change.item() > 1e-12: # 使用一个非常小的阈值
                    #     print("✅ param changes!")
                    # else:
                    #     print("❌ no change param!")
                    # print("--------------------------------\n")
        
            avg_epoch_loss = epoch_loss / batch_count
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            current_lr = self.scheduler.get_last_lr()[0]
            print('Epoch {} LR: {:.6f}, Loss: {:.4f}'.format(epoch+1, current_lr, avg_epoch_loss))
            self.clip_ad_model.adapter.eval()

            val_metrics = self.get_metrics("val", self.clip_ad_model, self.alpha)
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
                test_metrics = self.get_metrics("test", self.clip_ad_model, self.alpha)
                test_map = test_metrics["mAP"]
                print(f"test mAP from best clip adapter model {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch+1})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})

        test_metrics["clip_ad_alpha"] = self.alpha
        wandb.finish()
        return test_metrics

    def search_init_hp(self, alpha, val_loader, tool_model, clip_ad_model, text_weights):
        optimizer = torch.optim.SGD(clip_ad_model.adapter.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs / 20)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, 1, 0.00001)
        # Train
        print('\nStart Training procedure')

        clip_ad_model.adapter.train()
        for train_idx in range(int(self.epochs / 20)):
            # Train
            for i, (images, target, _) in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    images = images.cuda()
                    target = target.cuda()
                    image_features, __ = tool_model.model.extract_feat_img(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = clip_ad_model(image_features, text_weights, alpha)

                loss = self.criterion(logits, target) / self.accumulate_step
                loss.backward()

                if i % self.accumulate_step == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
        # Eval
        clip_ad_model.adapter.eval()

        return clip_ad_model.adapter

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class CustomCLIP(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.logit_scale = 10
        self.adapter = Adapter(768, 2)
        self.model = model
            
    def forward(self, image_features, text_features, alpha):
        x = self.adapter(image_features)

        # alpha = 0.2
        image_features = alpha * x + (1 - alpha) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale * image_features @ text_features.T

        return logits

class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)
                

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

