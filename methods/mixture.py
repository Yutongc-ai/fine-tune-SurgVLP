import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
from typing import Callable, Optional
from methods.loss import InfoNCE
device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int = 4,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1_kv = norm_layer(d_model)

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0] # (attned features, attned weights) return attentioned features

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        return x

class Mixture(nn.Module): # Include cross attention and negation text
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
        self.checkpoint_path =  configs.checkpoint_path
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        self.annealling = configs.annealling

        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes

        # ======================================Training Parameters=====================================
        # self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.image_query_attn = CrossAttentionBlock(self.feature_width).to(device)
        self.text_query_attn = CrossAttentionBlock(self.feature_width).to(device)
        self.text_mlp = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.image_mlp = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        
        # normalization layer
        self.test_norm2 = nn.LayerNorm(self.feature_width).to(device)

        # balanced parameters
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.lr_alpha = 0.01

        # metrcis
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # loss function
        self.loss_func = InfoNCE(negative_mode='paired')

        # self.norm1 = nn.LayerNorm(self.layer4_width).to(device)
        # self.norm2 = nn.LayerNorm(self.feature_width).to(device)

        # ====================================Unfreeze Settings=========================================
        self.unfreeze_vision = configs["unfreeze_vision"]
        self.unfreeze_text = configs["unfreeze_text"]

        # unfreeze part of image encoder
        for param in self.model.parameters():
            # print(param.requires_grad)
            # print(name)
            param.requires_grad = False

        if self.unfreeze_vision:
            print("Unfreeze vision encoder")
            for param in self.model.backbone_img.parameters():
                param.requires_grad = True
        else:
            print("Keep vision encoder frozen")

        if self.unfreeze_text:
            print("Unfreeze text encoder")
            for param in self.model.backbone_text.parameters():
                param.requires_grad = True
        else:
            print("Keep text encoder frozen")

    def get_metrics(self, split):
        # total_loss = 0.0
        all_text_probs = []
        all_img_probs = []
        all_labels = []

        if split == "val":
            feature_loader = self.val_feature
        elif split == "test":
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")

        with torch.no_grad():

            for _ in range(len(feature_loader)):
                global_image_features, local_image_features, label = feature_loader[_]
                global_image_features = global_image_features.to(device)
                local_image_features = local_image_features.to(device)

                _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                feats_templates = feats_templates.to(device)

                img_logits, text_logits, attned_img, attn_text = self.bidirect_cross_attn(local_image_features, global_image_features, feats_templates[:7])

                text_prob = text_logits.sigmoid()
                img_prob = img_logits.sigmoid()

                all_text_probs.append(text_prob)
                all_img_probs.append(img_prob)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_img_probs = torch.cat(all_img_probs, dim=0).to('cpu')
        img_metrics = multilabel_metrics(final_labels, final_img_probs)

        final_text_probs = torch.cat(all_text_probs, dim=0).to('cpu')
        text_metrics = multilabel_metrics(final_labels, final_text_probs)

        return [img_metrics, text_metrics]
        # return img_metrics

    def bidirect_cross_attn(self, local_image_features, global_image_features, feats_templates):
        """
            local_image_features: [bs, h, w, layer4_width(2048)]
            global_image_features: [bs, feature_width(768)]
        """
        # 1.get global and local image features
        bs, h, w = local_image_features.shape[0], local_image_features.shape[1], local_image_features.shape[2]
        local_image_features = local_image_features.view(bs, h*w, -1)
        # project to 768 feature space
        local_image_features = self.model.backbone_img.global_embedder(local_image_features)

        # norm
        local_image_features = self.test_norm2(local_image_features)
        global_image_features = self.test_norm2(global_image_features)

        detached_global_image_features = global_image_features.unsqueeze(1).detach()
        
        # 2. get template text features
        # _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
        batched_template_features = feats_templates.repeat(bs, 1, 1)
        
        # 3. global image feature query text features
        attned_text_f  = self.image_query_attn(detached_global_image_features, batched_template_features, batched_template_features).squeeze(1)
        text_logits = self.text_mlp(attned_text_f)

        # 4. global text features query image features
        detached_text_f = attned_text_f.detach().unsqueeze(1)
        attned_img_f = self.text_query_attn(detached_text_f, local_image_features, local_image_features).squeeze(1)
        img_logits = self.image_mlp(attned_img_f)

        return img_logits, text_logits, attned_img_f, attned_text_f

    def get_nce_labels(self, target, negated_target, feats_template):
        batch_size = target.shape[0]
        _, emb_size = feats_template.shape
        positive_keys = torch.zeros((batch_size, self.num_classes, emb_size), device=device)
        negative_keys = torch.zeros((batch_size, self.num_classes, emb_size), device=device)

        for bs in range(batch_size):
            for index, mask in enumerate(target[bs]):
                if mask:
                    positive_keys[bs][index] = feats_template[index]

        for bs in range(batch_size):
            for index, mask in enumerate(negated_target[bs]):
                if mask:
                    negative_keys[bs][index] = feats_template[index + 7]
        
        return positive_keys, negative_keys

    def forward(self,
                dataset: MultiLabelDatasetBase):
        wandb.init(
            project="few-shot-surgvlp-mixture",
            name=f"shots{self.configs.num_shots}_epoch{self.epochs}_lr{self.lr}_bs{str(self.batch_size * self.accumulate_step)}_{'annealling' if self.annealling else ''}",
            config=self.configs,
        )

        # test data preparations
        templates = self.tokenizer(dataset.templates + dataset.negated_templates, device = device)
        
        self.input_ids = templates['input_ids']
        self.token_type_ids = templates['token_type_ids']
        self.attention_masks = templates['attention_mask']

        # (num_classes, dim)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.configs.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)
        
        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.configs.batch_size, is_train = False, tfm = self.preprocess,
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

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)


        optim_params = [
            {'params': self.text_mlp.parameters(), 'lr': self.lr * 10},
            {'params': self.image_mlp.parameters(), 'lr': self.lr * 10},
            {'params': self.image_query_attn.parameters(), 'lr': self.lr * 10},
            {'params': self.text_query_attn.parameters(), 'lr': self.lr * 10},
            {'params': [self.log_alpha], 'lr': self.lr_alpha},
        ]

        if self.unfreeze_vision:
            optim_params.append({'params': self.model.backbone_img.parameters(), 'lr': self.lr})
            
        if self.unfreeze_text:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr})
        
        self.optimizer = torch.optim.AdamW(optim_params)
        if self.annealling:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs * len(train_loader))

        train_loss = []
        best_val_img_map = 0
        best_val_text_map = 0
        update_test_img_metric = False
        update_test_text_metric = False

        best_model_weight = None

        res_img_metrics, res_text_metrics = None, None
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # train mode
            # self.text_mlp.train()
            self.image_mlp.train()
            self.text_query_attn.train()
            self.image_query_attn.train()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.train()
            
            for i, (images, target, negated_target) in enumerate(tqdm(train_loader)):
                # self.optimizer.zero_grad()
                images, target, negated_target = images.to(device), target.to(device), negated_target.to(device)
                
                # 1.get global and local image features
                global_image_features, local_image_features = self.model.extract_feat_img(images)
                local_image_features = local_image_features.permute(0, 2, 3, 1)
                
                # 2.get original and negation text templates embedding
                _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                feats_templates = feats_templates.to(device)

                # 3. cross attention part loss
                img_logits, text_logits, attned_img, attned_text = self.bidirect_cross_attn(local_image_features, global_image_features, feats_templates[: 7])
                
                text_loss = self.criterion(text_logits, target)
                img_loss = self.criterion(img_logits, target)
                
                alpha = torch.exp(self.log_alpha)
                loss1 = alpha * text_loss + (1 - alpha) * img_loss
                
                # 4. negation part loss
                positive_keys, negative_keys = self.get_nce_labels(target, negated_target, feats_templates)
                
                loss2 = self.loss_func(attned_img, positive_keys, negative_keys) / self.accumulate_step
                
                # combine loss1 and loss2
                loss = (loss1 + loss2) / 2
                loss = loss / self.accumulate_step

                loss.backward()

                if i % self.accumulate_step == 0:
                    self.optimizer.step()
                    
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                batch_count += 1
                
            wandb.log({"alpha": torch.exp(self.log_alpha), "epoch": epoch})

            avg_epoch_loss = epoch_loss / batch_count
            # wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            wandb.log({"loss": avg_epoch_loss, "epoch": epoch})

            self.text_mlp.eval()
            self.image_mlp.eval()
            self.text_query_attn.eval()
            self.image_query_attn.eval()
            if self.unfreeze_text or self.unfreeze_vision:
                self.model.eval()

            train_loss.append(avg_epoch_loss)

            val_img_metrics, val_text_metrics = self.get_metrics("val")

            cur_val_img_map = val_img_metrics["mAP"]
            wandb.log({"img_val_map": cur_val_img_map, "epoch": epoch})
            wandb.log({"img_val_f1": val_img_metrics["f1"], "epoch": epoch})
            wandb.log({"img_val_precision": val_img_metrics["precision"], "epoch": epoch})
            wandb.log({"img_val_recall": val_img_metrics["recall"], "epoch": epoch})

            cur_val_text_map = val_text_metrics["mAP"]
            wandb.log({"text_val_map": cur_val_text_map, "epoch": epoch})
            wandb.log({"text_val_f1": val_text_metrics["f1"], "epoch": epoch})
            wandb.log({"text_val_precision": val_text_metrics["precision"], "epoch": epoch})
            wandb.log({"text_val_recall": val_text_metrics["recall"], "epoch": epoch})


            save_model = self.early_stopping(cur_val_img_map)
            if save_model:
                torch.save(self.state_dict(), self.checkpoint_path)
                print(f'Validation map decreased. Saving model...')

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if cur_val_img_map > best_val_img_map:
                best_val_img_map = cur_val_img_map
                update_test_img_metric = True

            if cur_val_text_map > best_val_text_map:
                best_val_text_map = cur_val_text_map
                update_test_text_metric = True

            if update_test_text_metric or update_test_img_metric:

                test_img_metrics, test_text_metrics = self.get_metrics("test")
                if update_test_img_metric:
                    res_img_metrics = test_img_metrics
                    map = res_img_metrics["mAP"]
                    wandb.log({"img_test_map": map, "epoch": epoch})
                    wandb.log({"img_test_f1": res_img_metrics["f1"], "epoch": epoch})
                    wandb.log({"img_test_precision": res_img_metrics["precision"], "epoch": epoch})
                    wandb.log({"img_test_recall": res_img_metrics["recall"], "epoch": epoch})

                    # # update best model weight to save
                    # self.early_stopping.save_checkpoint(self.state_dict(), self.checkpoint_path)

                if update_test_text_metric:
                    res_text_metrics = test_text_metrics
                    map = res_text_metrics["mAP"]
                    wandb.log({"text_test_map": map, "epoch": epoch})
                    wandb.log({"text_test_f1": res_text_metrics["f1"], "epoch": epoch})
                    wandb.log({"text_test_precision": res_text_metrics["precision"], "epoch": epoch})
                    wandb.log({"text_test_recall": res_text_metrics["recall"], "epoch": epoch})

            update_test_img_metric = False
            update_test_text_metric = False

        wandb.finish()
        return res_img_metrics, res_text_metrics