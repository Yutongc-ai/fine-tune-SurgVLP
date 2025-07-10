import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics
from tqdm import tqdm
import copy
import wandb
from typing import Callable, Optional
from collections import OrderedDict
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

class ResidualCrossAttn(nn.Module):
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs
        self.attn_pooling = configs.attention_pooling

        self.temperature = 100
        self.threshold = 0.5
        self.lr = 0.001
        self.lr_alpha = 0.01
        self.epochs = configs.epochs

        self.layer4_width = 2048
        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes

        # self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.image_query_attn = CrossAttentionBlock(self.feature_width).to(device)
        self.text_query_attn = CrossAttentionBlock(self.feature_width).to(device)
        # self.text_mlp = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.image_mlp = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.test_norm1 = nn.LayerNorm(self.layer4_width).to(device)
        self.test_norm2 = nn.LayerNorm(self.feature_width).to(device)

        self.norm1 = nn.LayerNorm(self.layer4_width).to(device)
        self.norm2 = nn.LayerNorm(self.feature_width).to(device)

        # unfreeze part of image encoder
        for name, param in self.model.named_parameters():
            # print(param.requires_grad)
            # print(name)
            param.requires_grad = False
        
        self.unfreeze = configs["unfreeze"]
        self.unfreeze_layer = configs.unfreeze_layer

        if self.unfreeze:
            print("Unfreeze text encoder")
            for name, param in self.model.backbone_text.named_parameters():
                # print(name, param.requires_grad)
                param.requires_grad = True
            
            for param in self.model.backbone_img.global_embedder.parameters():
                param.requires_grad = True

            if self.unfreeze_layer == 'last':
                for param in self.model.backbone_img.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfrozen layer4 of vision encoder")
        else:
            print("Keep vision encoder frozen")

    def get_metrics(self, split):
        # total_loss = 0.0
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

                img_logits = self.bidirect_cross_attn(local_image_features, global_image_features)

                img_prob = img_logits.sigmoid()

                all_img_probs.append(img_prob)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_img_probs = torch.cat(all_img_probs, dim=0).to('cpu')
        img_metrics = multilabel_metrics(final_labels, final_img_probs)

        return img_metrics

    def bidirect_cross_attn(self, local_image_features, global_image_features):
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

        global_image_features = global_image_features.unsqueeze(1)
        detached_global_image_features = global_image_features.detach()
        
        # 2. get template text features
        _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)

        batched_template_features = feats_templates.repeat(bs, 1, 1)
        
        # 3. global image feature query text features
        attned_text_f = self.image_query_attn(detached_global_image_features, batched_template_features, batched_template_features).squeeze(1)
        # text_logits = self.text_mlp(attned_text_f)

        # 4. global text features query image features
        detached_text_f = attned_text_f.detach().unsqueeze(1)
        attned_img_f = self.text_query_attn(detached_text_f, local_image_features, local_image_features).squeeze(1)
        global_image_features = global_image_features.squeeze(1)
        attned_img_f = (attned_img_f + global_image_features) / 2

        img_logits = self.image_mlp(attned_img_f)

        return img_logits

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates

        wandb.init(
            project="few-shot-surgvlp-cross-attn-residual",
            name=f"multihead_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
        )

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']

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

        # for i, (images, target) in enumerate(tqdm(train_loader)):
        #     print(target)

        # assert(0)


        optim_params = [
            {'params': self.image_mlp.parameters(), 'lr': self.lr},
            {'params': self.image_query_attn.parameters(), 'lr': self.lr},
            {'params': self.text_query_attn.parameters(), 'lr': self.lr},
        ]

        if self.unfreeze:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr * 0.1})
            optim_params.append({'params': self.model.backbone_img.global_embedder.parameters(), 'lr': self.lr * 0.1})
            
            if self.unfreeze_layer == 'last':
                optim_params.append({'params': self.model.backbone_img.model.layer4.parameters(), 'lr': self.lr * 0.1})
        
        self.optimizer = torch.optim.Adam(optim_params, weight_decay=1e-5)

        train_loss = []
        best_val_img_map = 0
        update_test_img_metric = False

        res_img_metrics = None
        
        # initial_params = copy.deepcopy({name: param.data.clone() for name, param in self.model.backbone_img.named_parameters()})

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # train mode
            # self.text_mlp.train()
            self.image_mlp.train()
            self.text_query_attn.train()
            self.image_query_attn.train()
            if self.unfreeze:
                self.model.train()
            
            for i, (images, target, _) in enumerate(tqdm(train_loader)):
                self.optimizer.zero_grad()
                images, target = images.to(device), target.to(device)
                
                if self.unfreeze:
                    # 1.get global and local image features
                    global_image_features, local_image_features = self.model.extract_feat_img(images)
                    local_image_features = local_image_features.permute(0, 2, 3, 1)
                    img_logits = self.bidirect_cross_attn(local_image_features, global_image_features)

                else:
                    assert(0, "Can only unfreeze encoder when doing cross attention")
                
                img_loss = self.criterion(img_logits, target)
                
                loss = img_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                

            avg_epoch_loss = epoch_loss / batch_count
            # wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            wandb.log({"loss": avg_epoch_loss, "epoch": epoch})
            
            # after_epoch1_params = {name: param.data.clone() for name, param in self.model.backbone_img.named_parameters()}
            # updated_in_epoch1 = False
            # for name in initial_params:
            #     if not torch.equal(initial_params[name], after_epoch1_params[name]):
            #         diff = torch.sum(torch.abs(initial_params[name] - after_epoch1_params[name]))
            #         print(f"✅ {name} 在 epoch1 更新 | 变化量: {diff:.6f}")
            #         updated_in_epoch1 = True
            #     else:
            #         print(f"❌ {name} 在 epoch1 未更新！请检查优化器/梯度")

            # if not updated_in_epoch1:
            #     raise RuntimeError("所有参数在 epoch1 均未更新！")
            
            # assert(0)

            self.image_mlp.eval()
            self.text_query_attn.eval()
            self.image_query_attn.eval()
            if self.unfreeze:
                self.model.eval()

            train_loss.append(avg_epoch_loss)

            val_img_metrics = self.get_metrics("val")

            cur_val_img_map = val_img_metrics["mAP"]
            wandb.log({"img_val_map": cur_val_img_map, "epoch": epoch})
            wandb.log({"img_val_f1": val_img_metrics["f1"], "epoch": epoch})
            wandb.log({"img_val_precision": val_img_metrics["precision"], "epoch": epoch})
            wandb.log({"img_val_recall": val_img_metrics["recall"], "epoch": epoch})

            if cur_val_img_map > best_val_img_map:
                best_val_img_map = cur_val_img_map
                update_test_img_metric = True

            if update_test_img_metric:

                test_img_metrics = self.get_metrics("test")
                res_img_metrics = test_img_metrics
                map = res_img_metrics["mAP"]
                wandb.log({"img_test_map": map, "epoch": epoch})
                wandb.log({"img_test_f1": res_img_metrics["f1"], "epoch": epoch})
                wandb.log({"img_test_precision": res_img_metrics["precision"], "epoch": epoch})
                wandb.log({"img_test_recall": res_img_metrics["recall"], "epoch": epoch})

            update_test_img_metric = False

        wandb.finish()
        return res_img_metrics