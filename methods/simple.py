import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.loss import InfoNCE
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics
from tqdm import tqdm
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

class Simple(nn.Module):
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs
        self.attn_pooling = configs.attention_pooling

        if self.attn_pooling:
            assert(0, "attn pooling version of negation method is not realized yet")

        self.temperature = 0.01
        self.threshold = 0.5
        self.lr = 0.001
        self.epochs = configs.epochs
        
        self.feature_width = configs.model_config.backbone_img.num_classes
        self.num_classes = configs.dataset_config.num_classes

        self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_func = InfoNCE(temperature= 0.01)
        
        self.norm = nn.LayerNorm(self.feature_width).to(device)

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
        
        # unfreeze settings

        # vision encoder settings
        self.unfreeze_vision = configs["unfreeze_vision"]
        self.unfreeze_vision_layer = configs.unfreeze_vision_layer
        if self.unfreeze_vision:
            for param in self.model.backbone_img.global_embedder.parameters():
                param.requires_grad = True
            
            if self.unfreeze_vision_layer == 'last':
                for param in self.model.backbone_img.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfrozen layer4 of vision encoder")
        else:
            print("Keep vision encoder frozen")

        # text encoder settings
        self.unfreeze_text = configs["unfreeze_text"]
        if self.unfreeze_text:
            for name, param in self.model.backbone_text.named_parameters():
                param.requires_grad = True
            print("Unfreeze whole text encoder")
        else:
            print("Keep text encoder frozen")

    def get_metrics(self, split):
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
                global_image_features, local_image_features, label = feature_loader[_]
                local_image_features = local_image_features.to(device)
                global_image_features = global_image_features.to(device)
                local_image_features = local_image_features.permute(0, 3, 1, 2)

                if self.attn_pooling:
                    image_features = self.model.attention_pooling(local_image_features, self.templates)
                else:
                    image_features = global_image_features

                _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)

                image_features = image_features / image_features.norm(dim = -1, keepdim = True)

                feats_templates = feats_templates / feats_templates.norm(dim = -1, keepdim = True)
                feats_templates = feats_templates.cuda()

                logits = self.temperature * image_features @ feats_templates.T
                probs = logits.sigmoid()

                all_probs.append(probs)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def build_labeled_embeddings(self, text_embedding, label):
        """
        Args:
            text_embedding: [7, 768], 7个类别的embedding
            label: [bs, 7], 每个样本的7个二分类标签（0或1）
        Returns:
            output: [bs, 7, 768], 正样本位置填对应embedding，负样本位置为0
        """
        # 将label扩展为[bs, 7, 1]，用于mask
        mask = label.unsqueeze(-1).float()  # [bs, 7, 1]
        # 将text_embedding扩展为[bs, 7, 768]
        expanded_emb = text_embedding.unsqueeze(0).repeat(label.size(0), 1, 1)  # [bs, 7, 768]
        # 按mask筛选，正样本保留embedding，负样本置0
        output = expanded_emb * mask  # [bs, 7, 768]

        return output

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates

        wandb.init(
            project="negation-few-shot-surgvlp",
            name=f"simple_{'attn' if self.attn_pooling else 'avg'}_{'unfreeze' if self.unfreeze_vision or self.unfreeze_text else 'freeze'}_shot{self.configs.num_shots}_epoch{self.epochs}",
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

        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                       num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80Features(self.configs, "val")

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
        # val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)


        optim_params = [
            {'params': self.classifier.parameters(), 'lr': self.lr},
        ]

        if self.unfreeze_vision:
            optim_params.append({'params': self.model.backbone_img.global_embedder.parameters(), 'lr': self.lr * 0.1})
            if self.unfreeze_vision_layer == 'last':
                optim_params.append({'params': self.model.backbone_img.model.layer4.parameters(), 'lr': self.lr * 0.1})
        
        if self.unfreeze_text:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr * 0.1})

        self.optimizer = torch.optim.Adam(optim_params, weight_decay=1e-5)

        train_loss = []
        best_val_map = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            self.classifier.train()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.train()
            
            for i, (images, target, _) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                
                if self.unfreeze_vision or self.unfreeze_text:
                    # get image embedding
                    # global_features: [bs, 768] local_features: [bs, 2048, 7, 7]
                    global_features, image_features = self.model.extract_feat_img(images)
                    if self.attn_pooling:
                        image_features = self.model.attention_pooling(image_features, self.templates)
                    else:
                        image_features = global_features
                    
                    # get text embedding
                    # [7, 768]
                    _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)

                else:
                    assert(0, "Negation method need to unfreeze encoder")
                
                image_features = image_features / image_features.norm(dim = -1, keepdim = True)

                feats_templates = feats_templates / feats_templates.norm(dim = -1, keepdim = True)
                feats_templates = feats_templates.cuda()

                pos_embedding = self.build_labeled_embeddings(feats_templates, target)

                self.optimizer.zero_grad()
                
                loss = self.loss_func(image_features, pos_embedding)    
                
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            
            self.classifier.eval()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.eval()

            train_loss.append(avg_epoch_loss)

            val_metrics = self.get_metrics("val")
            cur_val_map = val_metrics["mAP"]
            wandb.log({"val_map": cur_val_map, "epoch": epoch})
            wandb.log({"val_f1": val_metrics["f1"], "epoch": epoch})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch})

            if cur_val_map > best_val_map:
                best_val_map = cur_val_map
                test_metrics = self.get_metrics("test")
                test_map = test_metrics["mAP"]
                print(f"Epoch {epoch + 1} best val mAP {best_val_map:.4f} test mAP {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch})
        
        wandb.finish()
        return test_metrics