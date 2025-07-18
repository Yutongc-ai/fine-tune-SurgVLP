import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features
from methods.utils import multilabel_metrics
from tqdm import tqdm
import math
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

class ZoomIn(nn.Module):
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
        self.epochs = configs.epochs
        
        self.feature_width = configs.model_config.backbone_img.num_classes
        self.num_classes = configs.dataset_config.num_classes

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
        # unfreeze settings

        self.unfreeze = configs.unfreeze
        self.unfreeze_layer = configs.unfreeze_layer
        if self.unfreeze:
            for param in self.model.backbone_img.global_embedder.parameters():
                param.requires_grad = True
            print("Unfrozen global embedder of vision model")

            if self.unfreeze_layer == 'last':
                for param in self.model.backbone_img.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfrozen layer4 of vision encoder")

    def get_zoomin_patches(self, test_features):
        """
            test_features: (bs, HW, 768)
            templates: (7, 77) 7 classes prompts
        """
        feature_width = int(math.sqrt(test_features.shape[1]))

        similarity_matrix = torch.einsum('bpd,cd->bpc', test_features, self.feats_templates)  # [B, HW, C]
    
        batch_size, num_patches, num_classes = similarity_matrix.shape
        flat_similarities = similarity_matrix.view(batch_size, -1)  # [B, P*C]
        
        top_similarities, flat_indices = torch.topk(flat_similarities, k=feature_width, dim=1)  # [B, H]
        
        top_patch_indices = flat_indices // num_classes  # [B, H]
        top_class_indices = flat_indices % num_classes   # [B, H]

        neg_flat_similarities = -1 * flat_similarities
        
        neg_top_similarities, neg_flat_indices = torch.topk(neg_flat_similarities, k=feature_width, dim=1)  # [B, H]
        
        neg_top_patch_indices = neg_flat_indices // num_classes  # [B, H]
        neg_top_class_indices = neg_flat_indices % num_classes   # [B, H]
        
        return (top_similarities, top_patch_indices, top_class_indices), (neg_top_similarities, neg_top_patch_indices, neg_top_class_indices)

    def get_test_metrics(self):
        # total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_features, batch_labels in self.test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            if self.attn_pooling:
                attn_batch_features = self.model.attention_pooling(batch_features, self.templates)  # (bs, 768)
            else:
                attn_batch_features = self.model.average_pooling(batch_features)  # (bs, 768)
            
            batch_logits = self.temperature * attn_batch_features @ self.feats_templates.T

            assert not torch.isnan(batch_logits).any(), "NaN in logits"
            assert not torch.isinf(batch_logits).any(), "Inf in logits"
            
            batch_prob = batch_logits.sigmoid()
            batch_pred = (batch_prob > self.threshold).int()
            
            # total_loss += nn.BCEWithLogitsLoss()(batch_logits, batch_labels).item() * batch_features.size(0)
            all_preds.append(batch_pred.cpu())
            all_labels.append(batch_labels.cpu())

        # avg_loss = total_loss / len(self.test_loader)
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        metrics = multilabel_metrics(final_labels, final_preds)
        
        return metrics
    
    def get_val_metrics(self, val_loader):
        # total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                
                _, batch_features = self.model.extract_feat_img(val_images)
                batch_features = batch_features / (batch_features.norm(dim = 1, keepdim = True) + 1e-8)

                if self.attn_pooling:
                    attn_batch_features = self.model.attention_pooling(batch_features, self.templates)  # (bs, 768)
                else:
                    attn_batch_features = self.model.average_pooling(batch_features)  # (bs, 768)
                
                batch_logits = self.temperature * attn_batch_features @ self.feats_templates.T

                assert not torch.isnan(batch_logits).any(), "NaN in logits"
                assert not torch.isinf(batch_logits).any(), "Inf in logits"

                batch_prob = batch_logits.sigmoid()
                batch_pred = (batch_prob > self.threshold).int()
                
                # total_loss += nn.BCEWithLogitsLoss()(batch_logits, batch_labels).item() * batch_features.size(0)
                all_preds.append(batch_pred.cpu())
                all_labels.append(val_labels.cpu())

        # avg_loss = total_loss / len(self.test_loader)
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        metrics = multilabel_metrics(final_labels, final_preds)
        
        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):
        wandb.init(
            project="few-shot-surgvlp",
            name=f"zoomin_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
        )
        templates = dataset.templates

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        input_ids = self.templates['input_ids']
        token_type_ids = self.templates['token_type_ids']
        attention_masks = self.templates['attention_mask']

        # (num_classes, dim)
        _, feats_templates, _ = self.model.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
        self.feats_templates = feats_templates / (feats_templates.norm(dim = 1, keepdim = True) + 1e-8)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.configs.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        test_features, test_labels = preload_local_features(self.configs, "test", self.model, test_loader)

        self.test_dataset = TensorDataset(test_features, test_labels)
        batch_size = self.configs.batch_size
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        del test_features, test_labels

        test_metrics = self.get_test_metrics()
        test_f1 = test_metrics["f1"]
        print(f"Epoch {0} test f1 {test_f1}")
        wandb.log({"test_f1": test_f1, "epoch": 0})
        wandb.log({"test_precision": test_metrics["precision"], "epoch": 0})
        wandb.log({"test_recall": test_metrics["recall"], "epoch": 0})

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
        val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)
        val_loader = build_data_loader(data_source=val_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                       num_classes = dataset.num_classes)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        optim_params = []

        if self.unfreeze:
            vision_params = [{'params': self.model.backbone_img.global_embedder.parameters(), 'lr': self.lr * 0.1}]

            if self.unfreeze_layer == 'last':
                vision_params.append({'params': self.model.backbone_img.model.layer4.parameters(), 'lr': self.lr * 0.1})

        optim_params.extend(vision_params)
        self.optimizer = torch.optim.Adam(optim_params, weight_decay=1e-5)

        train_loss = []
        best_val_f1 = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_sim = 0.0
            batch_count = 0
            
            if self.unfreeze:
                self.model.backbone_img.model.train()
            
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                _, image_features = self.model.extract_feat_img(images)
                
                image_features = image_features / (image_features.norm(dim = 1, keepdim = True) + 1e-8)

                batch_size = image_features.shape[0]
                local_feature_dim = image_features.shape[1]
                image_features = image_features.view(batch_size, local_feature_dim, -1).transpose(1, 2) # result into (batch_size. H*W, dim)
                image_features = self.model.backbone_img.global_embedder(image_features) # result into (bs, HW, 768)

                image_features = image_features / (image_features.norm(dim = -1, keepdim = True) + 1e-8)
                
                (sim_scores, patch_indexes, class_indexes), (neg_sim_scores, neg_patch_indexes, neg_class_indexes) = self.get_zoomin_patches(image_features)

                # print("sim score")
                # print(sim_scores.shape)
                # print("neg sim score")
                # print(neg_sim_scores.shape)
                # print(sim_scores)
                # print(neg_sim_scores)
                loss_sim = (sim_scores.mean(dim=1) + neg_sim_scores.mean(dim=1)) / 2
                # print("first mean", loss_sim)
                loss_sim = loss_sim.mean()
                # print("final sim", loss_sim)

                global_image_features = torch.mean(image_features, dim=1).squeeze(1) # (bs, 768)
                logits = self.temperature * global_image_features @ self.feats_templates.T

                self.optimizer.zero_grad()

                loss_ce = self.criterion(logits, target)

                loss = -loss_sim + loss_ce
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_sim += loss_sim.item()
                batch_count += 1
                
            
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_loss_ce = epoch_loss_ce / batch_count
            avg_epoch_loss_sim = epoch_loss_sim / batch_count

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            wandb.log({"epoch_loss_ce": avg_epoch_loss_ce, "epoch": epoch+1})
            wandb.log({"epoch_loss_sim": avg_epoch_loss_sim, "epoch": epoch+1})
            
            train_loss.append(avg_epoch_loss)
            
            if self.unfreeze:
                self.model.backbone_img.model.train()

            val_metrics = self.get_val_metrics(val_loader)
            cur_val_f1 = val_metrics["f1"]
            wandb.log({"val_f1": cur_val_f1, "epoch": epoch+1})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch+1})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch+1})

            if cur_val_f1 > best_val_f1:

                best_val_f1 = cur_val_f1
                test_metrics = self.get_test_metrics()
                test_f1 = test_metrics["f1"]
                print(f"Epoch {epoch + 1} best val f1 {best_val_f1:.4f} test f1 {test_f1}")
                wandb.log({"test_f1": test_f1, "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})

        wandb.finish()
        return test_metrics