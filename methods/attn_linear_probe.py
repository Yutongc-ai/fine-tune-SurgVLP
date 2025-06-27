import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features
from methods.utils import multilabel_metrics
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class LP(nn.Module):
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
        # 七个二分类？
        self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
        # unfreeze settings

        self.unfreeze = configs.unfreeze
        self.unfreeze_layer = configs.unfreeze_layer
        if self.unfreeze:
            if self.unfreeze_layer == 'last':
                for param in self.model.backbone_img.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfrozen layer4 of vision encoder")

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
            
            batch_logits = self.classifier(attn_batch_features)
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
                batch_features = batch_features / (batch_features.norm(dim = -1, keepdim = True) + 1e-8)

                if self.attn_pooling:
                    attn_batch_features = self.model.attention_pooling(batch_features, self.templates)  # (bs, 768)
                else:
                    attn_batch_features = self.model.average_pooling(batch_features)  # (bs, 768)
                
                batch_logits = self.classifier(attn_batch_features)
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
        templates = dataset.templates

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        input_ids = self.templates['input_ids']
        token_type_ids = self.templates['token_type_ids']
        attention_masks = self.templates['attention_mask']
        
        # (num_classes, dim)
        _, self.text_embeddings, _ = self.model.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.configs.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        test_features, test_labels = preload_local_features(self.configs, "test", self.model, test_loader)

        self.test_dataset = TensorDataset(test_features, test_labels)
        batch_size = self.configs.batch_size
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        del test_features, test_labels

        test_metrics = self.get_test_metrics()

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
        val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)
        val_loader = build_data_loader(data_source=val_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                       num_classes = dataset.num_classes)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        optim_params = [
            {'params': self.classifier.parameters(), 'lr': self.lr},
        ]

        if self.unfreeze:
            vision_params = []
            
            if self.unfreeze_layer == 'last':
                vision_params.append({'params': self.model.backbone_img.model.layer4.parameters(), 'lr': self.lr * 0.1})
        optim_params.extend(vision_params)
        self.optimizer = torch.optim.Adam(optim_params, weight_decay=1e-5)

        train_loss = []
        best_val_f1 = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            self.classifier.train()
            if self.unfreeze:
                self.model.backbone_img.model.train()
            
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                
                if self.unfreeze:
                    _, image_features = self.model.extract_feat_img(images)
                    image_features = image_features / (image_features.norm(dim = -1, keepdim = True) + 1e-8)
                    if self.attn_pooling:
                        image_features = self.model.attention_pooling(image_features, self.templates)
                    else:
                        image_features = self.model.average_pooling(image_features)
                else:
                    with torch.no_grad():
                        _, image_features = self.model.extract_feat_img(images)
                        image_features = image_features / (image_features.norm(dim = -1, keepdim = True) + 1e-8)
                        if self.attn_pooling:
                            image_features = self.model.attention_pooling(image_features, self.templates)
                        else:
                            image_features = self.model.average_pooling(image_features)
                
                self.optimizer.zero_grad()
                logits = self.classifier(image_features)
                
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                
            
            avg_epoch_loss = epoch_loss / batch_count

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            
            self.classifier.eval()
            train_loss.append(avg_epoch_loss)
            
            cur_val_f1 = self.get_val_metrics(val_loader)["f1"]
            if cur_val_f1 > best_val_f1:
                best_val_f1 = cur_val_f1
                test_metrics = self.get_test_metrics()
                test_f1 = test_metrics["f1"]
                print(f"Epoch {epoch + 1} best val f1 {best_val_f1:.4f} test f1 {test_f1}")

        return test_metrics