import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features
from methods.utils import multilabel_metrics
from tqdm import tqdm
import wandb
import copy
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_centroid(features, labels):
    """
    features: (1, K*N, dim)
    labels: (1, K*N, num_classes)
    """
    labels = labels.transpose(1, 2)
    centroids = labels.bmm(features)
    
    return centroids

def calculate_init_alpha(features, labels, shots, text_weights):
    alpha_tilde = (compute_centroid((features @ text_weights.T).unsqueeze(0), labels.unsqueeze(0)) / 7 )[0]
    alpha_tilde = alpha_tilde.double()
    alpha_init = 250 / shots * alpha_tilde
    final_init_alpha_mean = torch.mean(alpha_init)
    return final_init_alpha_mean

class LPPlus2(nn.Module):
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
        self.lr_alpha = 0.1
        self.init_alpha = configs.init_alpha
        self.epochs = configs.epochs
        
        self.feature_width = configs.model_config.backbone_img.num_classes
        self.num_classes = configs.dataset_config.num_classes

        self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)

        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
        # unfreeze settings

        self.unfreeze = configs["unfreeze"]
    
        self.unfreeze_layer = configs.unfreeze_layer
        if self.unfreeze:
            if self.unfreeze_layer == 'last':
                for param in self.model.backbone_img.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfreeze layer4 of vision encoder")
        else:
            print("Keep vision encoder frozen")
            # elif self.unfreeze_layer == 'last2':
            #     for param in self.model.layer3.parameters():
            #         param.requires_grad = True
            #     for param in self.model.layer4.parameters():
            #         param.requires_grad = True
            #     print("Unfrozen layer3 and layer4 of vision encoder")

    def get_test_metrics(self):
        # total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in self.test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                if self.attn_pooling:
                    attn_batch_features = self.model.attention_pooling(batch_features, self.templates)  # (bs, 768)
                else:
                    attn_batch_features = self.model.average_pooling(batch_features)  # (bs, 768)
                
                batch_logits = self.compute_logits(attn_batch_features)
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
                
                batch_logits = self.compute_logits(attn_batch_features)

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

    def compute_logits(self, features):
        vision_logits = self.classifier(features)
        text_logits = self.temperature * features @ self.text_embeddings.T

        logits = (vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits) / (1 + self.alpha_vec[0])
        
        return logits
    
    def compute_training_logits(self, features):
        vision_logits = self.classifier(features)

        text_logits = self.temperature * features @ self.text_embeddings.T

        # print(vision_logits)
        # print(torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits)
        # print(1 + self.alpha_vec[0])
        # print(vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits)
        logits = (vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits) / (1 + self.alpha_vec[0])

        return logits

    def forward(self,
                dataset: MultiLabelDatasetBase):
        wandb.init(
            project="few-shot-surgvlp",
            name=f"lpplus_{'attn' if self.attn_pooling else ''}_{'unfreeze' if self.unfreeze else ''}_init_alpha{str(self.init_alpha)}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
        )
        templates = dataset.templates

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        input_ids = self.templates['input_ids']
        token_type_ids = self.templates['token_type_ids']
        attention_masks = self.templates['attention_mask']
        
        # (num_classes, dim)
        with torch.no_grad():
            _, self.text_embeddings, _ = self.model.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim = -1, keepdim = True)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.configs.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        test_features, test_labels = preload_local_features(self.configs, "test", self.model, test_loader)

        self.test_dataset = TensorDataset(test_features, test_labels)
        batch_size = self.configs.batch_size
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        del test_features, test_labels

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
        val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)
        val_loader = build_data_loader(data_source=val_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                       num_classes = dataset.num_classes)

        # compute centroid
        train_features ,train_labels = [], []
        with torch.no_grad():
            for images, target in train_loader:
                images, target = images.cuda(), target.cuda()
            
                _, image_features = self.model.extract_feat_img(images)
                image_features = image_features / (image_features.norm(dim = 1, keepdim = True) + 1e-8)
                if self.attn_pooling:
                    image_features = self.model.attention_pooling(image_features, self.templates)
                else:
                    image_features = self.model.average_pooling(image_features)

                train_features.append(image_features)
                train_labels.append(target)
        
        train_features = torch.cat(train_features)
        train_labels = torch.cat(train_labels)

        classifier_init_w = compute_centroid(train_features.unsqueeze(0), train_labels.unsqueeze(0))[0]
        classifier_init_w = classifier_init_w / classifier_init_w.norm(dim = -1, keepdim = True)
        self.classifier.weight.data = classifier_init_w

        # init_alpha = calculate_init_alpha(train_features, train_labels, self.configs.num_shots, self.text_embeddings)
        init_alpha = self.init_alpha
        
        self.alpha_vec = torch.full(
            size=(1, self.num_classes),
            fill_value=float(init_alpha),
            dtype=train_features.dtype,
            device='cuda',
            requires_grad=True
        )
    
        self.criterion = torch.nn.BCEWithLogitsLoss()
        optim_params = [
            {'params': self.classifier.parameters(), 'lr': self.lr},
            {'params': self.alpha_vec, 'lr': self.lr}
        ]

        vision_params = []
        if self.unfreeze:
            
            if self.unfreeze_layer == 'last':
                vision_params.append({'params': self.model.backbone_img.model.layer4.parameters(), 'lr': self.lr * 0.1})

        optim_params.extend(vision_params)
        self.optimizer = torch.optim.Adam(optim_params, weight_decay=0.1)

        test_metrics = self.get_test_metrics()

        train_loss = []
        best_val_f1 = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            self.classifier.train()
            if self.unfreeze:
                self.model.backbone_img.model.train()
            
            for i, (images, target) in enumerate(train_loader):
                images, target = images.cuda(), target.cuda()
                
                if self.unfreeze:
                    _, image_features = self.model.extract_feat_img(images)
                    image_features = image_features / (image_features.norm(dim = 1, keepdim = True) + 1e-8)
                    if self.attn_pooling:
                        image_features = self.model.attention_pooling(image_features, self.templates)
                    else:
                        image_features = self.model.average_pooling(image_features)
                else:
                    with torch.no_grad():
                        _, image_features = self.model.extract_feat_img(images)
                        image_features = image_features / (image_features.norm(dim = 1, keepdim = True) + 1e-8)
                        if self.attn_pooling:
                            image_features = self.model.attention_pooling(image_features, self.templates)
                        else:
                            image_features = self.model.average_pooling(image_features)

                assert not torch.isnan(image_features).any(), "NaN in image_features"
                assert not torch.isinf(image_features).any(), "Inf in image_features"
                assert not torch.isnan(target).any(), "NaN in target"

                self.optimizer.zero_grad()
                logits = self.compute_training_logits(image_features)

                if torch.isnan(logits).any():
                    print("NaN in logits before loss calculation!")
                loss = self.criterion(logits, target)

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} Alpha: {self.alpha_vec[0]}')
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
            wandb.log({"mean_alpha": self.alpha_vec[0].mean(), "epoch": epoch})

            train_loss.append(avg_epoch_loss)
            
            self.classifier.eval()
            if self.unfreeze:
                self.model.backbone_img.model.eval()
            
            val_metrics = self.get_val_metrics(val_loader)
            cur_val_f1 = val_metrics["f1"]
            wandb.log({"val_f1": cur_val_f1, "epoch": epoch})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch})

            if cur_val_f1 > best_val_f1:
                best_val_f1 = cur_val_f1
                test_metrics = self.get_test_metrics()
                test_f1 = test_metrics["f1"]
                print(f"Epoch {epoch + 1} best val f1 {best_val_f1:.4f} test f1 {test_f1}")
                wandb.log({"test_f1": test_f1, "epoch": epoch})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch})
        
        wandb.finish()

        return test_metrics