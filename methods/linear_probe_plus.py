import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from methods.utils import multilabel_metrics, WarmupCosineAnnealing
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features, Cholec80FeaturesVal
import wandb
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
        self.lr = configs.learning_rate
        self.lr_alpha = 0.01
        self.init_alpha = configs.init_alpha
        self.epochs = configs.epochs

        self.annealling = configs.annealling
        
        self.feature_width = configs.model_config.backbone_img.num_classes
        self.num_classes = configs.dataset_config.num_classes

        self.checkpoint_path = configs.checkpoint_path
        
        self.accumulate_step = configs.accumulate_step
        self.batch_size = configs.batch_size
        print(f"accumulate step: {self.accumulate_step}")

        self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False
        # unfreeze settings

        self.unfreeze_vision = configs["unfreeze_vision"]
        self.unfreeze_text = configs["unfreeze_text"]
        
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
                global_image_features, local_image_features, label, _ = feature_loader[_]
                local_image_features = local_image_features.to(device)
                global_image_features = global_image_features.to(device)
                local_image_features = local_image_features.permute(0, 3, 1, 2)

                if self.attn_pooling:
                    local_image_features = self.model.attention_pooling(local_image_features, self.templates)
                else:
                    local_image_features = global_image_features

                local_image_features = local_image_features / local_image_features.norm(dim = -1, keepdim = True)
                logits = self.compute_logits(local_image_features)
                probs = logits.sigmoid()

                all_probs.append(probs)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def compute_logits(self, features):
        vision_logits = self.classifier(features)
        text_logits = self.temperature * features @ self.text_embeddings.T

        logits = (vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits) / (1 + self.alpha_vec[0])
        
        return logits
    
    def forward(self,
                dataset: MultiLabelDatasetBase):
        wandb.init(
            project=f"lp++-few-shot-{self.configs.model_config.type}",
            name=f"{'unfreeze' if self.unfreeze_vision or self.unfreeze_text else ''}_init_alpha{str(self.init_alpha)}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
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
        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                       num_classes = dataset.num_classes)
        
        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)

        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80FeaturesVal(self.configs, "val")

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")

        train_loader = build_data_loader(data_source=train_data, batch_size = self.configs.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)

        # compute centroid
        train_features ,train_labels = [], []
        with torch.no_grad():
            for images, target,_, _ in train_loader:
                images, target = images.cuda(), target.cuda()
            
                global_image_features, image_features = self.model.extract_feat_img(images)
                if self.attn_pooling:
                    image_features = self.model.attention_pooling(image_features, self.templates)
                else:
                    image_features = global_image_features
                image_features = image_features / (image_features.norm(dim = 1, keepdim = True) + 1e-8)

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
    
        optim_params = [
            {'params': self.classifier.parameters(), 'lr': self.lr},
            {'params': self.alpha_vec, 'lr': self.lr}
        ]

        if self.unfreeze_vision:
            optim_params.append({'params': self.model.backbone_img.parameters(), 'lr': self.lr * 0.01})
        
        if self.unfreeze_text:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr * 0.01})

        self.optimizer = torch.optim.AdamW(optim_params)
        if self.annealling:
            self.scheduler = WarmupCosineAnnealing(self.optimizer, warmup_epochs=5, total_epochs=self.epochs, train_loader_length=math.ceil(len(train_loader) / self.accumulate_step))
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10 * len(train_loader) / self.accumulate_step)

        train_loss = []
        best_val_map = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            self.classifier.train()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.train()
            
            for i, (images, target, _, _) in enumerate(train_loader):
                images, target = images.cuda(), target.cuda()
                
                if self.unfreeze_vision or self.unfreeze_text:
                    global_image_features, image_features = self.model.extract_feat_img(images)
                    if self.attn_pooling:
                        image_features = self.model.attention_pooling(image_features, self.templates)
                    else:
                        image_features = global_image_features
                else:
                    with torch.no_grad():
                        global_image_features, image_features = self.model.extract_feat_img(images)
                        if self.attn_pooling:
                            image_features = self.model.attention_pooling(image_features, self.templates)
                        else:
                            image_features = global_image_features
                
                image_features = image_features / image_features.norm(dim = -1, keepdim = True)

                assert not torch.isnan(image_features).any(), "NaN in image_features"
                assert not torch.isinf(image_features).any(), "Inf in image_features"
                assert not torch.isnan(target).any(), "NaN in target"

                logits = self.compute_logits(image_features)

                if torch.isnan(logits).any():
                    print("NaN in logits before loss calculation!")
                loss = self.criterion(logits, target) / self.accumulate_step
                loss.backward()

                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader):
                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} Alpha: {self.alpha_vec[0]}')
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
            wandb.log({"mean_alpha": self.alpha_vec[0].mean(), "epoch": epoch})

            train_loss.append(avg_epoch_loss)
            
            self.classifier.eval()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.eval()
            
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
                print(f"Epoch {epoch + 1} best val map {best_val_map:.4f} test map {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch})
        
        wandb.finish()

        return test_metrics