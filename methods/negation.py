import torch
import torch.nn as nn
from methods.loss import InfoNCE, WeightedInfoNCE, class_freq, class_neg_freq
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features, Cholec80FeaturesVal
from methods.utils import multilabel_metrics, WarmupCosineAnnealing, cal_phase_metrics
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

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

class CustomAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter(768, 2)

    def forward(self, image_features, alpha = 0.2):
        x = self.adapter(image_features)

        # alpha = 0.2
        image_features = alpha * x + (1 - alpha) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

class Negation(nn.Module):
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs

        self.vision_adapter = CustomAdapter()
        self.text_adapter = CustomAdapter()
        self.alpha = configs.alpha

        self.temperature = 100
        self.threshold = 0.5
        self.lr = configs.learning_rate
        print(f"learning rate is {self.lr}")
        self.epochs = configs.epochs
        
        self.num_classes = configs.dataset_config.num_classes
        self.loss_func = InfoNCE(negative_mode='paired')

        self.patience = configs.patience
        self.annealling = configs.annealling
        self.checkpoint_path = configs.checkpoint_path
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)

        self.num_shots = configs.num_shots
        if self.num_shots != -1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            # full shot
            pos_weight = [freq_pos / freq_neg for freq_pos, freq_neg in zip(class_freq, class_neg_freq)]
            pos_weight = torch.tensor(pos_weight)
            pos_weight = pos_weight.cuda()
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        
        self.accumulate_step = configs.accumulate_step
        self.batch_size = configs.batch_size
        print(f"accumulate step: {self.accumulate_step}")

        for param in self.model.parameters():
            param.requires_grad = False
        # unfreeze settings
        # vision encoder settings
        self.unfreeze_vision = configs["unfreeze_vision"]
        if self.unfreeze_vision:
            for param in self.model.backbone_img.parameters():
                param.requires_grad = True
            
            print("Unfreeze vision encoder")
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
        all_phase_logits = []
        all_labels = []
        all_phase_labels = []

        if split == "val":
            feature_loader = self.val_feature
        elif split == "test":
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")

        with torch.no_grad():

            for _ in range(len(feature_loader)):
                image_features, local_image_features, label, phase_label = feature_loader[_]
                image_features = image_features.cuda()

                _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                _, phase_feats_templates, _ = self.model.extract_feat_text(ids=self.phase_input_ids, attn_mask=self.phase_attention_masks, token_type=self.phase_token_type_ids)
                
                ada_image_features = self.vision_adapter(image_features, self.alpha)
                # ada_image_features = image_features / image_features.norm(dim = -1, keepdim = True)
                feats_templates = feats_templates.cuda()
                ada_feats_templates = self.text_adapter(feats_templates, self.alpha)
                # ada_phase_feats_templates = self.text_adapter(phase_feats_templates, self.alpha)

                # tool logit
                pos_logit = ada_image_features @ ada_feats_templates[:7, :].T
                neg_logit = ada_image_features @ ada_feats_templates[7:, :].T
                logits = pos_logit - neg_logit

                # phase logit
                image_features = image_features / image_features.norm(dim=-1, keepdim = True)
                phase_feats_templates = phase_feats_templates / phase_feats_templates.norm(dim = -1, keepdim = True)
                phase_logits = image_features @ phase_feats_templates.T
                
                probs = logits.sigmoid()

                all_probs.append(probs)
                all_phase_logits.append(phase_logits)
                all_labels.append(label)
                all_phase_labels.append(phase_label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_phase_logits = torch.cat(all_phase_logits, dim=0)
        final_phase_labels = torch.cat(all_phase_labels, dim=0)
        
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)
        phase_metrics = cal_phase_metrics(final_phase_logits, final_phase_labels)
        metrics.update(phase_metrics)

        return metrics
    
    def get_nce_labels(self, target, negated_target, feats_template):
        batch_size = target.shape[0]
        _, emb_size = feats_template.shape
        positive_keys = torch.zeros((batch_size, self.num_classes, emb_size), device=device)
        negative_keys = torch.zeros((batch_size, self.num_classes, emb_size), device=device)

        for bs in range(batch_size):
            for index, mask in enumerate(target[bs]):
                if mask:
                    positive_keys[bs][index] = feats_template[index]
                    negative_keys[bs][index] = feats_template[index + 7]

        for bs in range(batch_size):
            for index, mask in enumerate(negated_target[bs]):
                if mask:
                    # positive_keys[bs][index] = feats_template[index + 7]
                    negative_keys[bs][index] = feats_template[index]
        
        return positive_keys, negative_keys

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates
        negated_templates = dataset.negated_templates
        phase_templates = dataset.phase_templates

        wandb.init(
            project=f"negation-few-shot-{self.configs.model_config.type}",
            name=f"batchsize{self.batch_size * self.accumulate_step}_lr{self.lr}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
        )

        # test data preparations
        self.templates = self.tokenizer(templates + negated_templates, device = device)
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']

        self.phase_templates = self.tokenizer(phase_templates, device = device)
        self.phase_input_ids = self.phase_templates['input_ids']
        self.phase_token_type_ids = self.phase_templates['token_type_ids']
        self.phase_attention_masks = self.phase_templates['attention_mask']
        
        self.vision_adapter.to(device)
        self.text_adapter.to(device)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.batch_size, tfm=self.preprocess, is_train=False, 
                                       num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80FeaturesVal(self.configs, "val")

        # Generate few shot data
        if self.configs.num_shots == -1:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                             num_classes = dataset.num_classes)
        else:
            train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
            # val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

            train_loader = build_data_loader(data_source=train_data, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                            num_classes = dataset.num_classes)

        optim_params = [{'params': self.vision_adapter.parameters(), 'lr': self.lr},
                        {'params': self.text_adapter.parameters(), 'lr': self.lr}]
        # optim_params = [{'params': self.text_adapter.parameters(), 'lr': self.lr}]
        if self.unfreeze_vision:
            optim_params.append({'params': self.model.backbone_img.parameters(), 'lr': self.lr * 0.001})
        
        if self.unfreeze_text:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr * 0.001})

        self.optimizer = torch.optim.AdamW(optim_params)
        if self.annealling:
            self.scheduler = WarmupCosineAnnealing(self.optimizer, warmup_epochs=5, total_epochs=self.epochs, train_loader_length=math.ceil(len(train_loader) / self.accumulate_step))
        
        train_loss = []
        best_val_map = 0
        
        test_metrics = self.get_metrics("test")
        print(test_metrics)
        test_map = test_metrics["mAP"]
        print(f"Initial test mAP {test_map}")
        wandb.log({"test_map": test_map, "epoch": 0})
        wandb.log({"test_f1": test_metrics["f1"], "epoch": 0})
        wandb.log({"test_precision": test_metrics["precision"], "epoch": 0})
        wandb.log({"test_recall": test_metrics["recall"], "epoch": 0})
        
        wandb.log({"test_p_acc": test_metrics["phase_acc"], "epoch": 0})
        wandb.log({"test_p_f1": test_metrics["phase_f1"], "epoch": 0})

        val_metrics = self.get_metrics("val")
        val_map = val_metrics["mAP"]
        print(f"Initial val mAP {val_map}")
        wandb.log({"val_map": val_map, "epoch": 0})
        wandb.log({"val_f1": val_metrics["f1"], "epoch": 0})
        wandb.log({"val_precision": val_metrics["precision"], "epoch": 0})
        wandb.log({"val_recall": val_metrics["recall"], "epoch": 0})

        wandb.log({"val_p_acc": val_metrics["phase_acc"], "epoch": 0})
        wandb.log({"val_p_f1": val_metrics["phase_f1"], "epoch": 0})

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.train()
            self.vision_adapter.train()
            self.text_adapter.train()
            
            for i, (images, target, negated_target, _) in enumerate(tqdm(train_loader)):
                images, target, negated_target = images.cuda(), target.cuda(), negated_target.cuda()

                if self.unfreeze_vision or self.unfreeze_text:
                    # get image embedding
                    # global_features: [bs, 768] local_features: [bs, 2048, 7, 7]
                    image_features, local_features = self.model.extract_feat_img(images)
                    
                    # get text embedding
                    # [14, 768]
                    _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)

                else:
                    with torch.no_grad():
                        image_features, local_features = self.model.extract_feat_img(images)
                        _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                
                feats_templates = feats_templates.cuda()
                
                image_features = self.vision_adapter(image_features, self.alpha)
                # image features has been normed in vision adapter
                # image_features = image_features / image_features.norm(dim = -1, keepdim = True)
                feats_templates = self.text_adapter(feats_templates, self.alpha)
                # feats_templates = feats_templates / feats_templates.norm(dim = -1, keepdim = True)
                pos_logit = image_features @ feats_templates[:7, :].T * self.temperature
                neg_logit = image_features @ feats_templates[7:, :].T * self.temperature
                logits = pos_logit - neg_logit
                # positive_keys, negative_keys = self.get_nce_labels(target, negated_target, feats_templates)
                
                # loss = self.loss_func(image_features, positive_keys, negative_keys)
                loss = self.criterion(logits, target)
                loss /= self.accumulate_step
                loss.backward()

                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader):
                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log({"lr": self.optimizer.param_groups[0]['lr'], "epoch": epoch+1})

                epoch_loss += loss.item()
                batch_count += 1

                del image_features, feats_templates, loss, images, _

            avg_epoch_loss = epoch_loss * self.accumulate_step / batch_count
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.eval()
            self.vision_adapter.eval()
            self.text_adapter.eval()

            train_loss.append(avg_epoch_loss)

            val_metrics = self.get_metrics("val")
            cur_val_map = val_metrics["mAP"]
            wandb.log({"val_map": cur_val_map, "epoch": epoch+1})
            wandb.log({"val_f1": val_metrics["f1"], "epoch": epoch+1})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch+1})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch+1})

            wandb.log({"val_p_acc": val_metrics["phase_acc"], "epoch": epoch+1})
            wandb.log({"val_p_f1": val_metrics["phase_f1"], "epoch": epoch+1})

            save_model = self.early_stopping(cur_val_map)
            if save_model:
                torch.save(self.state_dict(), self.checkpoint_path)
                print(f'Validation map increased. Saving model...')

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if cur_val_map > best_val_map:
                best_val_map = cur_val_map
                test_metrics = self.get_metrics("test")
                test_map = test_metrics["mAP"]
                print(f"Epoch {epoch + 1} best val mAP {best_val_map:.4f} test mAP {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch+1})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})

                wandb.log({"test_p_acc": test_metrics["phase_acc"], "epoch": epoch+1})
                wandb.log({"test_p_f1": test_metrics["phase_f1"], "epoch": epoch+1})
        
        test_metrics["clip_ad_alpha"] = self.alpha
        wandb.finish()
        return test_metrics