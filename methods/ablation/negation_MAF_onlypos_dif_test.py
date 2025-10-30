import torch
import torch.nn as nn
from methods.loss import InfoNCE, WeightedInfoNCE, class_freq, class_neg_freq
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features, Cholec80FeaturesVal
from methods.utils import multilabel_metrics, WarmupCosineAnnealing, cal_phase_metrics, cal_zs_phase_metrics
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
import math
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchMean(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class VisionAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(VisionAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Adapter(nn.Module):
    def __init__(self, c_in, reduction = 4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
            # nn.LayerNorm(c_in),
            # nn.ReLU(inplace=True),
            PatchMean(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class MultipleAdapter(nn.Module):
    def __init__(self, width = 768, layer = 4, is_fuse_type = 1):
        super(MultipleAdapter, self).__init__()
        self.is_fuse_type = is_fuse_type
        if self.is_fuse_type == 1:
            adapter_weight = nn.Identity()
        else:
            raise NotImplementedError()
        
        self.adapter = nn.ModuleDict({
            "adapter_1": Adapter(width, 2),  # adapter 1
            "adapter_2": Adapter(width, 2),  # adapter 2
            "adapter_3": Adapter(width, 2),  # adapter 3
            "adapter_4": Adapter(width, 2),  # adapter 4
            "adapter_weight": adapter_weight,  # adapter weight
        })
    
    def forward(self, layer_features, alpha = 0.2):
        features1, features2, features3, features4 = layer_features
        features1a = self.adapter["adapter_1"](features1)
        features2a = self.adapter["adapter_2"](features2)
        features3a = self.adapter["adapter_3"](features3)
        features4a = self.adapter["adapter_4"](features4)

        if self.is_fuse_type == 1:
            # sum mode, same with text backbone
            featuresa = (self.adapter["adapter_weight"](features1a + features2a + features3a + features4a)) / 4
        else:
            raise NotImplementedError()
        
        layer_features = torch.stack(layer_features) # [layers, batch, sent_len, embedding size]
        layer_features = layer_features.permute(1, 0, 2, 3) # [batch, layers, sent_len, embedding size]
        sent_embeddings = layer_features.mean(axis=2).sum(axis = 1)

        ret_features = alpha * featuresa + (1- alpha) * sent_embeddings
        ret_features = ret_features / ret_features.norm(dim = -1, keepdim = True)

        return ret_features


class CustomAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = VisionAdapter(768, 2)

    def forward(self, image_features, alpha = 0.2):
        x = self.adapter(image_features)

        # alpha = 0.2
        image_features = alpha * x + (1 - alpha) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

class NegationOPDif(nn.Module):
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs

        self.vision_adapter = CustomAdapter()
        self.text_adapter = MultipleAdapter()
        self.alpha = configs.alpha

        self.temperature = 100
        self.threshold = 0.5
        self.lr = configs.learning_rate
        print(f"learning rate is {self.lr}")
        self.epochs = configs.epochs
        
        self.num_classes = configs.dataset_config.num_classes

        self.patience = configs.patience
        self.annealling = configs.annealling
        self.checkpoint_path = configs.checkpoint_path + "1020.pth"
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        
        # self.criterion = torch.nn.BCEWithLogitsLoss()
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
        all_labels = []

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

                # _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                feats_outputs = self.model.backbone_text.model(self.input_ids, self.attention_masks, self.token_type_ids)
                layer_templates_feats = feats_outputs[2][-4:]
                layer_templates_feats = tuple(t.cuda() for t in layer_templates_feats)

                ada_image_features = self.vision_adapter(image_features, self.alpha)
                ada_feats_templates = self.text_adapter(layer_templates_feats, self.alpha)
                
                test_feats_outputs = self.model.backbone_text.model(self.test_input_ids, self.test_attention_masks, self.test_token_type_ids)
                test_layer_templates_feats = test_feats_outputs[2][-4:]
                test_layer_templates_feats = tuple(t.cuda() for t in test_layer_templates_feats)
                # print("test", len(test_layer_templates_feats))

                ada_test_feats_templates = self.text_adapter(test_layer_templates_feats, self.alpha)

                # tool logit
                pos_logit = ada_image_features @ ada_feats_templates[:7, :].T

                test_prompt_type_number = int(ada_test_feats_templates.shape[0] / 7)

                neg_logit = ada_image_features @ ada_test_feats_templates[: 7, :].T
                for test_prompt_type in range(1, test_prompt_type_number):
                    neg_logit = neg_logit + ada_image_features @ ada_test_feats_templates[7*test_prompt_type: 7*test_prompt_type + 7, :].T
                
                logits = pos_logit - neg_logit / test_prompt_type_number
                
                probs = logits.sigmoid()

                all_probs.append(probs)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)

        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def generate_tool_negation(self):
        prompts = []

        for tool in self.tool_names:
            template = random.choice(self.all_templates)
            prompts.append(template.format(tool = tool))
        
        return prompts
    
    def generate_all_tool_negation(self):
        prompts = []

        for template in self.all_templates:
            for tool in self.tool_names:
                prompts.append(template.format(tool = tool))

        return prompts
    
    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates
        # training negation templates
        self.all_templates = dataset.all_templates
        # test negation templates
        test_prompt = dataset.test_prompt
        self.tool_names = dataset.tool_names
        phase_templates = dataset.phase_templates
        self.random_select = self.configs["random_select"]
        training_prompt = self.generate_all_tool_negation()

        wandb.init(
            project=f"negation-few-shot-{self.configs.model_config.type}",
            name=f"maf_batchsize{self.batch_size * self.accumulate_step}_lr{self.lr}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            # mode="offline",
        )

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']

        self.test_prompt = self.tokenizer(test_prompt, device = device)
        self.test_input_ids = self.test_prompt['input_ids']
        self.test_token_type_ids = self.test_prompt['token_type_ids']
        self.test_attention_masks = self.test_prompt['attention_mask']

        self.train_prompt = self.tokenizer(training_prompt, device = device)
        self.train_input_ids = self.train_prompt['input_ids']
        self.train_token_type_ids = self.train_prompt['token_type_ids']
        self.train_attention_masks = self.train_prompt['attention_mask']

        train_feats_outputs = self.model.backbone_text.model(self.train_input_ids, self.train_attention_masks, self.train_token_type_ids)
        train_layer_templates_feats = train_feats_outputs[2][-4:]
        train_layer_templates_feats = tuple(t.cuda() for t in train_layer_templates_feats)

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


        val_metrics = self.get_metrics("val")
        val_map = val_metrics["mAP"]
        print(f"Initial val mAP {val_map}")
        wandb.log({"val_map": val_map, "epoch": 0})
        wandb.log({"val_f1": val_metrics["f1"], "epoch": 0})
        wandb.log({"val_precision": val_metrics["precision"], "epoch": 0})
        wandb.log({"val_recall": val_metrics["recall"], "epoch": 0})

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
                    
                else:
                    with torch.no_grad():
                        image_features, local_features = self.model.extract_feat_img(images)

                image_features = self.vision_adapter(image_features, self.alpha)
                # image features has been normed in vision adapter
                image_features = image_features / image_features.norm(dim = -1, keepdim = True)

                # negation_prompt = self.generate_tool_negation()
                # negation_prompt = self.tokenizer(negation_prompt, device = device)
                # negation_input_ids = negation_prompt['input_ids']
                # negation_token_type_ids = negation_prompt['token_type_ids']
                # negation_attention_masks = negation_prompt['attention_mask']
                
                # positive feats
                feats_outputs = self.model.backbone_text.model(self.input_ids, self.attention_masks, self.token_type_ids)
                layer_templates_feats = feats_outputs[2][-4:]
                layer_templates_feats = tuple(t.cuda() for t in layer_templates_feats)

                # negative feats
                # negation_feats_outputs = self.model.backbone_text.model(negation_input_ids, negation_attention_masks, negation_token_type_ids)
                # negation_layer_templates_feats = negation_feats_outputs[2][-4:]
                # negation_layer_templates_feats = tuple(t.cuda() for t in negation_layer_templates_feats)

                image_features = self.vision_adapter(image_features, self.alpha)
                # image_features = image_features / image_features.norm(dim = -1, keepdim = True)
                feats_templates = self.text_adapter(layer_templates_feats, self.alpha)
                # negation_feats_templates = self.text_adapter(negation_layer_templates_feats, self.alpha)
                # feats_templates = feats_templates / feats_templates.norm(dim = -1, keepdim = True)
                
                pos_logit = image_features @ feats_templates.T * self.temperature
                # neg_logit = image_features @ negation_feats_templates.T * self.temperature
                logits = pos_logit

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

            save_model = self.early_stopping(cur_val_map)
            if save_model:
                adapters_weights = {
                    'vision_adapter': self.vision_adapter.state_dict(),
                    'text_adapter': self.text_adapter.state_dict()
                }
                torch.save(adapters_weights, self.checkpoint_path)
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
        
        test_metrics["clip_ad_alpha"] = self.alpha

        test_metrics["train_test"] = dataset.equal
        test_metrics["random_select"] = self.random_select

        wandb.finish()
        return test_metrics