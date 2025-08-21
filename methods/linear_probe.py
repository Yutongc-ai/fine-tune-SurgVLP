import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics, WarmupCosineAnnealing
from tqdm import tqdm
import wandb
import math
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
        self.lr = configs.learning_rate
        self.epochs = configs.epochs

        self.annealling = configs.annealling

        self.checkpoint_path = configs.checkpoint_path
        
        self.accumulate_step = configs.accumulate_step
        self.batch_size = configs.batch_size
        print(f"accumulate step: {self.accumulate_step}")

        self.feature_width = configs.model_config.backbone_img.num_classes
        self.num_classes = configs.dataset_config.num_classes

        self.classifier = torch.nn.Linear(self.feature_width, self.num_classes, bias=False).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.norm = nn.LayerNorm(self.feature_width).to(device)

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
                global_image_features, local_image_features, label = feature_loader[_]
                local_image_features = local_image_features.to(device)
                global_image_features = global_image_features.to(device)
                local_image_features = local_image_features.permute(0, 3, 1, 2)

                if self.attn_pooling:
                    image_features = self.model.attention_pooling(local_image_features, self.templates)
                else:
                    image_features = global_image_features

                # image_features = image_features / image_features.norm(dim = -1, keepdim = True)
                image_features = self.norm(image_features)
                logits = self.classifier(image_features)
                probs = logits.sigmoid()

                all_probs.append(probs)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates

        wandb.init(
            project=f"lp-few-shot-{self.configs.model_config.type}",
            name=f"{'unfreeze' if self.unfreeze_vision or self.unfreeze_text else 'freeze'}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
        )

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']
        
        # (num_classes, dim)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.batch_size, tfm=self.preprocess, is_train=False, 
                                       num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80Features(self.configs, "val")

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")
        # val_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="val") 

        train_loader = build_data_loader(data_source=train_data, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)


        optim_params = [
            {'params': self.classifier.parameters(), 'lr': self.lr},
        ]

        if self.unfreeze_vision:
            optim_params.append({'params': self.model.backbone_img.parameters(), 'lr': self.lr * 0.01})
        
        if self.unfreeze_text:
            optim_params.append({'params': self.model.backbone_text.parameters(), 'lr': self.lr * 0.01})
        
        self.optimizer = torch.optim.AdamW(optim_params)
        if self.annealling:
            self.scheduler = WarmupCosineAnnealing(self.optimizer, warmup_epochs=5, total_epochs=self.epochs, train_loader_length=math.ceil(len(train_loader) / self.accumulate_step))
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10 * len(train_loader) / self.accumulate_step)

        val_metrics = self.get_metrics("val")
        cur_val_map = val_metrics["mAP"]
        print(f"Initial val mAP {cur_val_map}")
        wandb.log({"val_map": cur_val_map, "epoch": 0})
        wandb.log({"val_f1": val_metrics["f1"], "epoch": 0})
        wandb.log({"val_precision": val_metrics["precision"], "epoch": 0})
        wandb.log({"val_recall": val_metrics["recall"], "epoch": 0})

        test_metrics = self.get_metrics("test")
        test_map = test_metrics["mAP"]
        print(f"Initial test mAP {test_map}")
        wandb.log({"test_map": test_map, "epoch": 0})
        wandb.log({"test_f1": test_metrics["f1"], "epoch": 0})
        wandb.log({"test_precision": test_metrics["precision"], "epoch": 0})
        wandb.log({"test_recall": test_metrics["recall"], "epoch": 0})

        train_loss = []
        best_val_map = 0
        
        # =========================count sample distribution===========================
        # num_classes = 7
        # class_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

        # class_counts = torch.zeros(num_classes, dtype=torch.long)

        # # 用于统计每个样本的标签数
        # # key是标签数量(1, 2, 3...), value是拥有该数量标签的样本数
        # labels_per_sample_counts = {}

        # # 总样本数
        # total_samples = 0


        # # --- 2. 遍历 DataLoader 并进行统计 ---

        # print("开始遍历 DataLoader 进行统计...")
        # # 假设 train_loader 已定义好
        # for i, (images, targets, _) in enumerate(tqdm(train_loader)):
        #     # targets 的形状是 (batch_size, num_classes)
            
        #     # 统计总样本数
        #     batch_size = targets.size(0)
        #     total_samples += batch_size
            
        #     # 统计每个类别的样本数
        #     # targets 是 0/1 矩阵，我们按列求和（dim=0），就能得到这个批次中每个类别的出现次数
        #     # 然后累加到总的 class_counts 中
        #     class_counts += targets.sum(dim=0).long()
            
        #     # 统计每个样本的标签数
        #     # targets 按行求和（dim=1），得到一个长度为 batch_size 的张量
        #     # 其中每个元素是对应样本的标签总数
        #     labels_per_sample = targets.sum(dim=1).long()
        #     for num_labels in labels_per_sample:
        #         num = num_labels.item()
        #         labels_per_sample_counts[num] = labels_per_sample_counts.get(num, 0) + 1

                
        # # --- 3. 打印统计结果 ---

        # print(f"\ntotal samples: {total_samples}")

        # print("\nsample per class:")
        # for i in range(num_classes):
        #     count = class_counts[i].item()
        #     percentage = (count / total_samples) * 100
        #     print(f"  - {class_names[i]:<15}: {count:<5} samples ({percentage:.2f}%)")
        # assert(0)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            self.classifier.train()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.train()
            
            for i, (images, target, _) in enumerate(tqdm(train_loader)):
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
                
                # image_features = image_features / image_features.norm(dim = -1, keepdim = True)
                image_features = self.norm(image_features)

                logits = self.classifier(image_features)
                
                loss = self.criterion(logits, target) / self.accumulate_step
                loss.backward()

                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader):
                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    # print(f"Epoch {epoch}, LR: {self.optimizer.param_groups[0]['lr']}")
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                batch_count += 1
                
            
            avg_epoch_loss = epoch_loss / batch_count
            # wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})

            print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}')
            
            self.classifier.eval()
            if self.unfreeze_vision or self.unfreeze_text:
                self.model.eval()

            train_loss.append(avg_epoch_loss)

            val_metrics = self.get_metrics("val")
            cur_val_map = val_metrics["mAP"]
            wandb.log({"val_map": cur_val_map, "epoch": epoch+1})
            wandb.log({"val_f1": val_metrics["f1"], "epoch": epoch+1})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch+1})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch+1})

            if cur_val_map > best_val_map:
                best_val_map = cur_val_map
                test_metrics = self.get_metrics("test")
                test_map = test_metrics["mAP"]
                print(f"Epoch {epoch + 1} best val mAP {best_val_map:.4f} test mAP {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch+1})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})
        
        wandb.finish()
        return test_metrics