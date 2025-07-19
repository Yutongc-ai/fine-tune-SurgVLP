import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics
device = "cuda" if torch.cuda.is_available() else "cpu"

class ZeroShot(nn.Module):
    # Attention pooling, no classifier, no training
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs
        self.attn_pooling = configs.attention_pooling

        self.temperature = 100
        self.threshold = 0.5

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

                logits = self.temperature * image_features @ feats_templates.T
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
        negated_templates = dataset.negated_templates

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
        

        return metrics