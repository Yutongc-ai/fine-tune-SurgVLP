import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features
from methods.utils import multilabel_metrics, cal_phase_metrics
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

        self.temperature = 1
        self.threshold = 0.5

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
                global_image_features, local_image_features, label, phase_label = feature_loader[_]
                local_image_features = local_image_features.to(device)
                global_image_features = global_image_features.to(device)
                local_image_features = local_image_features.permute(0, 3, 1, 2)

                image_features = global_image_features

                _, feats_templates, _ = self.model.extract_feat_text(ids=self.input_ids, attn_mask=self.attention_masks, token_type=self.token_type_ids)
                _, phase_feats_templates, _ = self.model.extract_feat_text(ids=self.phase_input_ids, attn_mask=self.phase_attention_masks, token_type=self.phase_token_type_ids)

                image_features = image_features / image_features.norm(dim = -1, keepdim = True)

                feats_templates = feats_templates / feats_templates.norm(dim = -1, keepdim = True)
                feats_templates = feats_templates[:7, :].cuda()

                logits = self.temperature * image_features @ feats_templates.T
                phase_logits = self.temperature * image_features @ phase_feats_templates.T
                
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

        # print(metrics)
        # print(phase_metrics)
        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates
        negated_templates = dataset.negated_templates
        phase_templates = dataset.phase_templates

        # test data preparations
        self.templates = self.tokenizer(templates, device = device)
        
        self.input_ids = self.templates['input_ids']
        self.token_type_ids = self.templates['token_type_ids']
        self.attention_masks = self.templates['attention_mask']
        
        self.phase_templates = self.tokenizer(phase_templates, device = device)
        
        self.phase_input_ids = self.phase_templates['input_ids']
        self.phase_token_type_ids = self.phase_templates['token_type_ids']
        self.phase_attention_masks = self.phase_templates['attention_mask']

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

        self.model.eval()

        print(self.configs.num_shots)

        metrics = self.get_metrics("test")

        return metrics