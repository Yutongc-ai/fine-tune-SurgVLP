import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features
from methods.utils import multilabel_metrics
from torch.utils.data import TensorDataset, DataLoader
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

    def get_test_metrics(self):
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_features, batch_labels in self.test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            if self.attn_pooling:
                attn_batch_features = self.model.attention_pooling(batch_features, self.templates)  # (bs, 768)
            else:
                attn_batch_features = self.model.average_pooling(batch_features)  # (bs, 768)
            
            batch_logits = self.temperature * attn_batch_features @ self.text_embeddings.T  # (bs, 7)
            batch_prob = batch_logits.sigmoid()
            batch_pred = (batch_prob > self.threshold).int()
            
            total_loss += nn.BCEWithLogitsLoss()(batch_logits, batch_labels).item() * batch_features.size(0)
            all_preds.append(batch_pred.cpu())
            all_labels.append(batch_labels.cpu())

        avg_loss = total_loss / len(self.test_loader)
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        metrics = multilabel_metrics(final_labels, final_preds)
        
        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):
        templates = dataset.templates

        self.templates = self.tokenizer(templates, device = device)
        input_ids = self.templates['input_ids']
        token_type_ids = self.templates['token_type_ids']
        attention_masks = self.templates['attention_mask']
        
        # (num_classes, dim)
        _, self.text_embeddings, _ = self.model.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)

        test_loader = build_data_loader(data_source=dataset.test, batch_size = 64, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        test_features, test_labels = preload_local_features(self.configs, "test", self.model, test_loader)

        self.test_dataset = TensorDataset(test_features, test_labels)
        batch_size = self.configs.batch_size
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        del test_features, test_labels

        metrics = self.get_test_metrics()

        return metrics