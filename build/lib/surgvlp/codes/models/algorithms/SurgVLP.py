"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torch.nn as nn
from ..backbones.img_backbones import *
from ..backbones.text_backbones import *
from ...registry import MODELS
import math

@MODELS.register_module()
class SurgVLP(nn.Module):
    def __init__(self,
                 backbone_img: dict,
                 backbone_text: dict,
                 neck=None, # Optional[dict] 
                 head= None, # Optional[dict] 
                 pretrained= None, # Optional[str] 
                 ):

        super().__init__()

        self.backbone_img = MODELS.build(backbone_img)
        self.backbone_text = MODELS.build(backbone_text)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

    @property
    def dtype(self):
        return self.backbone_img.model.conv1.weight.dtype

    @property
    def with_neck(self) -> bool:
        """Check if the model has a neck module."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Check if the model has a head module."""
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_target_generator(self) -> bool:
        """Check if the model has a target_generator module."""
        return hasattr(
            self, 'target_generator') and self.target_generator is not None

    def extract_feat_img(self,
                     inputs, # : List[torch.Tensor]
                     ):
        """The forward function to extract features from neck.
        Args:
            inputs (List[torch.Tensor]): The input videos.
        Returns:
            Tuple[torch.Tensor]: visual feature.
        """
        img_emb_g, img_emb_l = self.backbone_img(inputs)
        return img_emb_g, img_emb_l

    def extract_feat_text(self,ids, attn_mask, token_type):
        """The forward function to extract features from neck.
        Args:
            inputs (List[torch.Tensor]): The input texts.
        Returns:
            Tuple[torch.Tensor]: textual feature.
        """
        text_emb_l, text_emb_g, sents = self.backbone_text(ids, attn_mask, token_type)
        return text_emb_l, text_emb_g, sents

    def loss(self, inputs):
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, inputs[0], mask)
        losses = dict(loss=loss)
        return losses
    
    def average_pooling(self, test_features):
        x = nn.AdaptiveAvgPool2d((1, 1))(test_features)
        x = x.view(x.shape[0], -1)
        x = self.backbone_img.global_embedder(x)
        x = x / x.norm(dim=-1, keepdim = True)

        return x
    
    def sep_attention_pooling(self, test_features, text_embeddings):
        batch_size = test_features.shape[0]
        local_feature_dim = test_features.shape[1]
        test_features = test_features.view(batch_size, local_feature_dim, -1).transpose(1, 2) # result into (batch_size. H*W, dim)
        test_features = self.backbone_img.global_embedder(test_features) # result into (bs, HW, 768)
        

    def attention_pooling(self, test_features, templates):
        """
            test_features: (batch_size, dim (2048), H(7), W(7))
        """
        batch_size = test_features.shape[0]
        local_feature_dim = test_features.shape[1]
        test_features = test_features.view(batch_size, local_feature_dim, -1).transpose(1, 2) # result into (batch_size. H*W, dim)
        test_features = self.backbone_img.global_embedder(test_features) # result into (bs, HW, 768)
        test_features = test_features / test_features.norm(dim = -1, keepdim = True)
        # text_features = text_features.transpose(1, 2) # result into (bs, 768, HW)

        input_ids = templates['input_ids']
        token_type_ids = templates['token_type_ids']
        attention_masks = templates['attention_mask']
        
        # (num_classes, dim)
        _, feats_templates, _ = self.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
        feats_templates = feats_templates / (feats_templates.norm(dim = -1, keepdim = True) + 1e-8)
        mean_templates = feats_templates.mean(dim = 0, keepdim=True)
        # (batch_size, 1, dim)
        query_templates = mean_templates.repeat(batch_size, 1, 1)
        
        attn_dim = query_templates.shape[-1]
        attn_scale = 1 / math.sqrt(attn_dim)

        attn_scores = torch.bmm(query_templates, test_features.transpose(1, 2))
        attn_scores = attn_scores * attn_scale
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, 1, H*W]
        
        # attention_pooling
        # [B, 1, H*W] @ [B, H*W, attn_dim] -> [B, 1, attn_dim]
        attended_feat = torch.bmm(attn_weights, test_features)
        attended_feat = attended_feat.squeeze(1)
        
        return attended_feat

    def extract_features(self, inputs_img=None, inputs_text=None, mode= 'all', fuse = False, classnames = None):
        if inputs_text is not None:
            input_ids = inputs_text['input_ids']
            token_type_ids = inputs_text['token_type_ids']
            attention_masks = inputs_text['attention_mask']

        if mode == 'video':
            feats_img, feats_img_local = self.extract_feat_img(inputs_img)

            if not fuse:
                return {'img_emb': feats_img}
            
            self.attention_pooling(feats_img_local, classnames)

        elif mode == 'text':
            feats_text_local, feats_text_global, sents = self.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            return {'text_emb': feats_text_global}
        elif mode == 'all':
            feats_img = self.extract_feat_img(inputs_img)

            feats_text_local, feats_text_global, sents = self.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            
            return {'img_emb': feats_img, 'text_emb':feats_text_global}

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
