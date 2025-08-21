import torch.nn as nn
from torch.nn import functional as F

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']

class_freq = [56800, 4106, 48437, 1624, 3217, 5384, 5760]
N_total = 86304

class WeightedInfoNCE(nn.Module):
    def __init__(self, temperature=100, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

        class_weights = [N_total / freq for freq in class_freq]
        class_weights = torch.tensor(class_weights)
        normalized_class_weights = class_weights / max(class_weights)
        self.normalized_class_weights = normalized_class_weights.cuda()

    def forward(self, query, positive_key, negative_keys=None, class_weights = None):
        if class_weights is None:
            return self.weighted_info_nce(query, positive_key, negative_keys,
                            temperature=self.temperature,
                            reduction=self.reduction,
                            negative_mode=self.negative_mode,
                            class_weights=self.normalized_class_weights)
        else:
            return self.weighted_info_nce(query, positive_key, negative_keys,
                            temperature=self.temperature,
                            reduction=self.reduction,
                            negative_mode=self.negative_mode,
                            class_weights=self.normalized_class_weights)

    def weighted_info_nce(self, query, positive_key, negative_keys=None, temperature=100, reduction='mean', 
                          negative_mode='unpaired', class_weights = None):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        # if positive_key.dim() != 2:
        #     raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        # no need for us cause alr conducted normalization
        # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
        # Explicit negative keys

        # Cosine between positive pairs

        # print("multiple positive keys")
        unsqueeze_query = query.unsqueeze(1)
        # print(unsqueeze_query)
        # print(positive_key)
        # print(transpose(positive_key))
        positive_logit = unsqueeze_query @ transpose(positive_key) * temperature
        # print("positive logit\n", positive_logit)
        positive_logit = positive_logit.squeeze(1)
        # positive logit: [bs, num_positive]
        
        # query [bs, 1, embedding_length] negative key transpose [bs, embedding_length, num_negative]
        query = query.unsqueeze(1)
        negative_logits = query @ transpose(negative_keys) * temperature
        # negative logits [bs, 1, num_negative]
        negative_logits = negative_logits.squeeze(1)
        # negative logits [bs, num_negative]

        exp_positive_logits = torch.exp(positive_logit)
        sumed_exp_positive = exp_positive_logits @ self.normalized_class_weights

        exp_negative_logits = torch.exp(negative_logits)
        sumed_exp_negative = exp_negative_logits.sum(dim = 1)

        log_term = torch.log(sumed_exp_positive / (sumed_exp_positive + sumed_exp_negative))
        losses = -log_term
        losses = torch.mean(losses)

        return losses

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=100, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=100, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    # if positive_key.dim() != 2:
    #     raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    # no need for us cause alr conducted normalization
    # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        if positive_key.dim() != 2:
            # print("multiple positive keys")
            unsqueeze_query = query.unsqueeze(1)
            # print(unsqueeze_query)
            # print(positive_key)
            # print(transpose(positive_key))
            positive_logit = unsqueeze_query @ transpose(positive_key) * temperature
            # print("positive logit\n", positive_logit)
            positive_logit = positive_logit.squeeze(1)
            # positive logit: [bs, num_positive]
        else:
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
            # positive logits: [bs, 1]

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            # query [bs, embedding_length] negative key transpose [embedding_length, num_negative]
            negative_logits = query @ transpose(negative_keys)
            # negative logits [bs, num_negative]

        elif negative_mode == 'paired':
            # query [bs, 1, embedding_length] negative key transpose [bs, embedding_length, num_negative]
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys) * temperature
            # negative logits [bs, 1, num_negative]
            negative_logits = negative_logits.squeeze(1)
            # negative logits [bs, num_negative]

        if positive_key.dim() != 2:
            exp_positive_logits = torch.exp(positive_logit)
            sumed_exp_positive = exp_positive_logits.sum(dim = 1)

            exp_negative_logits = torch.exp(negative_logits)
            sumed_exp_negative = exp_negative_logits.sum(dim = 1)

            log_term = torch.log(sumed_exp_positive / (sumed_exp_positive + sumed_exp_negative))
            losses = -log_term
            losses = torch.mean(losses)

            return losses, positive_logit, negative_logits

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device, dtype=float)
    
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class NegationLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.loss_func = nn.BCEWithLogitsLoss()
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_logits(self, image_features, text_features, logit_scale):
        
        logits = logit_scale * image_features @ text_features.T
        
        return logits

    def forward(self, image_features, text_features, logit_scale, labels, output_dict=False):


        logits = self.get_logits(image_features, text_features, logit_scale)
        
        loss = self.loss_func(logits, labels)

        return {"contrastive_loss": loss} if output_dict else loss
    
if __name__ == "__main__":
    loss_func = InfoNCE(negative_mode = "paired")
    
    batched_image = [
        [2, 3],
        [4, 5]
    ] # batch_size * embedding_length
    
    batched_image = torch.tensor(batched_image, dtype=float)

    pos_text_embedding = [
        [
            [1, 4],
            [5, 6],
        ],
        [
            [0, 1],
            [0, 0],
        ]
    ]

    pos_text_embedding = torch.tensor(pos_text_embedding, dtype=float)

    neg_text_embedding = [
        [
            [-1, -4],
            [-5, -6],
        ],
        [
            [0, -1],
            [0, 0],
        ]
    ]

    neg_text_embedding = torch.tensor(neg_text_embedding, dtype=float)

    loss = loss_func(batched_image, pos_text_embedding, neg_text_embedding)
    
    print("=====================")
    print(loss)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos = one_sided_w * los_pos
            los_neg = one_sided_w * los_neg

        loss = - (los_pos + los_neg)
        return loss.mean()

