import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict
import torch
from tqdm import tqdm
from torchmetrics import AveragePrecision
from torchmetrics.classification import MultilabelAveragePrecision
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

ap_metric = MultilabelAveragePrecision(num_labels = 7, average = None, thresholds = None)

def cal_phase_metrics(logits: torch.Tensor, ground_truth: torch.Tensor, average_type: str = 'macro'):
    """
    计算多类别分类任务的准确率和 F1 分数。

    Args:
        logits (torch.Tensor): 形状为 [bs, num_class] 的模型输出 (未经 softmax 或已 softmax)。
        ground_truth (torch.Tensor): 形状为 [bs, num_class] 的 one-hot 编码的真实标签。
        average_type (str): F1 分数的平均类型。可以是 'micro', 'macro', 'weighted', 'None'。
                            推荐 'weighted' 或 'macro'。

    Returns:
        tuple: (accuracy, f1)
    """

    # 1. 从 logits 获取预测的类别索引
    # torch.argmax 默认在指定维度上返回最大值的索引
    # dim=1 表示在每个样本的类别维度上找最大值
    predicted_classes = torch.argmax(logits, dim=1)

    # 2. 从 one-hot 编码的 ground_truth 获取真实的类别索引
    # 同样，在类别维度上找 1 所在的位置
    true_classes = torch.argmax(ground_truth, dim=1)

    # 将 PyTorch 张量转换为 NumPy 数组，因为 sklearn.metrics 函数通常接受 NumPy 数组
    predicted_classes_np = predicted_classes.cpu().numpy()
    true_classes_np = true_classes.cpu().numpy()

    # 3. 计算准确率
    accuracy = accuracy_score(true_classes_np, predicted_classes_np)

    # 4. 计算 F1 分数
    # average 参数对于多分类 F1 非常重要
    # - 'micro': 全局计算 TP, FP, FN，然后计算 F1。适用于类别不平衡的情况，但可能无法反映少数类的性能。
    # - 'macro': 为每个类别计算 F1，然后取平均。所有类别权重相同。
    # - 'weighted': 为每个类别计算 F1，然后按该类别在真实标签中的出现频率加权平均。推荐用于类别不平衡的情况。
    # - 'None': 返回每个类别的 F1 分数数组。
    f1 = f1_score(true_classes_np, predicted_classes_np, average=average_type)

    return {
        "phase_acc" : accuracy,
        "phase_f1": f1,
    }

def multilabel_metrics(targets, probs, threshold = 0.5):
    preds = (probs > threshold).int()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    micro_prec = tp_micro / (tp_micro + fp_micro + 1e-10)
    micro_rec = tp_micro / (tp_micro + fn_micro + 1e-10)
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec + 1e-10)
    
    targets = targets.int()
    ap = ap_metric(probs, targets)
    mAP = ap.mean()

    return {
        'precision': micro_prec.item(),
        'recall': micro_rec.item(),
        'f1': micro_f1.item(),
        'mAP': mAP.item(),
        'ap': ap,
    }

def save_experiment(configs, method_name, metrics, save_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, f"{method_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    result_path = os.path.join(exp_dir, "metrics.json")
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to: {exp_dir}")

def update_result_csv(
    method_name: str,
    metrics: dict,
    configs: dict,
    csv_path: str = "results.csv"
):
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "time_cost": 
        "method": method_name,
        "attn_pool": configs.get("attention_pooling", "NA"),
        "precision_avg": metrics.get("precision_avg", -1),
        "recall_avg": metrics.get("recall_avg", -1),
        "f1_avg": metrics.get("f1_avg", -1),
        "mAP_avg": metrics.get("mAP_avg", -1),
        "p_acc_avg": metrics.get("phase_acc_avg", -1),
        "p_f1_avg": metrics.get("phase_f1_avg", -1),
        "we_precision_avg": metrics.get("we_precision_avg", -1),
        "we_recall_avg": metrics.get("we_recall_avg", -1),
        "we_f1_avg": metrics.get("we_f1_avg", -1),
        "we_mAP_avg": metrics.get("we_mAP_avg", -1),
        "we_p_acc_avg": metrics.get("we_phase_acc_avg", -1),
        "we_p_f1_avg": metrics.get("we_phase_f1_avg", -1),
        "batch_size": int(configs.get("batch_size", -1)) * int(configs.get("accumulate_step", -1)),
        "early_stop": configs.get("patience") if configs.get("early_stop") else "NA",
        "annealling": configs.get("annealling", False),
        "lr": configs.get("learning_rate", -1),
        "num_shots": configs.get("num_shots", -1),
        "tasks": str(configs.get("tasks", -1)),
        "epochs": configs.get("epochs", -1),
        "unfreeze_vision": configs.get("unfreeze_vision", False),
        "unfreeze_text": configs.get("unfreeze_text", False),
        "init_alpha": configs.get("init_alpha", "NA"),
        "clip_ad_alpha": metrics.get("clip_ad_alpha_avg", "NA"),
        "backbone": configs.model_config.type,
        "finetune": configs.get("finetune", "NA"),
        "tip_alpha": configs.get("alpha", "NA"),
        "tip_beta": configs.get("beta", "NA"),
    }

    df = pd.DataFrame([row_data])
    
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def aggregate_metrics(metrics_list: List[Dict], configs: Dict) -> Dict:
    aggregated = {
        "batch_size": configs.get("batch_size", -1),
        "num_shots": configs.get("num_shots", -1),
        "tasks": str(configs.get("tasks", -1)),
        "epochs": configs.get("epochs", -1),
    }
    
    if not metrics_list:
        return aggregated
    
    metric_names = metrics_list[0].keys()
    for name in metric_names:
        values = [m[name] for m in metrics_list]
        
        if isinstance(values[0], torch.Tensor):
            values = torch.stack(values, dim = 0).cpu()
            # 在 aggregate_metrics 函数中
            mean_np = values.mean(dim=0).numpy()
            aggregated[f"{name}_avg"] = [round(float(x), 4) for x in mean_np]  # 直接格式化每个元素

            std_np = values.std(dim=0).numpy()
            aggregated[f"{name}_std"] = [round(float(x), 4) for x in std_np]

            min_np = values.min(dim=0).values.numpy()
            aggregated[f"{name}_min"] = [round(float(x), 4) for x in min_np]

            max_np = values.max(dim=0).values.numpy()
            aggregated[f"{name}_max"] = [round(float(x), 4) for x in max_np]

        else:
            values = np.array(values)
        
            aggregated[f"{name}_avg"] = round(float(np.mean(values)), 4)
            aggregated[f"{name}_std"] = round(float(np.std(values)), 4)
            aggregated[f"{name}_min"] = round(float(np.min(values)), 4)
            aggregated[f"{name}_max"] = round(float(np.max(values)), 4)
    
    return aggregated

class WarmupCosineAnnealing:
    def __init__(self, optimizer, warmup_epochs, total_epochs, train_loader_length):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.train_loader_length = train_loader_length
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10 * self.train_loader_length
        )
        self.warmup_steps = warmup_epochs * train_loader_length
        self.step_count = 1

        self.initial_lr = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.initial_lr.append(param_group["lr"])
    
    def step(self):
        if self.step_count < self.warmup_steps:
            # print("===============================================")
            # print(len(self.optimizer.param_groups))
            for i, param_group in enumerate(self.optimizer.param_groups):
                # 使用当前参数组的学习率进行 warmup
                # print(f"initial lr: {initial_lr}")
                warmup_lr = (self.step_count / self.warmup_steps) * self.initial_lr[i]
                
                # 更新当前参数组的学习率
                param_group['lr'] = warmup_lr
                # print(warmup_lr)

        else:
            self.scheduler.step()
        
        self.step_count += 1

def build_cache_model(cfg, clip_model, train_loader_cache, load_cache = False):
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            for i, (images, target, _) in enumerate(tqdm(train_loader_cache)):
                # images and target are batch variables
                images, target = images.cuda(), target.cuda()
                image_features, _ = clip_model.extract_feat_img(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                cache_keys.append(image_features)
                cache_values.append(target)
            
        cache_keys = torch.cat(cache_keys, dim = 0)
        cache_values = torch.cat(cache_values, dim = 0)

        # transform cache keys 
        cache_keys = cache_keys.permute(1, 0)

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['cur_task']) + '_' + str(cfg['num_shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['cur_task']) + '_' + str(cfg['num_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['cur_task']) + '_' + str(cfg['num_shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['cur_task']) + '_' + str(cfg['num_shots']) + "shots.pt")

    return cache_keys, cache_values

def search_hp_tip(cfg, affinity, clip_logits, cache_values, labels):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_mAP = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                # affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # clip_logits = 100. * features @ clip_weights
                tip_logits = (clip_logits + cache_logits * alpha) / (1 + alpha)
                tip_prob = tip_logits.sigmoid().cpu()
                metrics = multilabel_metrics(labels, tip_prob)
                cur_map = metrics['mAP']

                if cur_map > best_mAP:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; mAP: {:.4f}".format(beta, alpha, cur_map))
                    best_mAP = cur_map
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best mAP: {:.2f}.\n".format(best_mAP))

    return best_beta, best_alpha

class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]