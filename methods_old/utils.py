import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict
import torch
from torchmetrics import AveragePrecision
from torchmetrics.classification import MultilabelAveragePrecision
device = "cuda" if torch.cuda.is_available() else "cpu"

ap_metric = MultilabelAveragePrecision(num_labels = 7, average = None, thresholds = None)

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
        "method": method_name,
        "attn_pool": configs.get("attention_pooling", "NA"),
        "precision_avg": metrics.get("precision_avg", -1),
        "recall_avg": metrics.get("recall_avg", -1),
        "f1_avg": metrics.get("f1_avg", -1),
        "mAP_avg": metrics.get("mAP_avg", -1),
        "batch_size": configs.get("batch_size", -1),
        "num_shots": configs.get("num_shots", -1),
        "tasks": str(configs.get("tasks", -1)),
        "epochs": configs.get("epochs", -1),
        "unfreeze": configs.get("unfreeze", False),
        "unfreeze_layer": configs.get("unfreeze_layer", "NA"),
        "init_alpha": configs.get("init_alpha", "NA"),
    }
    if not row_data["unfreeze"]:
        row_data["unfreeze_layer"] = "NA"

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