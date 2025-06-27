import torch
import surgvlp
from mmengine.config import Config
import random
from datasets.cholec80 import Cholec80
from datasets.utils import *
from methods.zero_shot import ZeroShot
from methods.linear_probe import LP
from methods.linear_probe_plus import LPPlus2
from methods.zoom_in import ZoomIn
import argparse
import datetime
from methods.utils import *
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(configs, method_name):
    torch.cuda.reset_peak_memory_stats()

    # Load config file

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = Cholec80(config=configs.dataset_config)
    classnames = dataset.classnames
    templates = dataset.templates
    print("dataset class names:", classnames)
    print("template:", templates)

    experiment_data = {
        "classnames": classnames,
        "templates": templates,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if method_name == "zero_shot_attn_pooling" or method_name == "zero_shot_average_pooling":
        tasks = 1
    else:
        tasks = configs.tasks

    all_metrics = []
    for task in range(tasks):
        random.seed(task+200)
        torch.manual_seed(task+200)
        surgvlp_model, preprocess = surgvlp.load(configs.model_config, device='cuda', pretrain='/home/yongxuan/SurgVLP/checkpoints/SurgVLP.pth')

        if method_name == "zero_shot_attn_pooling" or method_name == "zero_shot_average_pooling":
            method = ZeroShot(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "avg_pooling_lp" or method_name == "attn_pooling_lp":
            method = LP(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "avg_pooling_lpplus" or method_name == "attn_pooling_lpplus":
            method = LPPlus2(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "zoom_in":
            method = ZoomIn(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        metrics = method(dataset)
        all_metrics.append(metrics)
    
    agg_metrics = aggregate_metrics(all_metrics, configs)

    experiment_data.update({
        "metrics": agg_metrics,
        "method_class": method_name
    })
    save_experiment(configs, method_name, experiment_data)
    update_result_csv(method_name, agg_metrics, configs)
    
    print("===========================results===========================")
    print(agg_metrics)
    wandb.finish()

if __name__ == "__main__":
    configs = Config.fromfile('./tests/config_surgvlp_few_shot.py')['config']
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', default="zero_shot_attn_pooling")
    args = parser.parse_args()
    
    main(configs[args.method], args.method)