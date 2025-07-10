import torch
import surgvlp
from mmengine.config import Config
import random
from datasets.cholec80 import Cholec80, NegationCholec80
from datasets.utils import *
from methods.zero_shot import ZeroShot
from methods.linear_probe import LP
from methods.linear_probe_plus import LPPlus2
from methods.zoom_in import ZoomIn
from methods.cross_attn import CrossAttn
from methods.cross_attn_residual import ResidualCrossAttn
from methods.negation import Negation
from methods.simple import Simple
from methods.aggre_negation import AggreNegation
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
    if method_name == "aggre_negation":
        dataset = NegationCholec80(config=configs.dataset_config)
    else:
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
    all_metrics_img = []
    all_metrics_text = []
    for task in range(tasks):
        random.seed(task+200)
        torch.manual_seed(task+200)
        surgvlp_model, preprocess = surgvlp.load(configs.model_config, device='cuda', pretrain='/home/yongxuan/SurgVLP/checkpoints/SurgVLP.pth')

        if method_name == "zero_shot":
            method = ZeroShot(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "linear_probe":
            method = LP(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "linear_probe++":
            method = LPPlus2(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "zoom_in":
            method = ZoomIn(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "bi_cross_attn":
            method = CrossAttn(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "residual_bi_cross_attn":
            method = ResidualCrossAttn(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "negation":
            method = Negation(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "simple":
            method = Simple(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        elif method_name == "aggre_negation":
            method = AggreNegation(configs, surgvlp_model, preprocess, surgvlp.tokenize)
        metrics = method(dataset)

        # all_metrics.append(metrics)
        if method_name != "bi_cross_attn":
            all_metrics.append(metrics)
        else:
            all_metrics_img.append(metrics[0])
            all_metrics_text.append(metrics[1])
    
    if method_name != "bi_cross_attn":
        agg_metrics = aggregate_metrics(all_metrics, configs)
    else:
        agg_metrics_img = aggregate_metrics(all_metrics_img, configs)
        agg_metrics_text = aggregate_metrics(all_metrics_text, configs)

    if method_name != "bi_cross_attn":
        experiment_data.update({
            "metrics": agg_metrics,
            "method_class": method_name
        })
        save_experiment(configs, method_name, experiment_data)
        update_result_csv(method_name, agg_metrics, configs, csv_path=configs.csv_path)
        
        print("===========================results===========================")
        print(agg_metrics)
        
    else:
        experiment_data.update({
            "img_metrics": agg_metrics_img,
            "method_class": method_name
        })
        experiment_data.update({
            "text_metrics": agg_metrics_text,
            "method_class": method_name
        })
        save_experiment(configs, method_name, experiment_data)
        update_result_csv(method_name + "_img", agg_metrics_img, configs, csv_path=configs.csv_path)
        update_result_csv(method_name + "_text", agg_metrics_text, configs, csv_path=configs.csv_path)
        
        print("===========================img logits results===========================")
        print(agg_metrics_img)

        print("===========================text logits results===========================")
        print(agg_metrics_text)

    wandb.finish()

if __name__ == "__main__":
    configs = Config.fromfile('./tests/config_surgvlp_few_shot.py')['config']
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', default="zero_shot_attn_pooling")
    
    parser.add_argument('--attention_pooling', 
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Whether to use attention pooling (true/false)')

    parser.add_argument('--num_shots',
                        type=int,
                        help='Number of shots for few-shot learning')

    parser.add_argument('--epochs',
                        type=int,
                        help='Number of training epochs')
    parser.add_argument('--init_alpha',
                        type=float)
    parser.add_argument('--csv_path',
                        type=str)
    
    args = parser.parse_args()

    method_configs = configs[args.method]

    if args.attention_pooling is not None:
        method_configs["attention_pooling"] = args.attention_pooling
    
    if args.num_shots:
        method_configs["num_shots"] = args.num_shots
    
    if args.epochs:
        method_configs["epochs"] = args.epochs

    if args.init_alpha:
        method_configs["init_alpha"] = args.init_alpha
    
    if args.csv_path:
        method_configs["csv_path"] = args.csv_path

    main(method_configs, args.method)