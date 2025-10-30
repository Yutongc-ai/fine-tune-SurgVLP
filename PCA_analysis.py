import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datasets.cholec80 import templates, simple_templates, hard_templates, tools

import torch
import surgvlp
from mmengine.config import Config
import random
from datasets.cholec80 import Cholec80, NegationCholec80
from datasets.utils import *
from methods.zero_shot import ZeroShot
from methods.linear_probe import LP
from methods.linear_probe_plus import LPPlus2
from methods.ablation.negation_maf_savet import NegationSaveT
from methods.ablation.negation_MAF import NegationMAF
from methods.ablation.negation_MAF_multemplate import NegationMul
from methods.ablation.negation_MAF_onlypos import NegationOP
from methods.clip_adapter import ClipAdapter
from methods.tip_adapter import TIPAdapter
from methods.coop import COOP
from methods.cocoop import COCOOP
from methods.dual_coop import DualCOOP
from methods.utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from datasets.utils import read_image

def load_models(surg_config, baseline_config, surgvlp_model, preprocess):

    surg_nat_model = NegationMul(surg_config, surgvlp_model, preprocess, surgvlp.tokenize) # 加载Surg-NAT模型
    surg_nat_model.eval()
    load_path = 'checkpoints/PCA.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(load_path, map_location=device)

    surg_nat_model.vision_adapter.load_state_dict(checkpoint['vision_adapter'])
    surg_nat_model.text_adapter.load_state_dict(checkpoint['text_adapter'])

    baseline_model = NegationOP(baseline_config, surgvlp_model, preprocess, surgvlp.tokenize) # 加载Baseline模型
    baseline_model.eval()
    
    load_path = 'checkpoints/negation.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(load_path, map_location=device)

    baseline_model.vision_adapter.load_state_dict(checkpoint['vision_adapter'])
    baseline_model.text_adapter.load_state_dict(checkpoint['text_adapter'])


    return surg_nat_model, baseline_model

configs = Config.fromfile('./tests/config_surgvlp_few_shot.py')['config']

surg_configs = configs["negation_mul"]
baseline_configs = configs["negation_op"]

model_path = f'/home/yongxuan/SurgVLP/checkpoints/PeskaVLP.pth'
surgvlp_model, preprocess = surgvlp.load(surg_configs.model_config, device='cuda', pretrain=model_path)

surg_nat_model, baseline_model = load_models(surg_configs, baseline_configs, surgvlp_model, preprocess)
surg_nat_model = surg_nat_model.to(device)
baseline_model = baseline_model.to(device)


templates = list(templates.values())

simple_templates = [
    [template.format(tool = tool_name) for tool_name in tools]
    for template in simple_templates
]
simple_templates = [
    item for sublist in simple_templates for item in sublist
]

hard_templates = [
    [template.format(tool = tool_name) for tool_name in tools]
    for template in hard_templates
]
hard_templates = [
    item for sublist in hard_templates for item in sublist
]

tokenizer = surgvlp.tokenize
pos_templates = tokenizer(templates, device = device)
pos_input_ids = pos_templates['input_ids']
pos_token_type_ids = pos_templates['token_type_ids']
pos_attention_masks = pos_templates['attention_mask']

simple_templates = tokenizer(simple_templates, device = device)
simple_input_ids = simple_templates['input_ids']
simple_token_type_ids = simple_templates['token_type_ids']
simple_attention_masks = simple_templates['attention_mask']

hard_templates = tokenizer(hard_templates, device = device)
hard_input_ids = hard_templates['input_ids']
hard_token_type_ids = hard_templates['token_type_ids']
hard_attention_masks = hard_templates['attention_mask']

_, text_pos, _ = surgvlp_model.extract_feat_text(pos_input_ids, pos_attention_masks, pos_token_type_ids)

_, text_simple, _ = surgvlp_model.extract_feat_text(simple_input_ids, simple_attention_masks, simple_token_type_ids)

_, text_hard, _ = surgvlp_model.extract_feat_text(hard_input_ids, hard_attention_masks, hard_token_type_ids)

features_before_pos_np = text_pos.detach().cpu().numpy()
features_before_simple_np = text_simple.detach().cpu().numpy()
features_before_hard_np = text_hard.detach().cpu().numpy()
simple_features_np = np.vstack((features_before_pos_np, features_before_simple_np))

labels = np.array([0] * features_before_pos_np.shape[0] + [1] * features_before_simple_np.shape[0])

print(f"\n合并后的特征数据形状: {simple_features_np.shape}")
print(f"合并后的标签形状: {labels.shape}")

scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(simple_features_np)

pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(all_features_scaled)

print(f"\nPCA结果形状: {pca_results.shape}")
print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA总解释方差: {np.sum(pca.explained_variance_ratio_):.2f}")

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=pca_results[:, 0],
    y=pca_results[:, 1],
    hue=labels, # 根据labels着色
    palette=['blue', 'red'], # 为0和1分别指定颜色
    legend='full',
    alpha=0.8,
    s=300 # 调整点的大小，因为样本数较少
)


handles, _labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Positive', 'Explicit Negative'], title='Prompt Type')

plt.title('PCA Embeddings of Positive and Explicit Negative Prompts')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

plt.savefig('before/pca_pos_simple_features.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


hard_features_np = np.vstack((features_before_pos_np, features_before_hard_np))
# 创建标签：0代表positive，1代表negative
labels = np.array([0] * features_before_pos_np.shape[0] + [1] * features_before_hard_np.shape[0])


scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(hard_features_np)

pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(all_features_scaled)

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=pca_results[:, 0],
    y=pca_results[:, 1],
    hue=labels, # 根据labels着色
    palette=['blue', 'yellow'], # 为0和1分别指定颜色
    legend='full',
    alpha=0.8,
    s=300 
)


handles, _labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Positive', 'Implicit Negative'], title='Prompt Type')

plt.title('PCA Embeddings of Positive and Implicit Negative Prompts')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

plt.savefig('before/pca_pos_hard_features.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()



hard_features_np = np.vstack((features_before_pos_np, features_before_simple_np, features_before_hard_np))
labels = np.array([0] * features_before_pos_np.shape[0] + [1] * features_before_simple_np.shape[0] + [2] * features_before_hard_np.shape[0])

print(f"\n合并后的特征数据形状: {hard_features_np.shape}")
print(f"合并后的标签形状: {labels.shape}")

scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(hard_features_np)

pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(all_features_scaled)

print(f"\nPCA结果形状: {pca_results.shape}")
print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA总解释方差: {np.sum(pca.explained_variance_ratio_):.2f}")

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=pca_results[:, 0],
    y=pca_results[:, 1],
    hue=labels, # 根据labels着色
    palette=['blue', 'red', 'yellow'], # 为0和1分别指定颜色
    legend='full',
    alpha=0.8,
    s=300
)


handles, _labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Positive', 'Expilcit Negative', 'Implicit Negative'], title='Prompt Type')

plt.title('PCA Embeddings of Positive and Negative Prompts')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

plt.savefig('before/pca_pos_all_features.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()