# --- 1. 导入库 (真实代码) ---
import torch
import surgvlp
from mmengine.config import Config
import random
from datasets.cholec80 import Cholec80, NegationCholec80
from datasets.utils import *
from methods.zero_shot import ZeroShot
from methods.linear_probe import LP
from methods.linear_probe_plus import LPPlus2
from methods_old.cross_attn import CrossAttn
from methods_old.cross_attn_noproj import CrossAttnNoProj
from methods_old.negation import Negation
from methods.ablation.negation_maf_savet import NegationSaveT
from methods_old.normal_finetune import NormalFinetune
from methods_old.weighted_negation_nce import WeightedNegationNCE
from methods_old.negation_nce_all import NegationNCEAll
from methods_old.negation_nce_dir import NegationNCEDir
from methods.ablation.negation_MAF import NegationMAF
from methods_old.negation_nce_MAF_attn import NegationMAFAttn
from methods_old.negation_nce_MAF_reg import NegationMAFReg
from methods.ablation.negation_MAF_multemplate import NegationMul
from methods.ablation.negation_MAF_onlypos import NegationOP
from methods.negation_multemplate import NegationMulLastLayer
from methods_old.ours import Ours
from methods_old.fine_tune import FineTune
from methods_old.part_negation_nce import PartNegationNCE
from methods_old.simple import Simple
from methods_old.aggre_negation import AggreNegation
from methods.clip_adapter import ClipAdapter
from methods.tip_adapter import TIPAdapter
from methods.coop import COOP
from methods.cocoop import COCOOP
from methods.dual_coop import DualCOOP
import argparse
import datetime
from methods.utils import *
import wandb
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from datasets.utils import read_image

def set_seeds(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置Python哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    
    # 这两个设置对于复现性很关键
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. 伪代码：模型和数据加载 ---

def load_models(surg_config, baseline_config, surgvlp_model, preprocess):
    # 加载你的Surg-NAT模型
    # 注意：Surg-NAT在文本编码器上使用了MAF [cite: 61, 73]
    # 并且在图像编码器上也可能有适配器 [cite: 50]

    surg_nat_model = NegationMul(surg_config, surgvlp_model, preprocess, surgvlp.tokenize) # 加载Surg-NAT模型
    surg_nat_model.eval()
    load_path = 'checkpoints/act_map_64.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(load_path, map_location=device)

    # 5. 将加载的 state_dict 注入到新模型的子模块中
    surg_nat_model.vision_adapter.load_state_dict(checkpoint['vision_adapter'])
    surg_nat_model.text_adapter.load_state_dict(checkpoint['text_adapter'])

    # 加载一个Baseline模型（例如，CLIP-Adapter ）
    baseline_model = NegationOP(baseline_config, surgvlp_model, preprocess, surgvlp.tokenize) # 加载Baseline模型
    baseline_model.eval()
    
    load_path = 'checkpoints/negationop.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(load_path, map_location=device)

    baseline_model.vision_adapter.load_state_dict(checkpoint['vision_adapter'])
    baseline_model.text_adapter.load_state_dict(checkpoint['text_adapter'])


    return surg_nat_model, baseline_model

def get_text_features(text_encoder, prompt):
    """
    (伪代码)
    提取文本的全局feature (例如[CLS] token的输出)。
    输出维度 [embed_dim]，例如 [768]。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    # Surg-NAT的文本编码器应用了MAF [cite: 61, 73]
    features = text_encoder(**inputs).pooler_output 
    return features.squeeze(0) # [768]

def compute_cosine_similarity(patches, text_feat):
    """ (伪代码) 计算余弦相似度 """
    # patches: [N, D] (N=49, D=768)
    # text_feat: [D] (D=768)
    
    patches_norm = torch.nn.functional.normalize(patches, p=2, dim=1)
    text_feat_norm = torch.nn.functional.normalize(text_feat, p=2, dim=0)
    
    # [N, D] @ [D] -> [N]
    similarity_map = torch.matmul(patches_norm, text_feat_norm)
    return similarity_map.detach().cpu().numpy() # [49]

# --- 4. 真实代码：绘图函数 ---

def plot_heatmap_on_image(ax, original_image, activation_map_1d, patch_grid_size, title, normalize=True):
    """
    将1D的激活图 (例如 1x49) 调整大小并叠加到原始图像上。
    """
    # 0. 确保图像是 0-1 范围的浮点数
    if original_image.max() > 1.0:
        img = original_image.astype(np.float32) / 255.0
    else:
        img = original_image.astype(np.float32)
        
    img_h, img_w = img.shape[:2]

    # 1. 归一化激活图
    if normalize:
        map_norm = (activation_map_1d - activation_map_1d.min()) / (activation_map_1d.max() - activation_map_1d.min() + 1e-8)
    else:
        # 对于Differential Map，我们可能希望在0附近有不同的颜色
        # 但为了简单起见，这里也使用标准归一化
        map_norm = (activation_map_1d - activation_map_1d.min()) / (activation_map_1d.max() - activation_map_1d.min() + 1e-8)
        
    # 2. Reshape到 7x7 (或 patch_grid_size x patch_grid_size)
    map_2d = map_norm.reshape(patch_grid_size, patch_grid_size)

    # 3. Upsample 到原始图像尺寸
    # 使用CUBIC插值使其更平滑
    heatmap = cv2.resize(map_2d, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    # 4. 应用 colormap (例如 'jet' 或 'viridis')
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3] # 取 RGB通道, 忽略 alpha

    # 5. 混合图像和heatmap
    alpha = 0.5
    overlay = (img * (1 - alpha)) + (heatmap_colored * alpha)
    
    # 6. 绘图
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(title, fontsize=16)

def plot_all_heatmaps(original_image, map_pos_base, map_neg_base, map_diff_base,
                      map_pos_surg, map_neg_surg, map_diff_surg,
                      patch_grid_size, save_path, title_prefix=""):
    """
    (真实代码)
    将所有6张图绘制并保存到一个文件中。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title_prefix, fontsize=16, y=1.02)
    
    # --- 第一行: Baseline ---
    
    # 1. 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    # 2. Baseline (Positive)
    # plot_heatmap_on_image(axes[0, 1], original_image, map_pos_base, patch_grid_size, "Baseline (Positive)")
    
    # 3. Baseline (Negative)
    plot_heatmap_on_image(axes[1], original_image, map_pos_base, patch_grid_size, "Baseline")

    # --- 第二行: Surg-NAT (Ours) ---
    
    # 4. Surg-NAT (Positive)
    # plot_heatmap_on_image(axes[1, 0], original_image, map_pos_surg, patch_grid_size, "Surg-NAT (Positive)")
    
    # 5. Surg-NAT (Negative)
    # plot_heatmap_on_image(axes[1, 1], original_image, map_neg_surg, patch_grid_size, "Surg-NAT (Negative)")
    
    # 6. Surg-NAT (Differential)
    plot_heatmap_on_image(axes[2], original_image, map_diff_surg, patch_grid_size, "Surg-NAT+")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # 关闭图像以释放内存，这对于循环处理很多图片至关重要

def main(surg_config, baseline_config):
    set_seeds(42)

    test_dataset = Cholec80(config=surg_configs.dataset_config)
    classnames = test_dataset.classnames
    templates = test_dataset.templates
    print("dataset class names:", classnames)
    print("template:", templates)

    model_path = f'/home/yongxuan/SurgVLP/checkpoints/PeskaVLP.pth'
    surgvlp_model, preprocess = surgvlp.load(surg_config.model_config, device='cuda', pretrain=model_path)

    surg_nat_model, baseline_model = load_models(surg_config, baseline_config, surgvlp_model, preprocess)
    surg_nat_model = surg_nat_model.to(device)
    baseline_model = baseline_model.to(device)

    output_dir = "ablation_outputs_scissors/"
    os.makedirs(output_dir, exist_ok=True)

    PATCH_GRID_SIZE = 7 

    for i, item in enumerate(test_dataset.train_x):
        image_path = item.impath
        label = item.labels

        # print(label)
        if 3 not in label:
            continue
        # 假设我们正在为 "grasper" 工具生成map
        tokenizer = surgvlp.tokenize
        prompt_pos = "I use scissors" # [cite: 30]
        prompt_neg = "I do not use scissors" # [cite: 31, 66]
        pos_templates = tokenizer(prompt_pos, device = device)
        pos_input_ids = pos_templates['input_ids']
        pos_token_type_ids = pos_templates['token_type_ids']
        pos_attention_masks = pos_templates['attention_mask']
        neg_templates = tokenizer(prompt_neg, device = device)
        neg_input_ids = neg_templates['input_ids']
        neg_token_type_ids = neg_templates['token_type_ids']
        neg_attention_masks = neg_templates['attention_mask']
        
        pos_feats_outputs = surgvlp_model.backbone_text.model(pos_input_ids, pos_attention_masks, pos_token_type_ids)
        pos_layer_templates_feats = pos_feats_outputs[2][-4:]
        pos_layer_templates_feats = tuple(t.cuda() for t in pos_layer_templates_feats)

        neg_feats_outputs = surgvlp_model.backbone_text.model(neg_input_ids, neg_attention_masks, neg_token_type_ids)
        neg_layer_templates_feats = neg_feats_outputs[2][-4:]
        neg_layer_templates_feats = tuple(t.cuda() for t in neg_layer_templates_feats)

        # 加载和预处理图像
        original_image = Image.open(image_path).convert("RGB")
        test_loader = build_data_loader(data_source=[item], batch_size = 1, is_train = False, tfm = preprocess,
                                    num_classes = 7)
        print(test_loader)
        for (image, _, _, _) in test_loader:
            image = image.to(device)
            _, local_image_features = surgvlp_model.extract_feat_img(image)
            local_image_features = local_image_features.permute(0, 2, 3, 1)
            local_image_features = local_image_features.view(49, -1)
            print(local_image_features.shape)
            local_image_features = surgvlp_model.backbone_img.global_embedder(local_image_features)
            base_ada_image_features = baseline_model.vision_adapter(local_image_features)
            surg_ada_image_features = surg_nat_model.vision_adapter(local_image_features)

            # --- A. Baseline模型处理 ---
            text_pos_base = baseline_model.text_adapter(pos_layer_templates_feats)
            text_neg_base = baseline_model.text_adapter(neg_layer_templates_feats)

            map_pos_base = compute_cosine_similarity(base_ada_image_features, text_pos_base.T)
            map_neg_base = compute_cosine_similarity(base_ada_image_features, text_neg_base.T)

            # --- B. Surg-NAT模型处理 ---
            text_pos_surg = surg_nat_model.text_adapter(pos_layer_templates_feats)
            text_neg_surg = surg_nat_model.text_adapter(neg_layer_templates_feats)

            map_pos_surg = compute_cosine_similarity(surg_ada_image_features, text_pos_surg.T)
            map_neg_surg = compute_cosine_similarity(surg_ada_image_features, text_neg_surg.T)

            # --- C. 计算Surg-NAT的Differential Map  ---
            map_diff_surg = map_pos_surg - map_neg_surg
            map_diff_base = map_pos_base - map_neg_base

            # --- D. 绘图 (调用下面的真实代码) ---
            save_path = f"{output_dir}/image_{i:04d}_ablation.png"
            plot_all_heatmaps(
                np.array(original_image),
                map_pos_base, map_neg_base, map_diff_base,
                map_pos_surg, map_neg_surg, map_diff_surg,
                PATCH_GRID_SIZE,
                save_path,
                title_prefix=f"Scissors"
            )

            if i > 50: # 按需处理足够多的图片
                break

if __name__ == "__main__":
    configs = Config.fromfile('./tests/config_surgvlp_few_shot.py')['config']

    surg_configs = configs["negation_mul"]
    baseline_configs = configs["negation_op"]

    main(surg_configs, baseline_configs)