import pandas as pd
import io

# --- 配置区 ---
# 将你的 CSV 文件名放在这里
CSV_FILE_PATH = 'results_sum_peska.csv' 
# 你想要比较的方法名称列表，按你希望在表格中出现的顺序列出
METHOD_ORDER = [
    'linear_probe',
    'linear_probe++',
    'clip_adapter',
    'tip_adapter',
    'coop',
    'dual_coop'
    'bi_cross_attn_img',
    'bi_cross_attn_text',
    'negation_nce',
]
FREEZE_ORDER = [
    True,
    False,
]
# 你想要比较的骨干网络列表，按你希望在表格中出现的顺序列出
BACKBONE_ORDER = [
    'PeskaVLP',
    # 'SurgVLP',
    # 'PeskaVLP_negation_pretrain',
]
# 你想要比较的 shot 数量，按你希望在表格中出现的顺序列出
SHOT_ORDER = [1, 16, 64, 128]

# --- 模拟数据 ---
# 如果你还没有 `results.csv` 文件，可以使用下面的模拟数据来测试脚本
# 只需取消注释下一行即可
# CSV_FILE_PATH = None 

# 用于测试的模拟 CSV 数据，现在包含了多种 backbone
DUMMY_CSV_DATA = """
timestamp,method,backbone,mAP_avg,f1_avg,lr,num_shots
2023-10-27,Linear-probe,ViT-B/16,21.14,20.5,0.01,1
2023-10-27,Linear-probe,ViT-B/16,31.67,30.9,0.01,2
2023-10-27,Linear-probe,ViT-B/16,51.59,50.8,0.01,16
2023-10-27,Linear-probe,ResNet-50,19.50,18.9,0.01,1
2023-10-27,Linear-probe,ResNet-50,28.90,28.1,0.01,2
2023-10-27,CoOp,ViT-B/16,53.12,52.5,0.002,1
2023-10-27,CoOp,ViT-B/16,56.47,55.9,0.002,4
2023-10-27,CoOp,ViT-B/16,60.46,60.1,0.002,16
2023-10-27,CLIP-Adapter,ViT-B/16,58.17,57.6,0.001,1
2023-10-27,CLIP-Adapter,ViT-B/16,61.33,60.8,0.001,16
2023-10-27,MyMethod,ViT-B/16,59.50,58.9,0.01,1
2023-10-27,MyMethod,ViT-B/16,59.45,58.8,0.02,1
2023-10-27,MyMethod,ViT-B/16,62.70,62.1,0.01,16
2023-10-27,MyMethod,ResNet-50,57.80,57.1,0.01,1
2023-10-27,MyMethod,ResNet-50,57.95,57.3,0.02,1
2023-10-27,MyMethod,ResNet-50,61.50,60.9,0.01,16
"""

def generate_comparison_table(df, metric_col, method_order, freeze_order, backbone_order, shot_order):
    """
    根据指定的指标，从原始数据中生成一个格式化的比较表，同时考虑 backbone。

    Args:
        df (pd.DataFrame): 包含实验结果的DataFrame。
        metric_col (str): 要比较的指标列名 (例如 'mAP_avg')。
        method_order (list): 控制方法主顺序的列表。
        backbone_order (list): 控制 backbone 子顺序的列表。
        shot_order (list): 控制shot列顺序的列表。

    Returns:
        pd.DataFrame: 格式化后的结果表。
    """
    print(f"--- Generating table for metric: {metric_col} ---")
    
    # 1. 定义用于分组的唯一标识符
    grouping_keys = ['method', 'backbone', 'unfreeze_vision', 'num_shots']
    
    # 2. 找到每个 (method, backbone, num_shots) 组合的最佳结果
    best_results = df.sort_values(metric_col, ascending=False).drop_duplicates(grouping_keys)
    
    # 3. 使用 pivot_table 创建表格结构，现在 index 是一个 MultiIndex
    
    def format_with_epoch(row):
        metric_val = row[metric_col]
        epoch_val = row['epochs']
        # 检查 epoch 值是否存在且不为空
        if pd.notna(epoch_val):
            return f"{metric_val:.2f} ({int(epoch_val)})"
        else:
            return f"{metric_val:.2f}"

    best_results['display_val'] = best_results.apply(format_with_epoch, axis=1)

    pivot_df = best_results.pivot_table(
        index=['method', 'unfreeze_vision', 'backbone'], 
        columns='num_shots', 
        values='display_val',
        aggfunc = 'first',
    )
    
    # print(pivot_df)
    # 如果 pivot_df 为空，则提前返回
    if pivot_df.empty:
        print("Warning: No data available to create a pivot table.")
        return pd.DataFrame()

    # 4. 格式化和排序
    
    # 筛选和排序 'shots' (列)
    shots_in_data = [shot for shot in shot_order if shot in pivot_df.columns]
    if not shots_in_data:
        print(f"Warning: None of the specified shots {shot_order} were found in the data.")
        return pd.DataFrame()
    pivot_df = pivot_df[shots_in_data]
    
    # 筛选和排序 'method' 和 'backbone' (行)
    # 创建我们期望的 MultiIndex 顺序
    available_methods = pivot_df.index.get_level_values('method').unique()
    available_backbones = pivot_df.index.get_level_values('backbone').unique()
    available_freeze = pivot_df.index.get_level_values('unfreeze_vision').unique()
    # print(available_freeze)
    
    final_method_order = [m for m in method_order if m in available_methods]
    final_backbone_order = [b for b in backbone_order if b in available_backbones]
    # print(freeze_order)
    final_freeze_order = [f for f in freeze_order if f in available_freeze]
    # print(final_method_order)
    # print(final_backbone_order)
    # print(final_freeze_order)

    # 根据用户定义的顺序重新索引
    pivot_df = pivot_df.reindex(
        pd.MultiIndex.from_product([final_method_order, final_freeze_order, final_backbone_order], names=['method', 'unfreeze_vision', 'backbone'])
    ).dropna(how='all') # 删除完全是NA的行

    # 格式化数值
    # formatted_df = pivot_df.applymap(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    final_df = pivot_df.fillna('-')
    
    # 重命名列名，使其更清晰
    final_df.columns.name = 'Few-shot Setup'
    final_df.index.names = ['Method', 'Unfreeze', 'Backbone']
    
    return final_df

def main():
    """主函数：读取数据并生成报告"""
    try:
        if CSV_FILE_PATH:
            df = pd.read_csv(CSV_FILE_PATH)
            print(f"Successfully loaded data from '{CSV_FILE_PATH}'.")
        else:
            print("Loading dummy data for demonstration.")
            df = pd.read_csv(io.StringIO(DUMMY_CSV_DATA))
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        return

    # 确保关键列存在
    required_cols = ['method', 'backbone', 'unfreeze_vision', 'num_shots', 'mAP_avg', 'f1_avg']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV file must contain the following columns: {required_cols}")
        print(f"Missing columns: {[col for col in required_cols if col not in df.columns]}")
        return

    # 生成 mAP 表格
    map_table = generate_comparison_table(df, 'mAP_avg', METHOD_ORDER, FREEZE_ORDER, BACKBONE_ORDER, SHOT_ORDER)
    if not map_table.empty:
        print("\n" + "="*55)
        print("                 mAP (%) Comparison Table")
        print("="*55)
        print(map_table)
        print("="*55)

    # 生成 F1 表格
    f1_table = generate_comparison_table(df, 'f1_avg', METHOD_ORDER, FREEZE_ORDER, BACKBONE_ORDER, SHOT_ORDER)
    if not f1_table.empty:
        print("\n" + "="*55)
        print("               F1-Score (%) Comparison Table")
        print("="*55)
        print(f1_table)
        print("="*55)
    
    map_table.to_csv('mAP_comparison_table_peska.csv', index=True, encoding='utf-8-sig')
    print("mAP table saved to mAP_comparison_table.csv")

    f1_table.to_csv('F1_comparison_table_peska.csv', index=True, encoding='utf-8-sig')
    print("F1 table saved to F1_comparison_table.csv")

if __name__ == '__main__':
    main()

