import itertools
import subprocess
import time
import os
import sys

# 定义要遍历的参数组合
# attention_pooling_options = [False]
num_shots_options = [1, 16, 64, 128] # 256, 512]
epochs_options = [50]

# 生成所有参数组合
all_combinations = list(itertools.product(
    # attention_pooling_options,
    num_shots_options,
    epochs_options
))

# 检查GPU使用情况（可选）
def check_gpu_available():
    try:
        # 使用nvidia-smi检查GPU内存使用情况
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        memory_used = [int(x) for x in result.stdout.strip().split('\n')]
        
        # 如果有GPU且内存使用超过阈值，等待
        if any(used > 100 for used in memory_used):  # 100MB阈值
            print("GPU still in use, waiting...")
            return False
        return True
    except Exception:
        # 如果没有nvidia-smi，跳过检查
        return True

# 遍历每个组合并运行实验
for idx, (num_shot, epochs) in enumerate(all_combinations, 1):
    print(f"\n{'='*50}")
    print(f"Running experiment {idx}/{len(all_combinations)}")
    print(f"Config: num_shots={num_shot}, epochs={epochs}")
    print(f"{'='*50}")
    
    # 构建命令行参数
    command = [
        "python", "few_shot.py",
        "--method", "bi_cross_attn",
        "--model_type", "PeskaVLP",
        "--num_shots", str(num_shot),
        "--epochs", str(epochs),
        "--csv_path", "results_cross_attn.csv",
    ]
    
    # 打印命令以便调试
    print("Executing:", " ".join(command))
    
    # 运行命令
    try:
        # 使用Popen启动进程
        process = subprocess.Popen(command)
        
        # 等待进程完成
        process.wait()
        
        # 可选：添加额外等待时间确保资源释放
        if process.returncode == 0:
            print(f"Experiment {idx} completed successfully!")
            
            # 等待1秒确保资源完全释放
            print("Waiting for resources to be released...")
            time.sleep(5)
            
            # 可选：检查GPU资源是否释放
            # while not check_gpu_available():
            #     print("GPU still in use, waiting 5 seconds...")
            #     time.sleep(5)
        else:
            print(f"Experiment {idx} failed with return code: {process.returncode}")
        
    except Exception as e:
        print(f"Experiment {idx} failed with error: {e}")
        # 可以选择继续运行
        # continue

print("\nAll experiments completed!")