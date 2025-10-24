"""
测试脚本：验证 LoRA-Null-v2 的零空间属性
运行方式: python test_null_space.py
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapterlib.datautils import get_calib_data
from adapterlib.act_aware_utils import calib_cov_distribution
from adapterlib.decomposition import build_model2
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--calib_loader_size", type=int, default=16, help="使用较小的样本量进行快速测试")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("="*80)
    print("LoRA-Null-v2 零空间验证测试")
    print("="*80)
    print(f"模型: {args.model_id}")
    print(f"秩: {args.r}")
    print(f"校准样本数: {args.calib_loader_size}")
    print("="*80)

    # 加载模型和分词器
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 收集校准数据
    print("\n[2/4] 收集校准数据...")
    calib_loader = get_calib_data(
        args.calib_dataset,
        tokenizer,
        args.model_id,
        args.calib_loader_size,
        seed=args.seed
    )

    # 计算协方差矩阵并缓存输入样本
    print("\n[3/4] 计算协方差矩阵并缓存输入样本...")
    calib_cov_distribution(
        model,
        calib_loader,
        use_cache=False,  # 不使用缓存，强制重新计算
        calib_dataset=args.calib_dataset,
        calib_size=args.calib_loader_size,
        seed=args.seed
    )

    # 执行分解并验证零空间属性
    print("\n[4/4] 执行SVD分解并验证零空间属性...")
    print("注意：以下将显示每一层的零空间验证结果\n")

    # 创建临时参数对象
    class TempArgs:
        def __init__(self):
            self.mode = "build_adapters"
            self.r = args.r
            self.act_aware = False
            self.cov_aware = False
            self.singular_aware = False
            self.singular_aware_2 = True  # 使用 LoRA-Null-v2
            self.first_eigen = False
            self.alpha = 0.5
            self.sigma_fuse = "UV"

    temp_args = TempArgs()
    build_model2(model, temp_args)

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print("\n关键指标说明：")
    print("1. ||BX|| / ||X||: 衡量投影到零空间后的激活幅度")
    print("   - 理论上应该 << 1 (远小于1)")
    print("   - 越接近0说明零空间近似越好")
    print("\n2. ||ABX|| / ||X||: 衡量适配器初始化时的输出变化")
    print("   - 理论上应该 ≈ 0 (因为A初始化为0)")
    print("   - 机器精度下应该 < 1e-6")
    print("\n3. Eigenvalue ratio (min/max): 协方差矩阵的最小/最大特征值比")
    print("   - 越小说明数据在某些方向上的方差越小")
    print("   - 这些低方差方向就是我们选择的'零空间'")
    print("="*80)

if __name__ == "__main__":
    main()
