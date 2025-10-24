"""
训练后零空间验证脚本
用于验证 LoRA-Null-v2 训练后模型的零空间属性

运行方式:
python verify_trained_null_space.py --trained_model_path <训练后模型路径>
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapterlib.datautils import get_calib_data
from tqdm import tqdm
import numpy as np
import os


@torch.no_grad()
def collect_activations_and_verify(model, calib_loader, max_samples=100):
    """
    收集激活值并验证零空间属性

    Args:
        model: 训练后的模型
        calib_loader: 校准数据加载器
        max_samples: 最多收集的样本数
    """
    model.eval()

    # 找到所有 CorDA_adapter 层
    adapter_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'ALinear') and hasattr(module, 'BLinear'):
            adapter_layers[name] = {
                'module': module,
                'cached_inputs': [],
                'input_count': 0
            }

    print(f"\n找到 {len(adapter_layers)} 个 CorDA_adapter 层")

    # 注册 hook 收集输入
    def make_hook(layer_name):
        def hook(module, input, output):
            if adapter_layers[layer_name]['input_count'] < max_samples:
                inp = input[0].detach()
                if inp.dim() == 3:  # (batch, seq_len, dim)
                    inp = inp.reshape(-1, inp.size(-1))  # (batch*seq_len, dim)

                # 只取前几个样本
                num_to_cache = min(max_samples - adapter_layers[layer_name]['input_count'], inp.size(0))
                adapter_layers[layer_name]['cached_inputs'].append(inp[:num_to_cache].cpu())
                adapter_layers[layer_name]['input_count'] += num_to_cache
        return hook

    # 注册所有 hooks
    handles = []
    for name, info in adapter_layers.items():
        handle = info['module'].register_forward_hook(make_hook(name))
        handles.append(handle)

    # 前向传播收集数据
    print("\n收集激活值...")
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

        # 检查是否收集够了
        if all(info['input_count'] >= max_samples for info in adapter_layers.values()):
            break

    # 移除 hooks
    for handle in handles:
        handle.remove()

    # 合并缓存的输入
    for name, info in adapter_layers.items():
        if len(info['cached_inputs']) > 0:
            info['cached_inputs'] = torch.cat(info['cached_inputs'], dim=0)

    return adapter_layers


@torch.no_grad()
def verify_null_space_properties(adapter_layers, verbose=True):
    """
    验证零空间属性

    Args:
        adapter_layers: 包含适配器层和缓存输入的字典
        verbose: 是否输出详细信息
    """
    print("\n" + "="*80)
    print("训练后零空间验证")
    print("="*80)

    all_stats = []

    for name, info in adapter_layers.items():
        module = info['module']
        X = info['cached_inputs']

        if X is None or len(X) == 0:
            continue

        X = X.float()  # (num_samples, in_features)

        # 获取 B 的权重 (BLinear: in_features -> rank)
        B_weight = module.BLinear.weight.data.cpu().float()  # (rank, in_features)

        # 获取 A 的权重 (ALinear: rank -> out_features)
        A_weight = module.ALinear.weight.data.cpu().float()  # (out_features, rank)

        # 计算 BX
        BX = B_weight @ X.t()  # (rank, num_samples)

        # 计算 ABX
        ABX = A_weight @ BX  # (out_features, num_samples)

        # 计算残差项 W_residual @ X
        W_residual = module.weight_residual.data.cpu().float()  # (out_features, in_features)
        ResidualX = W_residual @ X.t()  # (out_features, num_samples)

        # 完整输出
        FullOutput = ABX + ResidualX  # (out_features, num_samples)

        # 计算范数统计
        X_norm = torch.norm(X, dim=1).mean().item()
        BX_norm = torch.norm(BX, dim=0).mean().item()
        ABX_norm = torch.norm(ABX, dim=0).mean().item()
        ResidualX_norm = torch.norm(ResidualX, dim=0).mean().item()
        FullOutput_norm = torch.norm(FullOutput, dim=0).mean().item()

        # 计算权重统计
        B_norm = torch.norm(B_weight).item()
        A_norm = torch.norm(A_weight).item()
        Residual_norm = torch.norm(W_residual).item()

        stats = {
            'layer_name': name,
            'num_samples': X.size(0),
            'X_norm': X_norm,
            'BX_norm': BX_norm,
            'ABX_norm': ABX_norm,
            'ResidualX_norm': ResidualX_norm,
            'FullOutput_norm': FullOutput_norm,
            'BX_ratio': BX_norm / X_norm if X_norm > 0 else 0,
            'ABX_ratio': ABX_norm / X_norm if X_norm > 0 else 0,
            'adapter_contribution': ABX_norm / FullOutput_norm if FullOutput_norm > 0 else 0,
            'B_weight_norm': B_norm,
            'A_weight_norm': A_norm,
            'Residual_weight_norm': Residual_norm,
        }

        all_stats.append(stats)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Layer: {name}")
            print(f"{'='*80}")
            print(f"输入样本数: {stats['num_samples']}")
            print(f"输入维度: {X.shape[1]}, 秩: {B_weight.shape[0]}, 输出维度: {A_weight.shape[0]}")
            print(f"-" * 80)
            print(f"激活值范数:")
            print(f"  ||X|| (avg):           {stats['X_norm']:.6f}")
            print(f"  ||BX|| (avg):          {stats['BX_norm']:.6f}")
            print(f"  ||ABX|| (avg):         {stats['ABX_norm']:.6f}")
            print(f"  ||W_residual @ X||:    {stats['ResidualX_norm']:.6f}")
            print(f"  ||完整输出||:           {stats['FullOutput_norm']:.6f}")
            print(f"-" * 80)
            print(f"零空间指标:")
            print(f"  ||BX|| / ||X||:        {stats['BX_ratio']:.6e} {'✓ 仍然很小' if stats['BX_ratio'] < 0.1 else '⚠ 变大了'}")
            print(f"  ||ABX|| / ||X||:       {stats['ABX_ratio']:.6e} {'✓ 适配器更新有效' if stats['ABX_ratio'] > 1e-6 else '⚠ 适配器几乎没变化'}")
            print(f"-" * 80)
            print(f"适配器贡献:")
            print(f"  ||ABX|| / ||输出||:    {stats['adapter_contribution']:.6e} (适配器在总输出中的相对贡献)")
            print(f"-" * 80)
            print(f"权重范数:")
            print(f"  ||B||_F:               {stats['B_weight_norm']:.6f}")
            print(f"  ||A||_F:               {stats['A_weight_norm']:.6f}")
            print(f"  ||W_residual||_F:      {stats['Residual_weight_norm']:.6f}")
            print(f"{'='*80}")

    # 汇总统计
    if len(all_stats) > 0:
        print(f"\n{'='*80}")
        print("汇总统计（所有层平均）")
        print(f"{'='*80}")
        avg_BX_ratio = np.mean([s['BX_ratio'] for s in all_stats])
        avg_ABX_ratio = np.mean([s['ABX_ratio'] for s in all_stats])
        avg_adapter_contrib = np.mean([s['adapter_contribution'] for s in all_stats])

        print(f"平均 ||BX|| / ||X||:        {avg_BX_ratio:.6e}")
        print(f"平均 ||ABX|| / ||X||:       {avg_ABX_ratio:.6e}")
        print(f"平均适配器贡献比:            {avg_adapter_contrib:.6e}")
        print(f"{'='*80}")

        # 解释
        print("\n📊 结果解读:")
        print("-" * 80)
        if avg_BX_ratio < 0.1:
            print("✅ ||BX|| / ||X|| < 0.1: 零空间属性保持良好")
            print("   输入投影到 B 空间后幅度仍然很小，说明 B 仍在近似零空间中")
        else:
            print("⚠️  ||BX|| / ||X|| >= 0.1: 零空间属性有所退化")
            print("   这可能是因为训练中 BLinear 权重有更新（虽然理论上应该冻结）")

        print()
        if avg_ABX_ratio > 1e-4:
            print(f"✅ ||ABX|| / ||X|| = {avg_ABX_ratio:.2e}: 适配器学到了有效的更新")
            print("   ALinear 权重已经从初始的0更新到有意义的值")
        elif avg_ABX_ratio > 1e-6:
            print(f"⚠️  ||ABX|| / ||X|| = {avg_ABX_ratio:.2e}: 适配器更新较小")
            print("   可能需要更多训练或更大的学习率")
        else:
            print(f"❌ ||ABX|| / ||X|| = {avg_ABX_ratio:.2e}: 适配器几乎没有学到东西")
            print("   检查训练是否正常进行，ALinear 是否被冻结了")

        print()
        if avg_adapter_contrib > 0.01:
            print(f"✅ 适配器贡献 {avg_adapter_contrib:.2%}: 对模型输出有明显影响")
        else:
            print(f"⚠️  适配器贡献 {avg_adapter_contrib:.2%}: 对模型输出影响较小")

        print("-" * 80)

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="验证训练后 LoRA-Null-v2 模型的零空间属性")
    parser.add_argument("--trained_model_path", type=str, required=True,
                        help="训练后模型路径（通常是 output_dir/ft）")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "ptb", "triviaqa", "nqopen"],
                        help="用于收集激活的数据集")
    parser.add_argument("--calib_loader_size", type=int, default=16,
                        help="校准样本数量")
    parser.add_argument("--max_activation_samples", type=int, default=100,
                        help="每层最多收集的激活样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_stats", type=str, default=None,
                        help="保存统计结果的文件路径（.json）")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("="*80)
    print("LoRA-Null-v2 训练后零空间验证")
    print("="*80)
    print(f"训练后模型路径: {args.trained_model_path}")
    print(f"校准数据集: {args.calib_dataset}")
    print(f"校准样本数: {args.calib_loader_size}")
    print("="*80)

    # 检查路径
    if not os.path.exists(args.trained_model_path):
        raise ValueError(f"模型路径不存在: {args.trained_model_path}")

    # 加载训练后的模型
    print("\n[1/3] 加载训练后的模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.trained_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.trained_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"模型类型: {model.__class__.__name__}")

    # 收集校准数据
    print("\n[2/3] 收集校准数据...")
    calib_loader = get_calib_data(
        args.calib_dataset,
        tokenizer,
        args.trained_model_path,
        args.calib_loader_size,
        seed=args.seed
    )

    # 收集激活值
    print("\n[3/3] 收集激活值并验证零空间属性...")
    adapter_layers = collect_activations_and_verify(
        model,
        calib_loader,
        max_samples=args.max_activation_samples
    )

    # 验证零空间属性
    stats = verify_null_space_properties(adapter_layers, verbose=True)

    # 保存统计结果
    if args.save_stats and len(stats) > 0:
        import json
        output = {
            'model_path': args.trained_model_path,
            'calib_dataset': args.calib_dataset,
            'num_layers': len(stats),
            'stats': stats
        }
        with open(args.save_stats, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n统计结果已保存到: {args.save_stats}")

    print("\n✅ 验证完成!")


if __name__ == "__main__":
    main()
