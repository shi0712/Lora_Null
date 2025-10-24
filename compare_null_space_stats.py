"""
对比初始化和训练后的零空间统计

用法:
python compare_null_space_stats.py \
    --trained_model_path <训练后模型路径> \
    --init_model_path <初始化模型路径>
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapterlib.datautils import get_calib_data
import numpy as np
from tabulate import tabulate


@torch.no_grad()
def get_quick_stats(model, calib_loader, max_samples=50):
    """快速获取零空间统计"""
    model.eval()

    adapter_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'ALinear') and hasattr(module, 'BLinear'):
            adapter_layers[name] = {
                'module': module,
                'cached_inputs': []
            }

    # 收集激活
    def make_hook(layer_name):
        def hook(module, input, output):
            if len(adapter_layers[layer_name]['cached_inputs']) < max_samples:
                inp = input[0].detach()
                if inp.dim() == 3:
                    inp = inp.reshape(-1, inp.size(-1))
                num_to_cache = min(max_samples - len(adapter_layers[layer_name]['cached_inputs']), inp.size(0))
                adapter_layers[layer_name]['cached_inputs'].append(inp[:num_to_cache].cpu())
        return hook

    handles = []
    for name, info in adapter_layers.items():
        handle = info['module'].register_forward_hook(make_hook(name))
        handles.append(handle)

    for batch in calib_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)
        if all(len(info['cached_inputs']) >= max_samples for info in adapter_layers.values()):
            break

    for handle in handles:
        handle.remove()

    # 合并并计算统计
    stats = []
    for name, info in adapter_layers.items():
        if len(info['cached_inputs']) == 0:
            continue

        X = torch.cat(info['cached_inputs'], dim=0).float()
        module = info['module']

        B_weight = module.BLinear.weight.data.cpu().float()
        A_weight = module.ALinear.weight.data.cpu().float()

        BX = B_weight @ X.t()
        ABX = A_weight @ BX

        X_norm = torch.norm(X, dim=1).mean().item()
        BX_norm = torch.norm(BX, dim=0).mean().item()
        ABX_norm = torch.norm(ABX, dim=0).mean().item()

        stats.append({
            'layer': name,
            'BX_ratio': BX_norm / X_norm if X_norm > 0 else 0,
            'ABX_ratio': ABX_norm / X_norm if X_norm > 0 else 0,
            'A_norm': torch.norm(A_weight).item(),
            'B_norm': torch.norm(B_weight).item()
        })

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--calib_loader_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    print("="*100)
    print("初始化 vs 训练后零空间对比")
    print("="*100)

    # 加载初始化模型
    print("\n[1/4] 加载初始化模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)
    init_model = AutoModelForCausalLM.from_pretrained(
        args.init_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    calib_loader = get_calib_data(
        args.calib_dataset, tokenizer, args.init_model_path,
        args.calib_loader_size, seed=42
    )

    print("[2/4] 计算初始化模型统计...")
    init_stats = get_quick_stats(init_model, calib_loader, args.max_samples)
    del init_model
    torch.cuda.empty_cache()

    # 加载训练后模型
    print("[3/4] 加载训练后模型...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.trained_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print("[4/4] 计算训练后模型统计...")
    trained_stats = get_quick_stats(trained_model, calib_loader, args.max_samples)

    # 对比
    print("\n" + "="*100)
    print("层级对比")
    print("="*100)

    table_data = []
    for init, trained in zip(init_stats, trained_stats):
        layer_name = init['layer'].split('.')[-1]  # 简化层名
        table_data.append([
            layer_name,
            f"{init['BX_ratio']:.2e}",
            f"{trained['BX_ratio']:.2e}",
            f"{trained['BX_ratio'] / init['BX_ratio']:.2f}x" if init['BX_ratio'] > 0 else "N/A",
            f"{init['ABX_ratio']:.2e}",
            f"{trained['ABX_ratio']:.2e}",
            f"{trained['ABX_ratio'] / init['ABX_ratio']:.0f}x" if init['ABX_ratio'] > 1e-10 else "∞",
            f"{init['A_norm']:.2f}",
            f"{trained['A_norm']:.2f}",
        ])

    headers = [
        "Layer",
        "||BX||/||X|| (初始)",
        "||BX||/||X|| (训练后)",
        "变化倍数",
        "||ABX||/||X|| (初始)",
        "||ABX||/||X|| (训练后)",
        "变化倍数",
        "||A|| (初始)",
        "||A|| (训练后)",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 汇总统计
    print("\n" + "="*100)
    print("汇总统计")
    print("="*100)

    init_avg_BX = np.mean([s['BX_ratio'] for s in init_stats])
    trained_avg_BX = np.mean([s['BX_ratio'] for s in trained_stats])

    init_avg_ABX = np.mean([s['ABX_ratio'] for s in init_stats])
    trained_avg_ABX = np.mean([s['ABX_ratio'] for s in trained_stats])

    init_avg_A = np.mean([s['A_norm'] for s in init_stats])
    trained_avg_A = np.mean([s['A_norm'] for s in trained_stats])

    summary_data = [
        ["||BX|| / ||X|| (平均)", f"{init_avg_BX:.2e}", f"{trained_avg_BX:.2e}",
         f"{trained_avg_BX / init_avg_BX:.2f}x" if init_avg_BX > 0 else "N/A"],
        ["||ABX|| / ||X|| (平均)", f"{init_avg_ABX:.2e}", f"{trained_avg_ABX:.2e}",
         "∞" if init_avg_ABX < 1e-10 else f"{trained_avg_ABX / init_avg_ABX:.0f}x"],
        ["||A|| (平均)", f"{init_avg_A:.2f}", f"{trained_avg_A:.2f}",
         "∞" if init_avg_A < 1e-10 else f"{trained_avg_A / init_avg_A:.0f}x"],
    ]

    print(tabulate(summary_data, headers=["指标", "初始化", "训练后", "变化"],
                   tablefmt="grid"))

    # 结论
    print("\n" + "="*100)
    print("📊 分析结论")
    print("="*100)

    # 零空间保持性
    BX_change = trained_avg_BX / init_avg_BX if init_avg_BX > 0 else float('inf')
    print(f"\n1️⃣  零空间保持性 (||BX|| / ||X||):")
    if BX_change < 1.5:
        print(f"   ✅ 变化 {BX_change:.2f}x - 零空间属性保持良好")
    elif BX_change < 3.0:
        print(f"   ⚠️  变化 {BX_change:.2f}x - 零空间有轻微退化")
    else:
        print(f"   ❌ 变化 {BX_change:.2f}x - 零空间显著退化，检查 BLinear 是否被冻结")

    # 适配器更新
    print(f"\n2️⃣  适配器更新 (||ABX|| / ||X||):")
    if trained_avg_ABX > 1e-4:
        print(f"   ✅ 训练后 {trained_avg_ABX:.2e} - 适配器学到了有效更新")
    elif trained_avg_ABX > 1e-6:
        print(f"   ⚠️  训练后 {trained_avg_ABX:.2e} - 适配器更新较小")
    else:
        print(f"   ❌ 训练后 {trained_avg_ABX:.2e} - 适配器几乎没变化")

    # A 权重变化
    print(f"\n3️⃣  ALinear 权重 (||A||):")
    if trained_avg_A > init_avg_A * 10:
        print(f"   ✅ 从 {init_avg_A:.2f} → {trained_avg_A:.2f} - 权重显著更新")
    elif trained_avg_A > init_avg_A * 2:
        print(f"   ⚠️  从 {init_avg_A:.2f} → {trained_avg_A:.2f} - 权重有更新但较小")
    else:
        print(f"   ❌ 从 {init_avg_A:.2f} → {trained_avg_A:.2f} - 权重几乎没变化")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
