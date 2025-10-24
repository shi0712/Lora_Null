# 零空间验证修改说明

## 概述

为了验证 LoRA-Null-v2 中"零空间"的数学近似性，我们对代码进行了以下修改，添加了实时验证功能。

## 修改内容

### 1. `adapterlib/act_aware_utils.py`

**修改位置**: `calib_cov_distribution` 函数

**主要改动**:
- 在计算协方差矩阵的同时，缓存前 100 个输入样本
- 这些样本将用于后续验证零空间属性

```python
# 新增代码片段（第120-132行）
max_cached_samples = 100  # 缓存前100个token用于验证

# 在hook中缓存输入样本
if not hasattr(module, 'cached_input_samples'):
    module.cached_input_samples = []
if len(module.cached_input_samples) < max_cached_samples:
    num_to_cache = min(max_cached_samples - len(module.cached_input_samples), input.size(0))
    module.cached_input_samples.append(input[:num_to_cache].clone())
```

### 2. `adapterlib/decomposition.py`

**修改位置**: `decompose_to_adapter2` 函数中的 `singular_aware_2` 分支

**主要改动**:
- 在初始化适配器后，使用缓存的输入样本验证零空间属性
- 计算并输出以下指标：
  - `||BX||`: 投影到零空间后的激活范数
  - `||ABX||`: 适配器初始输出的范数（应该≈0）
  - `||BX|| / ||X||`: 相对误差（衡量零空间近似程度）
  - 协方差矩阵的最小/最大特征值比

```python
# 新增验证代码（第486-534行）
if hasattr(linear, 'cached_input_samples'):
    X = linear.cached_input_samples.float()

    # 计算 B @ X
    B_weight = V.t()  # U_min_K^T
    BX = B_weight @ X.t()

    # 计算 A @ B @ X (A初始化为0)
    A_weight = U.mul(S.sqrt())  # 全0矩阵
    ABX = A_weight @ BX

    # 统计分析
    BX_norm = torch.norm(BX, dim=0).mean().item()
    ABX_norm = torch.norm(ABX, dim=0).mean().item()
    X_norm = torch.norm(X, dim=1).mean().item()

    # 输出验证结果
    print(f"||BX|| / ||X||: {BX_norm / X_norm:.6e} (应该 << 1)")
    print(f"||ABX|| / ||X||: {ABX_norm / X_norm:.6e} (应该 ≈ 0)")
```

### 3. `test_null_space.py` (新文件)

**用途**: 专门用于验证零空间属性的测试脚本

**使用方法**:
```bash
# 快速测试（使用小模型和少量样本）
python test_null_space.py --model_id meta-llama/Llama-2-7b-hf --r 128 --calib_loader_size 16

# 完整测试（使用更多样本）
python test_null_space.py --model_id meta-llama/Llama-2-7b-hf --r 128 --calib_loader_size 256
```

## 验证指标解释

### 1. `||BX|| / ||X||`

**物理意义**: 输入激活投影到零空间后的相对幅度

**理论预期**: << 1 (远小于1)

**解释**:
- 如果协方差矩阵的最小特征值为 λ_min
- 那么 `||BX|| / ||X|| ≈ sqrt(λ_min)`
- 例如 λ_min = 1e-4 → 比值应约为 1e-2

**实际意义**:
- 比值越小，说明选择的零空间方向越准确
- 这些方向上的激活变化很小，在这里添加适配器对原始输出影响最小

### 2. `||ABX|| / ||X||`

**物理意义**: 适配器初始化时的输出变化

**理论预期**: ≈ 0 (在机器精度范围内应 < 1e-6)

**解释**:
- 因为 A 初始化为全0矩阵
- 所以 ABX = 0 @ (BX) = 0
- 实际由于浮点误差可能有微小非零值

**实际意义**:
- 验证初始化时模型输出确实保持不变
- 如果这个值很大（> 1e-3），说明初始化有问题

### 3. `Eigenvalue ratio (min/max)`

**物理意义**: 协方差矩阵的条件数（最小特征值/最大特征值）

**理论预期**: 应该很小（通常 < 1e-3）

**解释**:
- 反映了数据在不同方向上的方差差异
- 比值越小，说明存在明显的"低方差方向"（零空间）

**实际意义**:
- 如果比值接近1，说明数据在所有方向上方差都差不多，难以找到真正的零空间
- 如果比值很小，说明数据具有明显的低维结构，适合使用零空间方法

## 预期输出示例

运行测试脚本后，你应该会看到类似这样的输出：

```
============================================================
[Null Space Verification] Layer: Linear
============================================================
Input samples shape: torch.Size([100, 4096])
Covariance eigenvalues (min 128): [1.2e-05, 1.5e-05, 1.8e-05, 2.1e-05, 2.4e-05]...
Covariance eigenvalues (max 128): [145.3, 132.1, 128.7, 121.5, 115.2]...
Eigenvalue ratio (min/max): 8.26e-08
------------------------------------------------------------
||X|| (avg): 12.345678
||BX|| (avg): 0.001234
||ABX|| (avg): 0.000000
||BX|| / ||X||: 9.998e-05 (应该 << 1) ✓
||ABX|| / ||X||: 1.234e-09 (应该 ≈ 0) ✓
============================================================
```

## 理论验证

### 为什么是"近似"零空间？

1. **有限样本**: 只使用256个样本估计协方差矩阵，不是真实的数据分布
2. **数值精度**: SVD分解和浮点运算引入误差
3. **秩截断**: 只使用最小的r个特征向量，不是完整的零空间

因此，严格来说 `BX ≠ 0`，而是 `||BX|| << ||X||`

### 数学推导

对于协方差矩阵 C = X^T X 的特征分解：
```
C v_i = λ_i v_i
```

那么：
```
||X v_i||² = v_i^T (X^T X) v_i = v_i^T C v_i = λ_i
```

所以：
```
||X v_i|| = sqrt(λ_i)
```

选择最小的r个特征向量构成 U_min_K = [v_{n-r+1}, ..., v_n]，则：
```
||X U_min_K||_F² = Σ λ_i  (i = n-r+1 to n)
```

如果这些特征值都很小（例如 < 1e-4），那么：
```
||BX|| = ||U_min_K^T X|| ≈ sqrt(Σ λ_i) << ||X||
```

## 故障排查

如果验证结果不符合预期：

1. **||BX|| / ||X|| > 0.1**:
   - 可能原因：秩r太大，包含了一些重要方向
   - 解决方案：减小r，或检查协方差矩阵计算是否正确

2. **||ABX|| / ||X|| > 1e-3**:
   - 可能原因：A的初始化有问题（没有正确设为0）
   - 解决方案：检查 decomposition.py:483 是否正确执行 U.zero_()

3. **Eigenvalue ratio > 0.1**:
   - 可能原因：数据分布过于均匀，没有明显的低方差方向
   - 解决方案：增加校准样本数量，或使用不同的校准数据集

## 总结

通过这些修改，我们可以定量验证：

1. ✅ **零空间近似的准确性**: 通过 `||BX|| / ||X||` 衡量
2. ✅ **初始化的正确性**: 通过 `||ABX|| / ||X||` 验证
3. ✅ **数据特性的适用性**: 通过特征值比衡量

这些指标帮助我们理解为什么 LoRA-Null-v2 能够在保持预训练知识的同时进行微调。
