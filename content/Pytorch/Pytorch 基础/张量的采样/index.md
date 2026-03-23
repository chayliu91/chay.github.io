---
title: "张量的采样"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 7
---

在深度学习中，“采样”通常指从一个大张量中根据特定规则提取子集。PyTorch 提供了四种核心机制：索引采样、掩码采样、Gather 采样和Where 采样。理解它们的区别对于编写高效的数据加载器、注意力机制和损失函数至关重要。

# 索引采样 (Index Sampling)
索引采样直接使用整数索引列表或切片来选取元素。这是最直观、最高效的采样方式。

基础切片适用于连续区域的采样：
```
import torch

x = torch.arange(20).reshape(4, 5)
print(x)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

# 取前 2 行，后 3 列
# 在 PyTorch 中，这种切片返回的是“视图”(View)，不是副本
# 意味着：sample 和 x 共享同一块底层内存数据
sample = x[:2, 2:]
print(sample)
# tensor([[2, 3, 4],
#         [7, 8, 9]])

print(x.storage().data_ptr() == sample.storage().data_ptr()) # True
```

使用整数或列表进行非连续、乱序采样。返回的是数据的副本 (Copy)：
```
import torch

x = torch.arange(20).reshape(4, 5)
print(x)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

indices = torch.tensor([0, 2, 3], dtype=torch.int32) # 选取第 0, 2, 3 行
sample = x[indices]
print(sample.shape) # torch.Size([3, 5])

# 内容: 第0行, 第2行, 第3行
print(sample)
# tensor([[ 0,  1,  2,  3,  4],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

# 因为 sample 是副本 (Copy)，它拥有自己独立的内存空间
# 所以它们的存储起始地址完全不同
print(sample.storage().data_ptr() == x.storage().data_ptr()) # False
```

当提供多个索引张量时，它们会配对工作（类似 zip），而不是生成笛卡尔积：
```
import torch

x = torch.arange(20).reshape(4, 5)
print(x)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])


# 选取 (0,1), (2,3), (3,4) 这三个点
row_idx = torch.tensor([0, 2, 3])
col_idx = torch.tensor([1, 3, 4])
# - 第1个点: x[0, 1] -> 值 1
# - 第2个点: x[2, 3] -> 值 13
# - 第3个点: x[3, 4] -> 值 19
sample = x[row_idx, col_idx]
# 因为选取的是具体的标量点，结果是一个一维张量，长度等于索引列表的长度
print(sample.shape) # torch.Size([3])
print(sample) # tensor([ 1, 13, 19])
```
如果索引张量形状不同但可广播，结果形状由广播后的形状决定：
```
import torch

x = torch.arange(20).reshape(4, 5)
print(x)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])


# row_idx 形状为 (3, 1): 列向量，代表行号 0, 1, 2
row_idx = torch.tensor([[0], [1], [2]])
# col_idx 形状为 (1, 2): 行向量，代表列号 0, 4
col_idx = torch.tensor([[0, 4]])

# 先对 row_idx 和 col_idx 应用“广播机制”
# 将它们扩展为相同的形状，然后再进行坐标配对

# row_idx (3, 1)  [[0],      广播后 (3, 2)  [[0, 0],
#                  [1],   =>               [1, 1],
#                  [2]]                    [2, 2]]
#
# col_idx (1, 2)  [[0, 4]]   广播后 (3, 2)  [[0, 4],
#                                          [0, 4],

# 【最终配对坐标 (Row, Col)】:
# 位置 (0,0): (0, 0) -> 值 0
# 位置 (0,1): (0, 4) -> 值 4
# 位置 (1,0): (1, 0) -> 值 5
# 位置 (1,1): (1, 4) -> 值 9
# 位置 (2,0): (2, 0) -> 值 10
# 位置 (2,1): (2, 4) -> 值 14
sample = x[row_idx, col_idx]

print(sample.shape) # torch.Size([3, 2])
print(sample)
# tensor([[ 0,  4],
#         [ 5,  9],
#         [10, 14]])
```

# 掩码采样 (Mask Sampling)
使用布尔张量（Boolean Tensor）作为过滤器，保留 `True` 位置的元素。

当使用布尔张量作为索引时，PyTorch 会：
- 遍历 `mask` 中所有为 `True` 的位置
- 提取 `x` 中对应位置的值
- 将这些值按“行优先”顺序（Row-major / C-order）展平为一个一维张量，返回的是副本 (Copy)，不是视图 (View)

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 创建掩码：大于 5 的元素
# 结果是一个与 x 形状完全相同的布尔张量 (True/False)
mask = x > 5
print(mask)
# tensor([[False, False, False],
#         [False, False,  True],
#         [ True,  True,  True]])

# 应用掩码
sample = x[mask]
print(sample) # tensor([6, 7, 8, 9])
```

直接索引会压平维度。如果需要保持原形状（将不满足条件的置为 0 或 NaN）可以配合 `masked_fill` 或 `where`：
```
Tensor.masked_fill(mask, value) → Tensor
```
```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

mask = x > 5

# 在 mask 为 True 的位置，将原张量的值替换为 value
# ~mask 将 True 变 False，False 变 True
# 在 ~mask 为 True 的位置（即原数值 <= 5 的位置），填入 0。
# 在 ~mask 为 False 的位置（即原数值 > 5 的位置），保持原值不变
sample = x.masked_fill(~mask, 0)

print(sample)

# tensor([[0, 0, 0],
#         [0, 0, 6],
#         [7, 8, 9]])
```

# Gather 采样 (Gather Sampling)
Gather 采样是沿着指定维度 `dim`，根据 index 张量中的索引值收集数据。这是处理变长序列、Top-K 选择和注意力权重的核心工具。
```
Tensor.gather(dim, index) → Tensor
```
- `dim`: 沿着哪个维度进行收集
- `index`: 索引张量。关键规则：index 的形状可以与 `input` 不同，但在非 `dim` 维度上必须匹配（或可广播）。输出形状与 `index` 完全一致

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 目标：从每一行中提取特定列的元素
# - 第0行 -> 取第0列 (值: 1)
# - 第1行 -> 取第2列 (值: 6)
# - 第2行 -> 取第1列 (值: 8)

# torch.gather 要求 index 张量的维度数必须与输入 x 相同
# 因为 x 是 2维 (3, 3)，所以 index 也必须是 2维
# 这里我们想要保留“行”的维度，只改变“列”的选取，所以形状设为 (3, 1)
index = torch.tensor([[0],
                      [2],
                      [1]])

# dim=1: 表示沿着“列”方向抓取（即固定行，变列）
# index: 指定在每个位置要抓取的具体索引值
sample = torch.gather(x, dim=1, index=index)
print(sample.shape) # torch.Size([3, 1])
# - output[0, 0]: 取 x 的第0行，列索引为 index[0,0]=0 -> x[0, 0] = 1
# - output[1, 0]: 取 x 的第1行，列索引为 index[1,0]=2 -> x[1, 2] = 6
# - output[2, 0]: 取 x 的第2行，列索引为 index[2,0]=1 -> x[2, 1] = 8
print(sample)
# tensor([[1],
#         [6],
#         [8]])

# 模拟模型输出 (Logits)
# 形状: (Batch, Seq_Len, Vocab_Size)
logits = torch.randn(2, 3, 10)


# 假设我们要提取每个位置预测概率最高的那个值
max_indices = torch.argmax(logits, dim=-1, keepdim=True)
print(max_indices.shape) # torch.Size([2, 3, 1])

# 使用 gather 提取对应的 Logits 值
probs = torch.gather(logits, dim=-1, index=max_indices)
print(probs.shape) # torch.Size([2, 3, 1])
```

# Scatter  函数
`Scatter` 是 PyTorch 中用于根据索引将数据写入张量指定位置的操作。它是 `Gather` 的逆运算。
- `Gather`：`output[i] = input[index[i]]` (读)
- `Scatter`：`output[index[i]] = input[i]` (写)

| 函数名 | 是否原地操作 | 功能描述 | 典型应用场景 |
| :--- | :---: | :--- | :--- |
| `scatter_` | 是 | **覆盖写入**。目标位置原有值被直接替换。 | One-hot 编码、掩码填充、固定位置赋值。 |
| `scatter` |  否 | **返回新张量**（通常通过先创建零张量再 `scatter_` 实现）。 | 需要保留原数据的函数式编程场景。 |
| `scatter_add_` | 是 | **累加写入**。如果多个值映射到同一位置，则求和。 | 直方图统计、图神经网络消息聚合、Segment Reduction。 |
| `scatter_reduce_` | 是 | **(PyTorch 1.12+)** 支持多种归约操作 (`sum`, `mean`, `max`, `min`, `prod`, `amax`, `amin`)。 | 复杂的分组聚合操作。 |

```
import torch

# 目标张量 (全零)
target = torch.zeros(5, dtype=torch.float32)
# 源数据
src = torch.tensor([10, 20, 30], dtype=torch.float32)

# 索引：表示 src[0] 去 target[4], src[1] 去 target[0], src[2] 去 target[2]
index = torch.tensor([4, 0, 2])

# 执行 scatter: 在第 0 维，根据 index 写入 src
target.scatter_(0, index, src)
# index[0]=4 -> target[4] = src[0] (10)
# index[1]=0 -> target[0] = src[1] (20)
# index[2]=2 -> target[2] = src[2] (30)
# 其他位置保持为 0
print(target.shape)
print(target) # tensor([20.,  0., 30.,  0., 10.])


# 目标：3行4列
target = torch.zeros(3, 4, dtype=torch.float32)
# 源数据：3行2列 (意味着每行只写2个值)
src = torch.tensor([[1, 2],
                    [3, 4],
                    [5, 6]], dtype=torch.float32)

# 索引：指定每行的第几个位置被写入
# 第0行: 写入到 col 2 和 col 3
# 第1行: 写入到 col 0 和 col 2
# 第2行: 写入到 col 1 和 col 1 (冲突测试，后者覆盖前者)
index = torch.tensor([[2, 3],
                      [0, 2],
                      [1, 1]])
# dim=1: 改变的是列索引 (Column Index)
target.scatter_(1, index, src)
print(target)

# tensor([[0., 0., 1., 2.],  <- 第0行: col2=1, col3=2
#         [3., 0., 4., 0.],  <- 第1行: col0=3, col2=4
#         [0., 6., 0., 0.]]) <- 第2行: col1=6 (src[2][1]=6 覆盖了 src[2][0]=5)
```
生成 One-Hot 编码:
```
import torch

# 定义类别总数
# 表示一共有 5 个不同的类别 (标签范围: 0 ~ 4)
num_classes = 5

# 准备标签数据
# 这是一个包含 4 个样本的批次 (Batch)
# 每个整数代表该样本所属的类别索引
# 样本0 -> 类0, 样本1 -> 类2, 样本2 -> 类4, 样本3 -> 类1
labels = torch.tensor([0, 2, 4, 1])

# 获取批次大小
batch_size = labels.size(0)

# 初始化目标张量 (One-hot 容器)
# 创建一个 4行5列 的全零矩阵
# 每一行代表一个样本，每一列代表一个类别
one_hot = torch.zeros(batch_size, num_classes)

# 调整索引形状
# labels 原始形状: (4,)
# scatter_ 要求 index 参数的维度数必须与目标张量一致 (或者是可广播的)
# unsqueeze(1) 在第1维 (列方向) 增加一个维度
# 变换后形状: (4, 1)
index = labels.unsqueeze(1)

# 第0行: 在列 index[0][0]=0 处写入 1.0 -> [1, 0, 0, 0, 0]
# 第1行: 在列 index[1][0]=2 处写入 1.0 -> [0, 0, 1, 0, 0]
# 第2行: 在列 index[2][0]=4 处写入 1.0 -> [0, 0, 0, 0, 1]
# 第3行: 在列 index[3][0]=1 处写入 1.0 -> [0, 1, 0, 0, 0]
one_hot.scatter_(1, index, 1.0)
# print(one_hot)
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 0., 1.],
#         [0., 1., 0., 0., 0.]])
```

# Where 采样 (Conditional Sampling)

`Where` 是 PyTorch 中的元素级条件选择函数。它的作用类似于编程语言中的三元运算符：
```
torch.where(condition, input, other, *, out=None) → Tensor
```
- `condition`：布尔张量（或可转换为布尔的类型），作为开关
- `input`：当条件为 真 (True) 时选取的数据源
- `other`：当条件为 假 (False) 时选取的数据源
- `out`：指定输出张量。如果提供，结果将写入此张量并返回它（原地操作模式）

三个输入张量 `(condition, input, other)` 不需要具有完全相同的形状，但它们必须能够广播到同一个形状。
```
import torch

x = torch.tensor([[1, 2], [3, 4]])

# 创建条件掩码 (Condition Mask)
# 逻辑：判断每个元素是否为偶数 (对2取余等于0)
cond = x % 2 == 0
print(cond)
# tensor([[False,  True],
#         [False,  True]])

# 使用 torch.where 进行条件选择
# 如果 condition 为 True  -> 选取 input (这里是 x) 对应位置的值
# 如果 condition 为 False -> 选取 other (这里是 0) 对应位置的值
sample = torch.where(cond, x, 0)
print(sample)

# tensor([[0, 2],
#         [0, 4]])

# input: (2, 3)
x = torch.tensor([[0, 1, 2],[3, 4, 5]])

# other: (3,) -> 会广播成 (2, 3)
# [[10, 20, 30],
#  [10, 20, 30]]
other = torch.tensor([10, 20, 30])
# condition: (2, 1) -> 会广播成 (2, 3)
# [[True,  True,  True ],
#  [False, False, False]]
cond = torch.tensor([[True], [False]])

result = torch.where(cond, x, other)
print(result.shape) # torch.Size([2, 3])
print(result)
# tensor([[ 0,  1,  2],
#         [10, 20, 30]])
```

当只提供 `condition` 一个参数时，`torch.where` 的行为完全不同，而是等价于 `torch.nonzero(condition, as_tuple=True)`：
```
torch.where(condition)
```
它不再进行值的选择，而是返回满足条件的元素的索引坐标。
```
import torch

# 创建一个 2x3 的矩阵
x = torch.tensor([[1, 0, 3],
                  [0, 5, 0]])

# 等价于 torch.where(x != 0)
indices = torch.where(x)
print(type(indices)) # <class 'tuple'>
# 第一个张量是行索引: [0, 0, 1]
# 第二个张量是列索引: [0, 2, 1]
# 组合起来表示位置: (0,0), (0,2), (1,1) 这三个位置是非零的
print(indices) # (tensor([0, 0, 1]), tensor([0, 2, 1]))

# 常用技巧：直接获取这些位置的数值
values = x[indices]
print(values)  # tensor([1, 3, 5])
```