---
title: "张量的创建"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 2
---


# 直接创建

## torch.tensor
使用工厂函数 `torch.tensor()` 直接创建张量:

```
def tensor(data, *, dtype=None, device=None, pin_memory=False, requires_grad=False)
```

```
import torch


t1 = torch.tensor(9)
print(t1.dtype)   # 输出: torch.int64 (默认整型为 64 位)
print(t1.shape)   # 输出: torch.Size([]) (0 维，即标量)
print(t1.ndim)    # 补充: 0 (验证维度数)

t2 = torch.tensor(3.0)
print(t2.dtype)   # 输出: torch.float32 (默认浮点为 32 位)
print(t2.shape)   # 输出: torch.Size([]) (0 维，即标量)

t3 = torch.tensor([1.0, 2.0])
print(t3.dtype)   # 输出: torch.float32
print(t3.shape)   # 输出: torch.Size([2]) (1 维张量)

original_list = [5.0, 6.0]
t4 = torch.tensor(original_list)
original_list[0] = 999.0  # 修改原始数据
print(t4[0].item())       # 输出: 5.0 (证明 t4 是独立副本，未受影响)

x = torch.tensor([1.0, 2.0], requires_grad=True) # 一个需要梯度的张量
y = torch.tensor(x)      # 用 tensor() 包装它
print(y.requires_grad)   # 输出: False (源码中调用了 detach_(), 默认不追踪梯度)
                         # 注意: 此时还会抛出一个 UserWarning 建议改用 clone()

t5 = torch.tensor([1, 2.5]) # 列表包含 int 和 float
print(t5.dtype)             # 输出: torch.float32 (自动提升为浮点以保留精度)
print(t5)                   # 输出: tensor([1.0000, 2.5000]) (整数部分被转换)()
```

其行为逻辑为：

**数据拷贝:** 它接收用户提供的数据，并在内存中分配新的存储空间，将数据拷贝进去，修改原始数据不会影响生成的 Tensor

**类型推断:**
- 用户指定优先: 如果传入了 `dtype` 参数，直接使用，不进行推断
- 动推断规则 (当 `dtype=None` 时):
  - 输入是 `int (9)` 推断为 `torch.int64 (LongTensor)`
  - 输入是 `float (9.0)`  推断为 `torch.float32(FloatTensor)`
  - 输入是复杂结构 (list/tuple/np.ndarray)：
    - 递归扫描所有元素以确定“最宽”的类型
    -  如果列表中混合了 `int` 和 `float`，整体推断为 `float32` (整型部分会被转为浮点)
    -  空列表: `[]` 默认推断为 `float32`
  
**维度确定:**
维度确定：它会根据输入数据的结构确定 `shape`
- 输入标量: 0 维张量 (Scalar)，shape 为 `torch.Size([])`
- 输入一维列表/元组:  1 维张量
- 输入嵌套列表:  多 维张量
- 输入 NumPy 数组：直接映射 NumPy 的 `shape` 属性到 Tensor

**梯度切断:**
- 除非显示传递了 `requires_grad=True` 参数，否则即使输入数据是一个需要梯度的 `Tensor (requires_grad=True)`，`torch.tensor` 创建的新 Tensor 默认 `requires_grad=False`
- 如果输入是 Tensor，源码会发出 UserWarning，建议改用 `.clone().detach()`

**设备默认值:**
- 如果未指定 `device` 参数，默认强制在 CPU 上创建，即使当前 CUDA 可用


## torch.Tensor
需要注意与 `torch.Tensor` 的区别，`torch.Tensor` 是一个类，默认只创建浮点型张量:
```
import torch

# 传入单个整数 -> 被解析为尺寸
t1 = torch.Tensor(9)
print(t1)
print(t1.dtype) # torch.float32
print(t1.shape) # torch.Size([9])

# 传入列表 -> 被解析为数据
t2 = torch.Tensor([9])
print(t2)       # tensor([9.])
print(t2.dtype) # torch.float32
print(t2.shape) # torch.Size([1])

# 传入嵌套列表 -> 被解析为多维数据
t3 = torch.Tensor([[1, 2],[3,4]])
print(t3)
# tensor([[1., 2.],
#         [3., 4.]])
print(t3.dtype)  # torch.float32
print(t3.shape)  # torch.Size([2, 2])

# 传入单个浮点数 -> 报错
# 既不是合法的尺寸定义，也不是一个序列
t4 = torch.Tensor(9.0)  # TypeError: new(): data must be a sequence (got float)

```

`torch.Tensor` 能够自动根据传入参数的类型切换行为模式：
- 传入纯整数，将整数当成尺寸，分配内存并执行随机初始化
- 传入可迭代对象，采用数据拷贝模式，它不会把列表里的数字当成尺寸，而是把它们当成具体的数据内容：
  - 计算所需内存大小，分配内存
  - 拷贝数据
  - 强制将类型转换成默认的浮点类型


但在现代 PyTorch 开发规范中，推荐使用 `torch.tensor()` 创建张量，原因只有一个：类型推断的确定性。