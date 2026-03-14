---
title: "张量简介"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 1
---

# 张量简介
在深度学习的上下文中，张量是向量和矩阵向任意维度推广的通用形式。它本质上是一个多维数组。

![](./image/tensors.png)

- 标量 (Scalar)：0 维张量，一个单独的数
- 向量 (Vector)：1 维张量，需要一个索引 `x[2]`
- 矩阵 (Matrix)：2 维张量，需要两个索引 `x[1, 0]`
- 张量 (Tensor)：3 维及以上，需要多个索引 `x[0, 2, 1]`