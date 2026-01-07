---
layout: post
title: 《Python for Data Analysis, 3E》 阅读笔记（更新中）
date: 2025-12-30 09:02:08 +0000
tags: [Python, Data Analysis, 阅读笔记]
---

官方在线版：https://wesmckinney.com/book/

## 各章节内容

* **第 1 章：准备工作**

  介绍环境搭建、必要的安装步骤以及本书的学习路线。

*   **第 2 章：Python 语言基础、IPython 和 Jupyter Notebooks**
    
    讲解 Python 的基础语法，并介绍如何高效使用 IPython 交互式解释器和 Jupyter Notebook 开发环境。
    
*   **第 3 章：内置数据结构、函数和文件**
    
    介绍 Python 核心的内置数据结构（如列表、元组、字典），以及函数的定义和基础文件读写操作。
    
*   **第 4 章：NumPy 基础：数组与向量化计算**
    
    核心讲解 NumPy 库，重点在于 ndarray 多维数组对象的使用及高效的数学运算。
    
*   **第 5 章：pandas 入门**
    
    引入 pandas 库，详细介绍其两个核心数据结构：Series（一维）和 DataFrame（二维/表格）。
    
*   **第 6 章：数据加载、存储与文件格式**
    
    演示如何从各种来源（如 CSV、Excel、JSON、SQL 数据库）读取数据，以及如何将数据保存为不同格式。
    
*   **第 7 章：数据清洗与准备**
    
    探讨处理缺失数据、重复数据、字符串操作以及数据转换和替换的技巧。
    
*   **第 8 章：数据规整：连接、合并与重塑**
    
    讲解如何对多个数据集进行合并（Merge）、连接（Join）以及对数据结构进行重塑（Reshape/Pivot）。
    
*   **第 9 章：绘图与可视化**
    
    介绍使用 Matplotlib 等工具库创建静态图表和进行数据可视化的基础方法。
    
*   **第 10 章：数据聚合与分组运算**
    
    深入讲解 GroupBy 机制，展示如何对数据进行拆分、应用函数和组合，以及如何创建透视表。
    
*   **第 11 章：时间序列**
    
    专门讲述日期和时间的处理工具，以及如何对时间序列数据进行索引、频率转换和移动窗口统计。
    
*   **第 12 章：Python 建模库介绍**
    
    介绍如何将 pandas 中的数据与 statsmodels 和 scikit-learn 等主流统计与机器学习库进行衔接。
    
*   **第 13 章：数据分析案例**
    
    通过几个综合性的真实数据集案例，演示如何串联前面章节的知识解决实际问题。
    
*   **附录 A & B**
    
    分别补充了 NumPy 的进阶广播/底层功能以及 IPython 系统的更多高级用法。

## 阅读计划

- **略过**：第 9 章、附录 A & B
- **浏览**：第 1 章、第 2 章、第 3 章
- **摘录**：第 4 章、第 6 章、第 12 章
- **精读**：第 5 章、第 7 章、第 8 章、第 10 章、第 11 章、第 13 章

标记为**略过**和**浏览**的章节不做记录，这些内容已经比较熟悉或者工作中用不到；标记为**摘录**的章节会记录其中的要点，这些内容基本都是工具的使用，简单记录用于后续查找即可；标记为**精读**的章节，会在原文基础上写出自己的思考和见解，这些内容是讲实际应用的，属于内功，需要好好消化。

## 第 4 章：NumPy 基础：数组与向量化计算

NumPy 是 Numerical Python 的缩写，是 Python 数值计算最重要的基础包之一。许多提供科学功能的计算包使用 NumPy 的数组对象作为数据交换的标准接口语言之一。

> 提示
>
> 下文的所有代码片段省略了 numpy 的引入：`import numpy as np`。虽然也可以用 `from numpy import *` 引入 numpy，这样可以省略 `np`。但作者建议不要养成这种习惯。因为 numpy 命名空间很大，包含许多与内置 Python 函数（例如 min 和 max）冲突的同名函数。
{: .prompt-tip }

### 4.1 NumPy ndarray：多维数组对象

- **创建多维数组**

  ```python
  np.array([[1.5, -0.1, 3], [0, -3, 6.5]]) #直接传入
  np.array([1, 2, 3], dtype=np.float64) # 传入时指定数据类型
  np.zeros(10) # 一维全 0
  np.ones(10) # ..1
  np.zeros((3, 6)) # 二维全 0
  np.ones((3, 6)) # ..1
  np.empty((2, 3, 2)) # 三维空白数组，不保证默认值是什么
  np.arange(15) # 一维 0..15 数组
  ```

  NumPy 提供的常用多维数组（ndarray）创建函数如下表。

  | Function            | Description                                                  |
  | :------------------ | :----------------------------------------------------------- |
  | `array`             | 通过推断数据类型或显式指定数据类型，将输入数据（列表、元组、数组或其他序列类型）转换为 ndarray；默认复制输入数据及其类型 |
  | `asarray`           | 将输入转换为 ndarray，但如果输入已经是 ndarray，则不复制，而是直接引用 |
  | `arange`            | 与 Python 内置的 `range` 类似，但返回 ndarray 而不是 list    |
  | `ones, ones_like`   | 生成具有给定形状和数据类型（默认 float64）的全 1 数组；`ones_like` 接收另一个数组并生成具有相同形状和数据类型的 `ones` 数组 |
  | `zeros, zeros_like` | 与 `ones` 和 `ones_like` 类似，但生成全 0 数组               |
  | `empty, empty_like` | 通过分配新内存来创建新数组，但不填充任何值                   |
  | `full, full_like`   | 生成给定形状和数据类型（默认 float64）的数组，并将所有值设置为通过参数指定的 “填充值”； `full_like` 接收另一个数组并生成具有相同形状和数据类型的填充数组 |
  | `eye, identity`     | 创建一个 N × N 单位矩阵（对角线上为 1，其他位置为 0）        |

- **获取数组维度**

  ```python
  data.ndim # 2
  ```

- **获取数组形状**

  ```python
  data.shape # (2, 3)
  ```

- **获取数组元素类型**

  ```python
  data.dtype # dtype('float64')
  ```

  NumPy 提供的数据类型如下，常见的基本数据类型就不翻译了。

  | Type                                    | Type code      | Description                                                  |
  | :-------------------------------------- | :------------- | :----------------------------------------------------------- |
  | `int8, uint8`                           | `i1, u1`       | Signed and unsigned 8-bit (1 byte) integer types             |
  | `int16, uint16`                         | `i2, u2`       | Signed and unsigned 16-bit integer types                     |
  | `int32, uint32`                         | `i4, u4`       | Signed and unsigned 32-bit integer types                     |
  | `int64, uint64`                         | `i8, u8`       | Signed and unsigned 64-bit integer types                     |
  | `float16`                               | `f2`           | Half-precision floating point                                |
  | `float32`                               | `f4 or f`      | Standard single-precision floating point; compatible with C float |
  | `float64`                               | `f8 or d`      | Standard double-precision floating point; compatible with C double and Python `float` object |
  | `float128`                              | `f16 or g`     | Extended-precision floating point                            |
  | `complex64`, `complex128`, `complex256` | `c8, c16, c32` | 分别由两个 32、64 或 128 浮点数表示的复数                    |
  | `bool`                                  | ?              | Boolean type storing `True` and `False` values               |
  | `object`                                | O              | Python object type; a value can be any Python object         |
  | `string_`                               | S              | Fixed-length ASCII string type (1 byte per character); for example, to create a string data type with length 10, use `'S10'` |
  | `unicode_`                              | U              | Fixed-length Unicode type (number of bytes platform specific); same specification semantics as `string_` (e.g., `'U10'`) |

- **转换数组元素类型**

  ```python
  data.astype(np.int64) # 浮点数转整数时会发生截断，丢失小数位
  ```

  浮点数转整数时会发生截断，丢失小数位。

  如果由于某种原因转换失败（例如无法转换为 `float64` 的字符串），则会引起 `ValueError`。

  另外，这个方法的成功调用一定会产生一次数据复制。

- **转换数组形状**

  ```python
  data.reshape(3, 2) # 元素数量不一致会引起 ValueError
  ```

- **数组的算数运算**

  ```python
  arr = np.array([[1., 2., 3.], [4., 5., 6.]])
  arr * arr 	# [[1., 4., 9.], [16., 25., 36.]]
  arr - arr 	# [[0., 0., 0.], [0., 0., 0.]]
  1 / arr		# [[1., 0.5, 0.3333], [0.25, 0.2, 0.1667]]
  arr ** 2	# [[1., 4., 9.], [16., 25., 36.]]
  np.array([[0., 4., 1.], [7., 2., 12.]]) > arr # [[False, True, False], [True, False, True]]
  ```

  NumPy 的数组无需编写任何 for 循环，就能表达对数据的批量操作。NumPy 用户将此称为向量化。

  大小相等的数组之间可以运用任何算术运算。对 NumPy 数组的标量运算会传播到数组的所有元素上。

- **数组的索引和切片**

  ```python
  arr = np.arange(10) 	# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  arr[5] 					# 5
  arr[5:8] 				# [5, 6, 7]
  arr[5:8] = 12 			# [0, 1, 2, 3, 4, 12, 12, 12, 8, 9] 这个修改会反应在源数组上
  arr[5:8].copy() = 12 	# 这个修改不会反应在源数组上
  arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  arr2d[:2, 1:] 			# [[2, 3],[5, 6]] 切片可以多维度一起切
  arr2d[True, False, True] # [[1, 2, 3], [7, 8, 9]] 也可以用布尔值选择需要的元素，这种方式始终会创建一个新副本
  arr2d[[1, 2]]			# [[4, 5, 6], [7, 8, 9]] 可以把需要元素的索引位置包成数组，这叫花式索引，也会创建副本
  arr2d[[1, 2],[0, 0]]	# [4, 7]
  arr2d[[1, 2]][:, [1, 0]]# [[7, 8, 9], [1, 2, 3]] 这个意思是把 [1,2] 这两个位置的元素索引到，用 : 全选，然后再用 [1,0] 重排序
  ```

- **数组的转置**

  ```python
  arr = np.arange(15).reshape((3, 5))
  # array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])
  arr.T 或者 arr.transpose()
  # array([[ 0,  5, 10],
            [ 1,  6, 11],
            [ 2,  7, 12],
            [ 3,  8, 13],
            [ 4,  9, 14]])
  ```

  不用担心这个操作会很慢，它没有重排列内存元素，而是改变了解释内存元素的方式。

- **矩阵的内积**

  ```python
  # 两种操作都可以
  np.dot(arr.T, arr)
  arr.T @ arr
  ```

### 4.2 伪随机数生成

```python
rng = np.random.default_rng(seed=12345) # 配置随机数生成器的种子，相同的种子生成的随机数序列是相同的，所以是伪随机
rng.random.standard_normal(size=(4, 4)) 或 np.random.standard_normal(size=(4, 4))
# array([[-0.2047,  0.4789, -0.5194, -0.5557],
          [ 1.9658,  1.3934,  0.0929,  0.2817],
          [ 0.769 ,  1.2464,  1.0072, -1.2962],
          [ 0.275 ,  0.2289,  1.3529,  0.8864]])
```

NumPy 的这个随机数生成器比 Python 内置的 `random` 快一整个数量级。随机数生成器对象可调用的常用方法如下表。

| Function      | Description                                                  |
| :------------ | :----------------------------------------------------------- |
| `seed`        | 向随机数生成器传递种子（以确保随机结果的可复现性）。         |
| `permutation` | 返回一个序列的随机排列，或者返回一个乱序的整数范围序列。     |
| `shuffle`     | 对一个序列进行就地（in-place）随机排列（修改原数据）。       |
| `rand`        | 从均匀分布中抽取样本。                                       |
| `randint`     | 从给定的低值到高值范围内随机抽取整数。                       |
| `randn`       | 从标准正态分布（均值为0，标准差为1）中抽取样本（类似 MATLAB 的接口）。 |
| `binomial`    | 从二项分布中抽取样本。                                       |
| `normal`      | 从正态（高斯）分布中抽取样本。                               |
| `beta`        | 从 Beta 分布中抽取样本。                                     |
| `chisquare`   | 从卡方（Chi-square）分布中抽取样本。                         |
| `gamma`       | 从 Gamma 分布中抽取样本。                                    |
| `uniform`     | 从 [0, 1) 区间的均匀分布中抽取样本。                         |

### 4.3 通用函数：快速的逐元素数组函数

通用函数（或 ufunc）是对 ndarray 中的数据执行逐元素操作的函数。您可以将它们视为简单函数的快速向量化包装器，这些函数采用一个或多个标量值并生成一个或多个标量结果。

- **一元通用函数**

    ```python
    arr = np.arange(10)
    np.sqrt(arr) # 逐元素开方
    np.exp(arr) # 逐元素取 e 的 val 次方
    ```
    
     常见的一元通用函数如下表。

  | Function                                                     | Description                                                  |
  | :----------------------------------------------------------- | :----------------------------------------------------------- |
  | `abs`, `fabs`                                                | 计算整数、浮点数或复数的绝对值。对于非复数值，使用 `fabs` 会更快。 |
  | `sqrt`                                                       | 计算每个元素的平方根（相当于 `arr ** 0.5`）。                |
  | `square`                                                     | 计算每个元素的平方（相当于 `arr ** 2`）。                    |
  | `exp`                                                        | 计算每个元素的指数 $e^x$。                                   |
  | `log`, `log10`, `log2`, `log1p`                              | 分别计算自然对数（底数为 $e$）、底数为 10 的对数、底数为 2 的对数，以及 $\log(1 + x)$。 |
  | `sign`                                                       | 计算每个元素的符号：1（正数）、0（零）或 -1（负数）。        |
  | `ceil`                                                       | 计算每个元素的 Ceiling 值（即不小于该值的最小整数）。        |
  | `floor`                                                      | 计算每个元素的 Floor 值（即不大于该值的最大整数）。          |
  | `rint`                                                       | 将元素四舍五入到最接近的整数，保留 `dtype`。                 |
  | `modf`                                                       | 将数组的小数部分和整数部分作为两个独立的数组返回。           |
  | `isnan`                                                      | 返回一个布尔数组，指示哪些值是 `NaN`（非数字）。             |
  | `isfinite`, `isinf`                                          | 分别返回布尔数组，指示哪些值是有限的（非 `inf`，非 `NaN`）或无限的。 |
  | `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh`                  | 普通三角函数和双曲三角函数。                                 |
  | `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh` | 反三角函数。                                                 |
  | `logical_not`                                                | 计算每个元素 `not x` 的真值（相当于 `~arr`）。               |


- **二元通用函数**

  ```python
  x = rng.standard_normal(8) # [-1.3678,  0.6489,  0.3611, -1.9529,  2.3474,  0.9685, -0.7594, 0.9022]
  y = rng.standard_normal(8) # [-0.467 , -0.0607,  0.7888, -1.2567,  0.5759,  1.399 ,  1.3223, -0.2997]
  np.maximum(x, y)		   # [-0.467 ,  0.6489,  0.7888, -1.2567,  2.3474,  1.399 ,  1.3223, 0.9022]
  ```
  
  常见的二元通用函数如下表。
  
  
  | Function                                                     | Description                                                  |
  | :----------------------------------------------------------- | :----------------------------------------------------------- |
  | `add`                                                        | 将数组中对应的元素相加。                                     |
  | `subtract`                                                   | 从第一个数组中减去第二个数组的元素。                         |
  | `multiply`                                                   | 数组元素相乘。                                               |
  | `divide`, `floor_divide`                                     | 除法或整除（丢弃余数）。                                     |
  | `power`                                                      | 将第一个数组中的元素作为底数，第二个数组中的元素作为指数计算。 |
  | `maximum`, `fmax`                                            | 逐元素计算最大值。`fmax` 会忽略 `NaN`。                      |
  | `minimum`, `fmin`                                            | 逐元素计算最小值。`fmin` 会忽略 `NaN`。                      |
  | `mod`                                                        | 逐元素计算模（即除法的余数）。                               |
  | `copysign`                                                   | 将第二个参数的符号复制给第一个参数。                         |
  | `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal` | 执行逐元素的比较，生成布尔数组（相当于中缀运算符 `>`, `>=`, `<`, `<=`, `==`, `!=`）。 |
  | `logical_and`, `logical_or`, `logical_xor`                   | 计算逐元素的逻辑真值操作（相当于中缀运算符 `&`, `|`, `^`）。 |

- **返回多个数组的通用函数**

  ```python
  arr = rng.standard_normal(7) * 5 # [ 4.5146, -8.1079, -0.7909,  2.2474, -6.718 , -0.4084,  8.6237]
  remainder, whole_part = np.modf(arr) # 分别返回浮点数的证书和小数部分
  remainder  # [ 0.5146, -0.1079, -0.7909,  0.2474, -0.718 , -0.4084,  0.6237]
  whole_part # [ 4., -8., -0.,  2., -6., -0.,  8.]
  ```

### 4.4 使用数组进行面向数组编程

一般来说，向量化数组操作通常比纯 Python 等价物快得多，这对任何类型的数值计算都有最大的影响。

举一个简单的例子，假设我们希望在规则的值网格上计算函数 `sqrt(x^2 + y^2)`。 `numpy.meshgrid` 函数接受两个一维数组并生成两个二维矩阵，对应于这两个数组中的所有 `(x, y)` 对：

```python
points = np.arange(-5, 5, 0.01) # 100 equally spaced points
xs, ys = np.meshgrid(points, points)
ys
# [[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
#  [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
#  [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
#  ...,
#  [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
#  [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
#  [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]]
np.sqrt(xs ** 2 + ys ** 2)
# [[7.0711, 7.064 , 7.0569, ..., 7.0499, 7.0569, 7.064 ],
#  [7.064 , 7.0569, 7.0499, ..., 7.0428, 7.0499, 7.0569],
#  [7.0569, 7.0499, 7.0428, ..., 7.0357, 7.0428, 7.0499],
#  ...,
#  [7.0499, 7.0428, 7.0357, ..., 7.0286, 7.0357, 7.0428],
#  [7.0569, 7.0499, 7.0428, ..., 7.0357, 7.0428, 7.0499],
#  [7.064 , 7.0569, 7.0499, ..., 7.0428, 7.0499, 7.0569]]
```

- **将条件逻辑表示为数组运算**

  ```python
  xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
  yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
  cond = np.array([True, False, True, True, False])
  np.where(cond, xarr, yarr) # 当 cond[i] 为 True 时，取 xarr[i]，否则取 yarr[i]
  # 结果：[1.1, 2.2, 1.3, 1.4, 2.5]
  ---
  arr = rng.standard_normal((4, 4))
  # [[ 2.6182,  0.7774,  0.8286, -0.959 ],
  #  [-1.2094, -1.4123,  0.5415,  0.7519],
  #  [-0.6588, -1.2287,  0.2576,  0.3129],
  #  [-0.1308,  1.27  , -0.093 , -0.0662]]
  np.where(arr > 0, 2, -2) # 把正数替换为 2，负数替换为 -2
  # [[ 2,  2,  2, -2],
  #  [-2, -2,  2,  2],
  #  [-2, -2,  2,  2],
  #  [-2,  2, -2, -2]]
  ```

- **数学和统计方法**

  ```python
  arr = rng.standard_normal((5, 4))
  # [[-1.1082,  0.136 ,  1.3471,  0.0611],
  #  [ 0.0709,  0.4337,  0.2775,  0.5303],
  #  [ 0.5367,  0.6184, -0.795 ,  0.3   ],
  #  [-1.6027,  0.2668, -1.2616, -0.0713],
  #  [ 0.474 , -0.4149,  0.0977, -1.6404]]
  arr.mean() 或 np.mean(arr) # -0.08719744457434529 平均值
  arr.mean(axis=1) # [0.109, 0.3281, 0.165, -0.6672, -0.3709] 每行元素的平均值
  arr.sum() # -1.743948891486906 求和
  arr.sum(axis=0) # [-1.6292,  1.0399, -0.3344, -0.8203] 每列元素的和
  ---
  arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
  arr.cumsum() # [ 0,  1,  3,  6, 10, 15, 21, 28] 累积和
  arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
  # [[0, 1, 2],
  #  [3, 4, 5],
  #  [6, 7, 8]]
  arr.cumsum(axis=0) # 按行累计
  # [[ 0,  1,  2],
  #  [ 3,  5,  7],
  #  [ 9, 12, 15]]
  arr.cumsum(axis=1) # 按列累计
  # [[ 0,  1,  3],
  #  [ 3,  7, 12],
  #  [ 6, 13, 21]]
  ```

  常用的数组统计方法如下表。

  | Method             | Description                                                  |
  | :----------------- | :----------------------------------------------------------- |
  | `sum`              | 计算数组中所有元素的和，或者沿某个轴的和。零长度数组的和为 0。 |
  | `mean`             | 算术平均值。零长度数组的平均值为 NaN。                       |
  | `std`, `var`       | 分别计算标准差 (Standard Deviation) 和方差 (Variance)。可以选择自由度调整（默认分母为 $n$）。 |
  | `min`, `max`       | 最小值和最大值。                                             |
  | `argmin`, `argmax` | 分别返回最小值和最大值元素的索引（位置）。                   |
  | `cumsum`           | 从 0 开始的元素累积和 (Cumulative sum)。                     |
  | `cumprod`          | 从 1 开始的元素累积积 (Cumulative product)。                 |


- **布尔数组方法**

  ```python
  arr = rng.standard_normal(100)
  (arr > 0).sum() # 正数的数量
  (arr <= 0).sum() # 负数的数量
  ---
  bools = np.array([False, False, True, False])
  bools.any() # 是否任意为 True
  bools.all() # 是否全部为 True
  ```

- **就地排序**

  ```python
  arr = rng.standard_normal(6) # [ 0.0773, -0.6839, -0.7208,  1.1206, -0.0548, -0.0824]
  arr.sort() # [-0.7208, -0.6839, -0.0824, -0.0548,  0.0773,  1.1206]
  np.sort(arr) # 结果一样，但是这个不是就地排序，而是返回了排序后的副本
  ---
  arr = rng.standard_normal((5, 3))
  # [[ 0.936 ,  1.2385,  1.2728],
  #  [ 0.4059, -0.0503,  0.2893],
  #  [ 0.1793,  1.3975,  0.292 ],
  #  [ 0.6384, -0.0279,  1.3711],
  #  [-2.0528,  0.3805,  0.7554]]
  arr.sort(axis=0)
  # 按列排序
  # [[-2.0528, -0.0503,  0.2893],
  #  [ 0.1793, -0.0279,  0.292 ],
  #  [ 0.4059,  0.3805,  0.7554],
  #  [ 0.6384,  1.2385,  1.2728],
  #  [ 0.936 ,  1.3975,  1.3711]]
  arr.sort(axis=1)
  # 按行排序
  # [[-2.0528, -0.0503,  0.2893],
  #  [-0.0279,  0.1793,  0.292 ],
  #  [ 0.3805,  0.4059,  0.7554],
  #  [ 0.6384,  1.2385,  1.2728],
  #  [ 0.936 ,  1.3711,  1.3975]]
  ```

- **去重和其他集合逻辑**

  ```python
  np.array(["Bob", "Will", "Joe", "Bob", "Will", "Joe", "Joe"])
  np.unique(names) # ['Bob', 'Joe', 'Will'] 去重并返回排序后的结果
  ---
  values = np.array([6, 0, 0, 3, 2, 5, 6])
  np.in1d(values, [2, 3, 6]) # [True, False, False, True, True, False, True] 测试 values 的各元素是否在给定集合中
  ```

  其他集合运算相关的方法如下表。

  | Method              | Description                                                  |
  | :------------------ | :----------------------------------------------------------- |
  | `unique(x)`         | 计算 x 中排序后的唯一元素（即去重并排序）。                  |
  | `intersect1d(x, y)` | 计算 x 和 y 的交集（即两者都有的元素），并排序。             |
  | `union1d(x, y)`     | 计算 x 和 y 的并集（即两者中所有的不重复元素），并排序。     |
  | `in1d(x, y)`        | 计算一个布尔数组，表示 x 的每个元素是否包含在 y 中。         |
  | `setdiff1d(x, y)`   | 计算差集，即在 x 中但不在 y 中的元素。                       |
  | `setxor1d(x, y)`    | 计算对称差集，即存在于其中一个数组中，但不同时存在于两个数组中的元素。 |

### 4.5 使用数组进行文件输入和输出

```python
arr = np.arange(10)
np.save("some_array", arr) # 数组默认以未压缩的原始二进制格式保存，文件扩展名为 .npy
np.load("some_array.npy") # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
np.savez("array_archive.npz", a=arr, b=arr) # 一次保存多个数组
arch = np.load("array_archive.npz")
arch["b"] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
np.savez_compressed("arrays_compressed.npz", a=arr, b=arr) # 以压缩形式保存
```

### 4.6 线性代数

参考：https://wesmckinney.com/book/numpy-basics#numpy_linalg

## 第 5 章：pandas 入门

pandas 包含数据结构和数据操作工具，旨在使 Python 中的数据清理和分析变得快速、方便。

pandas 通常与 NumPy 和 SciPy 等数值计算工具、statsmodels 和 scikit-learn 等分析库以及 matplotlib 等数据可视化库配合使用。 

pandas 很大程度上借鉴了 NumPy 基于数组的计算风格，特别是在两方面：基于数组的函数和不使用 for 循环的数据处理（向量化）。

虽然 pandas 采用了 NumPy 的许多编码习惯，但最大的区别是 pandas 是为处理表格或异构数据而设计的。相比之下，NumPy 最适合处理同质类型的数值数组数据。

> 提示
>
> 下文的所有代码片段省略了包引入代码：
>
> ```python
> import numpy as np
> import pandas as pd
> from pandas import Series, DataFrame # 这两个使用非常频繁，事先引入
> ```
> {: .prompt-tip }

### 5.1 pandas 数据结构简介

使用 pandas 需要熟悉它的两种主力数据结构：Series 和 DataFrame。

#### Series

Series 是一个类似一维数组的对象，包含一系列索引和同类型的值。

可以仅使用数组创建最简单的 Series：

```python
obj = pd.Series([4, 7, -5, 3])
obj
# 0    4
# 1    7
# 2   -5
# 3    3
# dtype: int64
```

在交互模式下输出的 Series，左侧显示索引，右侧显示值。

由于没有为数据指定索引，因此会创建一个由整数 `0` 到 `N - 1`（其中 `N` 是数据的长度）组成的默认索引。

可以分别通过其 `array` 和 `index` 属性获取 Series 的数组表示形式和索引对象：

```python
obj.array
# <PandasArray>
# [4, 7, -5, 3]
# Length: 4, dtype: int64
obj.index
# RangeIndex(start=0, stop=4, step=1)
```

`.array` 属性的结果是一个 `PandasArray`，它通常包装了 NumPy 数组，但也可以包含特殊的扩展数组类型，这将在第 7.3 章：扩展数据类型中详细讨论。

通常来说，需要创建一个带有指定索引的 Series，该索引用标签标识每个数据点：

```python
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
obj2
# d    4
# b    7
# a   -5
# c    3
# dtype: int64
obj2.index
# Index(['d', 'b', 'a', 'c'], dtype='object')
```

与 NumPy 数组相比，在选择单个值或一组值时可以在索引中使用标签：

```python
obj2["a"]
# -5
obj2["d"] = 6
obj2[["c", "a", "d"]]
# c    3
# a   -5
# d    6
# dtype: int64
```

使用 NumPy 函数或类似 NumPy 的操作（例如使用布尔数组进行过滤、标量乘法或应用数学函数）将保留索引值链接：

```python
obj2[obj2 > 0]
# d    6
# b    7
# c    3
# dtype: int64
obj2 * 2
# d    12
# b    14
# a   -10
# c     6
# dtype: int64
np.exp(obj2)
# d     403.428793
# b    1096.633158
# a       0.006738
# c      20.085537
# dtype: float64
```

可以将 Series 视为固定长度的有序字典，因为它是索引值到数据值的映射。它可以在许多可能使用字典的情况下使用：

```python
"b" in obj2
# True
"e" in obj2
# False
```

如果 Python 字典中包含数据，则可以通过传递字典来创建 Series：

```python
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3
# Ohio      35000
# Texas     71000
# Oregon    16000
# Utah       5000
# dtype: int64
```

Series 可以使用其 `to_dict` 方法转换回字典：

```python
obj3.to_dict()
# {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
```

当仅传递字典时，Series 中的索引顺序遵循字典的 `key` 方法计算的顺序，即键插入顺序。

但是可以通过传递字典键作为索引进行过滤和排序：

```python
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
obj4
# California        NaN
# Ohio          35000.0
# Oregon        16000.0
# Texas         71000.0
# dtype: float64
```

上述代码片段中，在 `sdata` 中找到的三个值被放置在适当的位置，但由于没有找到 `“California”` 的值，因此它显示为 `NaN`（Not a Number），这在 pandas 中被认为是缺失标记或 NA 值。由于 `“Utah”` 未包含在 `states` 中，因此它被排除在结果对象之外。

pandas 中的 `isna` 和 `notna` 函数用于检测丢失的数据：

```python
pd.isna(obj4) 或 obj4.isna()
# California     True
# Ohio          False
# Oregon        False
# Texas         False
# dtype: bool
pd.notna(obj4) 或 obj4.notna()
# California    False
# Ohio           True
# Oregon         True
# Texas          True
# dtype: bool
```

很多时候会用到一个有用的 Series 功能：算术运算中按索引标签自动对齐：

```python
obj3
# Ohio      35000
# Texas     71000
# Oregon    16000
# Utah       5000
# dtype: int64
obj4
# California        NaN
# Ohio          35000.0
# Oregon        16000.0
# Texas         71000.0
# dtype: float64
obj3 + obj4
# California         NaN
# Ohio           70000.0
# Oregon         32000.0
# Texas         142000.0
# Utah               NaN
# dtype: float64
```

稍后将更详细地讨论数据对齐功能。如果你有数据库方面的经验，你可以将其视为类似于联接的操作。

Series 对象本身及其索引有一个 name 属性，可以用于指示对象本身及其索引的语义：

```python
obj4.name = "population"
obj4.index.name = "state"
obj4
# state
# California        NaN
# Ohio          35000.0
# Oregon        16000.0
# Texas         71000.0
# Name: population, dtype: float64
```

Series 的索引可以通过赋值来改变：

```python
obj
# 0    4
# 1    7
# 2   -5
# 3    3
# dtype: int64
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]
obj
# Bob      4
# Steve    7
# Jeff    -5
# Ryan     3
# dtype: int64
```

#### DataFrame

DataFrame 表示一个矩形数据表，并包含有序、命名的列集合，每个列可以是不同的值类型（数字、字符串、布尔值等）。 DataFrame 同时具有行索引和列索引；它可以被认为是共享同一个索引的 Series 的字典。

> 虽然 DataFrame 物理上是二维的，但可以使用分层索引以表格格式表示更高维的数据，我们将在第 8 章：数据整理：连接、组合和重塑中讨论该主题，也是 pandas 中一些更高级的数据处理功能的组成部分。
{: .promp-info }

构造 DataFrame 的方法有很多，最常见的方法之一是使用等长列表或 NumPy 数组的字典：

```python
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
```

与 Series 一样，生成的 DataFrame 将自动分配索引，并且根据数据中键的顺序放置列（即它们在字典中的插入顺序）：

```python
frame
#     state  year  pop
# 0    Ohio  2000  1.5
# 1    Ohio  2001  1.7
# 2    Ohio  2002  3.6
# 3  Nevada  2001  2.4
# 4  Nevada  2002  2.9
# 5  Nevada  2003  3.2
```

对于大型 DataFrame，可以用 `head` 方法或 `tail` 方法仅选择前五行或后五行：

```python
frame.head()
#     state  year  pop
# 0    Ohio  2000  1.5
# 1    Ohio  2001  1.7
# 2    Ohio  2002  3.6
# 3  Nevada  2001  2.4
# 4  Nevada  2002  2.9
frame.tail()
#     state  year  pop
# 1    Ohio  2001  1.7
# 2    Ohio  2002  3.6
# 3  Nevada  2001  2.4
# 4  Nevada  2002  2.9
# 5  Nevada  2003  3.2
```

如果创建时指定了列序列，DataFrame 的列将按该顺序排列：

```python
pd.DataFrame(data, columns=["year", "state", "pop"])
#    year   state  pop
# 0  2000    Ohio  1.5
# 1  2001    Ohio  1.7
# 2  2002    Ohio  3.6
# 3  2001  Nevada  2.4
# 4  2002  Nevada  2.9
# 5  2003  Nevada  3.2
```

如果传递了字典中未包含的列，则结果中将显示缺失值：

```python
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
frame2
#    year   state  pop debt
# 0  2000    Ohio  1.5  NaN
# 1  2001    Ohio  1.7  NaN
# 2  2002    Ohio  3.6  NaN
# 3  2001  Nevada  2.4  NaN
# 4  2002  Nevada  2.9  NaN
# 5  2003  Nevada  3.2  NaN
frame2.columns
# Index(['year', 'state', 'pop', 'debt'], dtype='object')
```

DataFrame 中的列可以通过类似字典的表示法或使用点属性表示法作为 Series 进行检索：

```python
frame2["state"]
# 0      Ohio
# 1      Ohio
# 2      Ohio
# 3    Nevada
# 4    Nevada
# 5    Nevada
# Name: state, dtype: object
frame2.year
# 0    2000
# 1    2001
# 2    2002
# 3    2001
# 4    2002
# 5    2003
# Name: year, dtype: int64
```

> 提示
> 
> `frame2[column]` 适用于任何列名称，但 `frame2.column` 仅当列名称是有效的 Python 变量名称并且不与 DataFrame 中的任何方法名称冲突时才有效。例如，如果列的名称包含空格或下划线以外的符号，则无法使用点属性方法访问它。
>
> 此外点属性表示法还有其他限制，所以通常建议使用字典表示法。
{: .prompt-tip }

请注意，返回的 Series 与 DataFrame 具有相同的索引，并且它们的 `name` 属性已适当设置。

还可以使用特殊的 `iloc` 和 `loc` 属性按位置或名称检索行（稍后将在使用 `loc` 和 `iloc` 在 DataFrame 上进行选择中详细介绍）：

```python
frame2.loc[1]
# year     2001
# state    Ohio
# pop       1.7
# debt      NaN
# Name: 1, dtype: object
frame2.iloc[2]
# year     2002
# state    Ohio
# pop       3.6
# debt      NaN
# Name: 2, dtype: object
```

可以通过赋值来修改列。例如，可以为空的 `debt` 列分配标量值或值数组：

```python
frame2["debt"] = 16.5
#    year   state  pop  debt
# 0  2000    Ohio  1.5  16.5
# 1  2001    Ohio  1.7  16.5
# 2  2002    Ohio  3.6  16.5
# 3  2001  Nevada  2.4  16.5
# 4  2002  Nevada  2.9  16.5
# 5  2003  Nevada  3.2  16.5
frame2["debt"] = np.arange(6.)
#    year   state  pop  debt
# 0  2000    Ohio  1.5   0.0
# 1  2001    Ohio  1.7   1.0
# 2  2002    Ohio  3.6   2.0
# 3  2001  Nevada  2.4   3.0
# 4  2002  Nevada  2.9   4.0
# 5  2003  Nevada  3.2   5.0
```

将列表或数组分配给列时，值的长度必须与 DataFrame 的长度匹配。

如果不想改变原表，可以使用 `copy` 方法修改副本。

如果你分配了一个 Series，它的标签将完全与 DataFrame 的索引重新对齐，在任何不存在的索引值中插入缺失值：

```python
val = pd.Series([-1.2, -1.5, -1.7], index=[2, 4, 5])
frame2["debt"] = val
frame2
#    year   state  pop  debt
# 0  2000    Ohio  1.5   NaN
# 1  2001    Ohio  1.7   NaN
# 2  2002    Ohio  3.6  -1.2
# 3  2001  Nevada  2.4   NaN
# 4  2002  Nevada  2.9  -1.5
# 5  2003  Nevada  3.2  -1.7
```

分配不存在的列将创建一个新列。

`del` 关键字将像字典一样删除列。作为示例，首先添加一个新的布尔值列，值是判断 `state` 列是否等于 `“Ohio”`：

```python
frame2["eastern"] = frame2["state"] == "Ohio"
frame2
#    year   state  pop  debt  eastern
# 0  2000    Ohio  1.5   NaN     True
# 1  2001    Ohio  1.7   NaN     True
# 2  2002    Ohio  3.6  -1.2     True
# 3  2001  Nevada  2.4   NaN    False
# 4  2002  Nevada  2.9  -1.5    False
# 5  2003  Nevada  3.2  -1.7    False
del frame2["eastern"]
frame2.columns
# Index(['year', 'state', 'pop', 'debt'], dtype='object')
```

> 提示
> 
> 无法使用 `frame2.eastern` 点属性表示法创建新列。
{: .prompt-tip  }

另一种常见的数据形式是字典的嵌套字典：

```python
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
               "Nevada": {2001: 2.4, 2002: 2.9}}
```

如果嵌套字典传递给 DataFrame，pandas 会将外部字典键解释为列，将内部键解释为行索引：

```python
frame3 = pd.DataFrame(populations)
frame3
#       Ohio  Nevada
# 2000   1.5     NaN
# 2001   1.7     2.4
# 2002   3.6     2.9
```

可以使用与 NumPy 数组类似的语法转置 DataFrame（交换行和列）：

```python
frame3.T
#         2000  2001  2002
# Ohio     1.5   1.7   3.6
# Nevada   NaN   2.4   2.9
```

> 警告
> 
> 如果原始表中各列的数据类型不一致（比如第一列是整数，第二列是字符串），那么转置操作会强制将所有数据转换成一种通用的类型，导致原始的精确类型丢失。
{: .prompt-warning }

可以在创建时显式指定索引，改变 pandas 对嵌套字典的默认解释方式：

```python
pd.DataFrame(populations, index=[2001, 2002, 2003])
#       Ohio  Nevada
# 2001   1.7     2.4
# 2002   3.6     2.9
# 2003   NaN     NaN
```

Series 词典的处理方式大致相同：

```python
pdata = {"Ohio": frame3["Ohio"][:-1],
         "Nevada": frame3["Nevada"][:2]}
pd.DataFrame(pdata)
#       Ohio  Nevada
# 2000   1.5     NaN
# 2001   1.7     2.4
```

DataFrame 构造函数的有效输入数据还有很多。

| Type                                                         | Notes                                                        |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 2D ndarray                                                   | 数据矩阵，可传入可选的行标签和列标签。                       |
| 由数组、列表或元组构成的字典 (dict of arrays, lists, or tuples) | 每个序列成为 DataFrame 的一列。所有的序列长度必须相同。      |
| NumPy 结构化/记录数组 (NumPy structured/record array)        | 处理方式与“由数组构成的字典”相同。                           |
| 由 Series 构成的字典 (dict of Series)                        | 每个 Series 成为一列。每个 Series 的索引会被合并（取并集）以形成结果的行索引。 |
| 由字典构成的字典 (dict of dicts)                             | 每个内部字典会成为一列。键会被合并（取并集）以形成行索引，处理方式与“由 Series 构成的字典”类似。 |
| 由字典或 Series 构成的列表 (List of dicts or Series)         | 列表中的每一项成为 DataFrame 的一行。字典键或 Series 索引的并集将成为 DataFrame 的列标签。 |
| 由列表或元组构成的列表 (List of lists or tuples)             | 处理方式与“2D ndarray”相同。                                 |
| 另一个 DataFrame                                             | 使用该 DataFrame 的索引，除非调用时指定了不同的索引。        |
| NumPy MaskedArray (掩码数组)                                 | 与“2D ndarray”类似，但被掩码（masked）的值在生成的 DataFrame 中会变成 NA/缺失值。 |

如果 DataFrame 的索引和列设置了名称属性，这些属性也会显示：

```python
frame3.index.name = "year"
frame3.columns.name = "state"
frame3
# state  Ohio  Nevada
# year               
# 2000    1.5     NaN
# 2001    1.7     2.4
# 2002    3.6     2.9
```

与 Series 不同，DataFrame 没有 `name` 属性。 

DataFrame 的 `to_numpy` 方法将 DataFrame 中包含的数据作为二维 ndarray 返回：

```python
frame3.to_numpy()
# [[1.5, nan],
#  [1.7, 2.4],
#  [3.6, 2.9]]
```

当 DataFrame 包含多种数据类型时，为了统一数据格式，将所有数据强制转换（Upcasting）为一种能够容纳所有数据的通用类型：

```python
frame2.to_numpy()
# array([[2000, 'Ohio', 1.5, nan],
#        [2001, 'Ohio', 1.7, nan],
#        [2002, 'Ohio', 3.6, -1.2],
#        [2001, 'Nevada', 2.4, nan],
#        [2002, 'Nevada', 2.9, -1.5],
#        [2003, 'Nevada', 3.2, -1.7]], dtype=object)
```

#### IndexObjects

pandas 的 Index 对象负责保存轴标签（包括 DataFrame 的列名称）和其他元数据（如轴名称）。构建 Series 或 DataFrame 时使用的任何数组或其他标签序列都会在内部转换为索引：

```python
obj = pd.Series(np.arange(3), index=["a", "b", "c"])
index = obj.index
index
# Index(['a', 'b', 'c'], dtype='object')
index[1:]
# Index(['b', 'c'], dtype='object')
```

索引对象是不可变的，因此用户不能修改：

```python
index[1] = "d"  # TypeError
```

不变性使得在数据结构之间共享 Index 对象更加安全

```python
labels = pd.Index(np.arange(3))
labels
# Index([0, 1, 2], dtype='int64')
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
# 0    1.5
# 1   -2.5
# 2    0.0
# dtype: float64
obj2.index is labels
# True
```

有些用户不会经常利用索引提供的功能，但由于某些操作会产生包含索引数据的结果，因此了解它们的工作原理非常重要。

除了类似于数组之外，索引的行为也类似于固定大小的集合：

```python
frame3
# state  Ohio  Nevada
# year               
# 2000    1.5     NaN
# 2001    1.7     2.4
# 2002    3.6     2.9
frame3.columns
# Index(['Ohio', 'Nevada'], dtype='object', name='state')
"Ohio" in frame3.columns
# True
2003 in frame3.index
# False
```

与 Python 集合不同，pandas 索引可以包含重复标签：

```python
pd.Index(["foo", "foo", "bar", "bar"])
# Index(['foo', 'foo', 'bar', 'bar'], dtype='object')
```

具有重复标签的选择将选择该标签的所有出现位置。

每个索引都有许多用于集合逻辑的方法和属性，它们回答了有关其包含的数据的其他常见问题。下表总结了一些有用的内容。

| Method         | Description                                          |
| :------------- | :--------------------------------------------------- |
| `append`       | 将额外的 Index 对象连接起来，产生一个新的 Index。    |
| `difference`   | 计算两个 Index 对象的差集（set difference）。        |
| `intersection` | 计算交集（set intersection）。                       |
| `union`        | 计算并集（set union）。                              |
| `isin`         | 计算一个布尔数组，表示每个值是否包含在传入的集合中。 |
| `delete`       | 计算一个新的 Index，其中索引 `i` 处的元素已被删除。  |
| `drop`         | 计算一个新的 Index，其中指定的值已被删除。           |
| `insert`       | 计算一个新的 Index，其中在索引 `i` 处插入了元素。    |
| `is_monotonic` | 如果每个元素都大于等于前一个元素，则返回 True。      |
| `is_unique`    | 如果 Index 中没有重复值，则返回 True。               |
| `unique`       | 计算 Index 中唯一值的数组。                          |

### 5.2 基本功能

本节将介绍与 Series 或 DataFrame 中包含的数据进行交互的基本机制和常用功能。

#### 重索引

pandas 对象的一个重要方法是 `reindex`，这意味着创建一个新对象，并重新排列值以与新索引对齐。例如：

```python
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj
# d    4.5
# b    7.2
# a   -5.3
# c    3.6
# dtype: float64
obj2 = obj.reindex(["a", "b", "c", "d", "e"]) # 根据新索引重新排列数据，如果尚不存在任何索引值，则会引入缺失值
obj2
# a   -5.3
# b    7.2
# c    3.6
# d    4.5
# e    NaN
# dtype: float64
```

对于时间序列等有序数据，可能需要在重索引时进行一些插值或填充。`method` 选项允许我们使用诸如 `ffill` 之类的方法来执行此操作，该方法向前填充（forward-fill）值：

```python
obj3 = pd.Series(["blue", "purple", "yellow"], index=[0, 2, 4])
obj3
# 0      blue
# 2    purple
# 4    yellow
# dtype: object
obj3.reindex(np.arange(6), method="ffill")
# 0      blue
# 1      blue
# 2    purple
# 3    purple
# 4    yellow
# 5    yellow
# dtype: object
```

使用 DataFrame，重索引可以更改（行）索引、列或两者。当仅传递一个序列时，它会重索引行：

```python
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=["a", "c", "d"],
                     columns=["Ohio", "Texas", "California"])
frame
#    Ohio  Texas  California
# a     0      1           2
# c     3      4           5
# d     6      7           8
frame2 = frame.reindex(index=["a", "b", "c", "d"])
frame2
#    Ohio  Texas  California
# a   0.0    1.0         2.0
# b   NaN    NaN         NaN
# c   3.0    4.0         5.0
# d   6.0    7.0         8.0
```

可以使用 `columns` 关键字对列重新索引：

```python
states = ["Texas", "Utah", "California"]
frame.reindex(columns=states)
#    Texas  Utah  California
# a      1   NaN           2
# c      4   NaN           5
# d      7   NaN           8
```

由于 `"Ohio"` 不在 `states` 内，因此该列的数据将从结果中删除。

重索引特定轴的另一种方法是将新轴标签作为位置参数传递，然后使用 `axis` 关键字指定要重新索引的轴：

```python
frame.reindex(states, axis="columns")
#    Texas  Utah  California
# a      1   NaN           2
# c      4   NaN           5
# d      7   NaN           8
```

有关重索引参数的更多信息，请参阅下表。

| Argument     | Description                                                  |
| :----------- | :----------------------------------------------------------- |
| `index`      | 用作索引的新序列。可以是 Index 实例或其他任何类似序列的 Python 数据结构。索引会被原样使用。 |
| `method`     | 插值（填充）方法；`'ffill'` 为前向填充，`'bfill'` 为后向填充。 |
| `fill_value` | 当引入缺失数据时使用的替代值。                               |
| `limit`      | 当进行前向或后向填充时，允许填充的最大间隙大小（以元素数量计）。 |
| `tolerance`  | 当进行前向或后向填充时，对于非精确匹配，允许填充的最大间隙（以绝对距离计）。 |
| `level`      | 在 MultiIndex 的指定级别（level）上匹配简单 Index；或者选择子集。 |
| `copy`       | 默认为 True，即如果新索引与旧索引相同，也会复制底层数据；如果为 False，当索引相同时则不复制数据。 |

还可以使用 `loc` 运算符（Label-based location 基于标签的定位）重索引，并且许多用户更喜欢始终这样做。仅当所有新索引标签已存在于 DataFrame 中时，此方法才有效（不同于 `reindex` 为新标签插入缺失的数据）：

```python
frame.loc[["a", "d", "c"], ["California", "Texas"]]
#    California  Texas
# a           2      1
# d           8      7
# c           5      4
```

#### 从轴上删除条目

虽然通过 `reindex` 方法或基于 `.loc` 的索引操作来删除轴上的一个或多个条目是很简单的。但由于这种方式可能需要进行一些数据整理和集合逻辑运算（例如计算差集），`drop` 方法提供了一种更直接的选择：它会返回一个新对象，并将指定标签的值从对应的轴上删除。

```python
obj = pd.Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
obj
# a    0.0
# b    1.0
# c    2.0
# d    3.0
# e    4.0
# dtype: float64
new_obj = obj.drop("c")
new_obj
# a    0.0
# b    1.0
# d    3.0
# e    4.0
# dtype: float64
obj.drop(["d", "c"])
# a    0.0
# b    1.0
# e    4.0
# dtype: float64
```

使用 DataFrame，可以从任一轴删除索引值。为了说明这一点，我们首先创建一个示例 DataFrame：

```python
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
data
#           one  two  three  four
# Ohio        0    1      2     3
# Colorado    4    5      6     7
# Utah        8    9     10    11
# New York   12   13     14    15
data.drop(index=["Colorado", "Ohio"])
#           one  two  three  four
# Utah        8    9     10    11
# New York   12   13     14    15
data.drop(columns=["two"]) 或 data.drop("two", axis=1)
#           one  three  four
# Ohio        0      2     3
# Colorado    4      6     7
# Utah        8     10    11
# New York   12     14    15
data.drop(["two", "four"], axis="columns")
#           one  three
# Ohio        0      2
# Colorado    4      6
# Utah        8     10
# New York   12     14
```

#### 索引、选择和过滤

Series 索引 (`obj[...]`) 的工作方式与 NumPy 数组索引类似，只不过你可以使用 Series 的索引值而不仅仅是整数。以下是一些例子：

```python
obj = pd.Series(np.arange(4.), index=["a", "b", "c", "d"])
obj
# a    0.0
# b    1.0
# c    2.0
# d    3.0
# dtype: float64
obj["b"]
# 1.0
obj[1]
# 1.0
obj[2:4]
# c    2.0
# d    3.0
# dtype: float64
obj[["b", "a", "d"]]
# b    1.0
# a    0.0
# d    3.0
# dtype: float64
obj[[1, 3]]
# b    1.0
# d    3.0
# dtype: float64
obj[obj < 2]
# a    0.0
# b    1.0
# dtype: float64
```

虽然可以通过这种方式使用标签选择数据，但选择索引值的首选方法是使用特殊的 `loc` 运算符：

```python
obj.loc[["b", "a", "d"]]
# b    1.0
# a    0.0
# d    3.0
# dtype: float64
```

优先 `loc` 的原因是因为用 `[]` 索引时对整数的处理是不一致的。只有在索引包含整数时，基于常规 `[]` 的索引才会将整数视为标签。

例如：

```python
obj1 = pd.Series([1, 2, 3], index=[2, 0, 1])
obj2 = pd.Series([1, 2, 3], index=["a", "b", "c"])
obj1
# 2    1
# 0    2
# 1    3
# dtype: int64
obj2
# a    1
# b    2
# c    3
# dtype: int64
obj1[[0, 1, 2]]
# 0    2
# 1    3
# 2    1
# dtype: int64
obj2[[0, 1, 2]]
# a    1
# b    2
# c    3
# dtype: int64
```

使用 `loc` 时，当索引不包含整数时，表达式 `obj.loc[[0, 1, 2]]` 将失败：

```python
obj2.loc[[0, 1]]
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_804589/4185657903.py in <module>
----> 1 obj2.loc[[0, 1]]

^ LONG EXCEPTION ABBREVIATED ^

KeyError: "None of [Int64Index([0, 1], dtype="int64")] are in the [index]"
```

由于 `loc` 运算符仅使用标签进行索引，因此还有一个 `iloc` （Integer-based Location）运算符仅使用整数进行索引，无论索引是否包含整数，都可以一致地工作：

```python
obj1.iloc[[0, 1, 2]]
# 2    1
# 0    2
# 1    3
# dtype: int64
obj2.iloc[[0, 1, 2]]
# a    1
# b    2
# c    3
# dtype: int64
```

您还可以使用标签进行切片，但它的工作方式与普通 Python 切片不同，因为端点是包含在内的，也就是闭区间：

```python
obj2.loc["b":"c"]
# b    2
# c    3
# dtype: int64
```

使用这些方法赋值会修改 Series 的相应部分：

```python
obj2.loc["b":"c"] = 5 # 注意 loc 后面是方括号，这个不是方法调用，不能用圆括号
obj2
# a    1
# b    5
# c    5
# dtype: int64
```

对 DataFrame 进行索引会检索具有单个值或序列的一个或多个列：

```python
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
data
#           one  two  three  four
# Ohio        0    1      2     3
# Colorado    4    5      6     7
# Utah        8    9     10    11
# New York   12   13     14    15

使用 [] 索引
---
data["two"]
# Ohio         1
# Colorado     5
# Utah         9
# New York    13
# Name: two, dtype: int64
data[["three", "one"]]
#           three  one
# Ohio          2    0
# Colorado      6    4
# Utah         10    8
# New York     14   12
data[:2]
#           one  two  three  four
# Ohio        0    1      2     3
# Colorado    4    5      6     7
data[data["three"] > 5]
#           one  two  three  four
# Colorado    4    5      6     7
# Utah        8    9     10    11
# New York   12   13     14    15
data < 5
#             one    two  three   four
# Ohio       True   True   True   True
# Colorado   True  False  False  False
# Utah      False  False  False  False
# New York  False  False  False  False
data[data < 5] = 0
#           one  two  three  four
# Ohio        0    0      0     0
# Colorado    0    5      6     7
# Utah        8    9     10    11
# New York   12   13     14    15

使用 .loc 和 .iloc 索引
---
data.loc["Colorado"]
# one      0
# two      5
# three    6
# four     7
# Name: Colorado, dtype: int64
data.loc[["Colorado", "New York"]]
#           one  two  three  four
# Colorado    0    5      6     7
# New York   12   13     14    15
data.loc["Colorado", ["two", "three"]]
# two      5
# three    6
# Name: Colorado, dtype: int64
data.loc[:"Utah", "two"]
# Ohio        0
# Colorado    5
# Utah        9
# Name: two, dtype: int64
data.iloc[2]
# one       8
# two       9
# three    10
# four     11
# Name: Utah, dtype: int64
data.iloc[[2, 1]]
#           one  two  three  four
# Utah        8    9     10    11
# Colorado    0    5      6     7
data.iloc[2, [3, 0, 1]]
# four    11
# one      8
# two      9
# Name: Utah, dtype: int64
data.iloc[[1, 2], [3, 0, 1]]
#           four  one  two
# Colorado     7    0    5
# Utah        11    8    9
data.iloc[:, :3][data.three > 5]
#           one  two  three
# Colorado    0    5      6
# Utah        8    9     10
# New York   12   13     14

# 布尔数组可以与 loc 一起使用，但不能与 iloc 一起使用：
data.loc[data.three >= 2]
#           one  two  three  four
# Colorado    0    5      6     7
# Utah        8    9     10    11
# New York   12   13     14    15
```

有多种方法可以选择和重排列 pandas 对象中包含的数据。对于 DataFrame，下表提供了其中许多内容的简短摘要。

| Type                        | Notes                                                        |
| :-------------------------- | :----------------------------------------------------------- |
| `df[val]`                   | 从 DataFrame 中选取单列或列序列；特殊情况便利：布尔数组（过滤行）、切片（切片行）或布尔 DataFrame（根据某些标准设置值）。 |
| `df.loc[val]`               | 通过标签选取 DataFrame 的单行或行子集。                      |
| `df.loc[:, val]`            | 通过标签选取单列或列子集。                                   |
| `df.loc[val1, val2]`        | 通过标签同时选取行和列。                                     |
| `df.iloc[where]`            | 通过整数位置选取 DataFrame 的单行或行子集。                  |
| `df.iloc[:, where]`         | 通过整数位置选取单列或列子集。                               |
| `df.iloc[where_i, where_j]` | 通过整数位置同时选取行和列。                                 |
| `df.at[label_i, label_j]`   | 通过行和列标签选取单个标量值。                               |
| `df.iat[i, j]`              | 通过行和列位置（整数）选取单个标量值。                       |
| `reindex`                   | 通过标签选择行或列（将一个或多个轴匹配到新索引）。           |
| `get_value, set_value`      | （已弃用）通过行和列标签选取单个值。                         |

使用由整数索引的 pandas 对象可能会成为新用户的绊脚石，因为它们的工作方式与内置的 Python 数据结构（如列表和元组）不同。例如，您可能不希望以下代码生成错误：

```python
ser = pd.Series(np.arange(3.))
ser
# 0    0.0
# 1    1.0
# 2    2.0
# dtype: float64
ser[-1]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
~/miniforge-x86/envs/book-env/lib/python3.10/site-packages/pandas/core/indexes/ra
nge.py in get_loc(self, key)
    344             try:
--> 345                 return self._range.index(new_key)
    346             except ValueError as err:
ValueError: -1 is not in range
The above exception was the direct cause of the following exception:
KeyError                                  Traceback (most recent call last)
<ipython-input-166-44969a759c20> in <module>
----> 1 ser[-1]
~/miniforge-x86/envs/book-env/lib/python3.10/site-packages/pandas/core/series.py 
in __getitem__(self, key)
   1010 
   1011         elif key_is_scalar:
-> 1012             return self._get_value(key)
   1013 
   1014         if is_hashable(key):
~/miniforge-x86/envs/book-env/lib/python3.10/site-packages/pandas/core/series.py 
in _get_value(self, label, takeable)
   1119 
   1120         # Similar to Index.get_value, but we do not fall back to position
al
-> 1121         loc = self.index.get_loc(label)
   1122 
   1123         if is_integer(loc):
~/miniforge-x86/envs/book-env/lib/python3.10/site-packages/pandas/core/indexes/ra
nge.py in get_loc(self, key)
    345                 return self._range.index(new_key)
    346             except ValueError as err:
--> 347                 raise KeyError(key) from err
    348         self._check_indexing_error(key)
    349         raise KeyError(key)
KeyError: -1
```

在这种情况下，pandas 可以 “退回” 整数索引，但通常很难在不向用户代码中引入细微错误的情况下做到这一点。假如我们有一个包含 0、1 和 2 的索引，但 pandas 不会猜测用户想要什么（到底是基于标签的索引还是基于位置的索引）：

```python
ser
# 0    0.0
# 1    1.0
# 2    2.0
# dtype: float64
```

另一方面，对于非整数索引，就不存在这样的歧义：

```python
ser2 = pd.Series(np.arange(3.), index=["a", "b", "c"])
ser2[-1]
# 2.0
```

另一方面，整数切片始终是面向整数的：

```python
ser[:2]
# 0    0.0
# 1    1.0
# dtype: float64
```

由于这些陷阱，最好始终首选使用 `loc` 和 `iloc` 进行索引以避免歧义。

可以通过标签或整数位置给列或行赋值：

```python
data.loc[:, "one"] = 1
data
#           one  two  three  four
# Ohio        1    0      0     0
# Colorado    1    5      6     7
# Utah        1    9     10    11
# New York    1   13     14    15
data.iloc[2] = 5
data
#           one  two  three  four
# Ohio        1    0      0     0
# Colorado    1    5      6     7
# Utah        5    5      5     5
# New York    1   13     14    15
data.loc[data["four"] > 5] = 3
data
#           one  two  three  four
# Ohio        1    0      0     0
# Colorado    3    3      3     3
# Utah        5    5      5     5
# New York    3    3      3     3
```

对于新的 pandas 用户来说，一个常见的问题是在链式选择时赋值，如下所示：

```python
data.loc[data.three == 5]["three"] = 6
---------------------------------------------------------------------------
<ipython-input-11-0ed1cf2155d5>:1: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
```

根据数据内容，这可能会打印一个特殊的 `SettingWithCopyWarning`，它警告你正在尝试修改临时值（`data.loc[data. Three == 5] `的非空结果）而不是原始 DataFrame 数据。可以看到，原始数据未修改：

```python
data
#           one  two  three  four
# Ohio        1    0      0     0
# Colorado    3    3      3     3
# Utah        5    5      5     5
# New York    3    3      3     3
```

在这些情况下，修复方法是把链式赋值重写成使用单个 `loc` 操作：

```python
data.loc[data.three == 5, "three"] = 6
#           one  two  three  four
# Ohio        1    0      0     0
# Colorado    3    3      3     3
# Utah        5    5      6     5
# New York    3    3      3     3
```

一个好的经验法则是在进行赋值时避免链式索引。

#### 算术运算和数据对齐

pandas 可以使处理具有不同索引的对象变得更加简单。例如，当你添加对象时，如果任何索引对不相同，则结果中的相应索引将是索引对的并集。让我们看一个例子：

```python
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=["a", "c", "d", "e"])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=["a", "c", "e", "f", "g"])
s1 + s2
# a    5.2
# c    1.1
# d    NaN
# e    0.0
# f    NaN
# g    NaN
# dtype: float64
```

内部数据对齐在不匹配的标签位置引入了缺失值。缺失值将在进一步的算术计算中传播。

对于 DataFrame，对齐是在行和列上执行的：

```python
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list("bcd"),
                   index=["Ohio", "Texas", "Colorado"])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"),
                   index=["Utah", "Ohio", "Texas", "Oregon"])
df1
#             b    c    d
# Ohio      0.0  1.0  2.0
# Texas     3.0  4.0  5.0
# Colorado  6.0  7.0  8.0
df2
#           b     d     e
# Utah    0.0   1.0   2.0
# Ohio    3.0   4.0   5.0
# Texas   6.0   7.0   8.0
# Oregon  9.0  10.0  11.0
df1 + df2
#             b   c     d   e
# Colorado  NaN NaN   NaN NaN
# Ohio      3.0 NaN   6.0 NaN
# Oregon    NaN NaN   NaN NaN
# Texas     9.0 NaN  12.0 NaN
# Utah      NaN NaN   NaN NaN
```

由于在两个 DataFrame 对象中均未找到 `“c”` 和 `“e”` 列，因此它们在结果中显示为缺失，行也是如此。

如果添加没有共同列或行标签的 DataFrame 对象，结果将包含所有空值：

```python
df1 = pd.DataFrame({"A": [1, 2]})
df2 = pd.DataFrame({"B": [3, 4]})
df1 + df2
#     A   B
# 0 NaN NaN
# 1 NaN NaN
```

在不同索引对象之间的算术运算中，当在一个对象中找到轴标签但在另一个对象中没有找到时，此时你可能需要填充特殊值，例如 0。

下面是一个示例，通过将 `np.nan` 分配给它来将特定值设置为 NA (null)：

```python
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list("abcde"))
df2.loc[1, "b"] = np.nan
df1
#      a    b     c     d
# 0  0.0  1.0   2.0   3.0
# 1  4.0  5.0   6.0   7.0
# 2  8.0  9.0  10.0  11.0
df2
#       a     b     c     d     e
# 0   0.0   1.0   2.0   3.0   4.0
# 1   5.0   NaN   7.0   8.0   9.0
# 2  10.0  11.0  12.0  13.0  14.0
# 3  15.0  16.0  17.0  18.0  19.0
df1 + df2
#       a     b     c     d   e
# 0   0.0   2.0   4.0   6.0 NaN
# 1   9.0   NaN  13.0  15.0 NaN
# 2  18.0  20.0  22.0  24.0 NaN
# 3   NaN   NaN   NaN   NaN NaN
```

使用 `df1` 上的 add 方法，将 `df2` 和参数传递给 `fill_value`，这将在计算之前，先给缺失的一方垫一个底：

```python
df1.add(df2, fill_value=0)
#       a     b     c     d     e
# 0   0.0   2.0   4.0   6.0   4.0
# 1   9.0   5.0  13.0  15.0   9.0
# 2  18.0  20.0  22.0  24.0  14.0
# 3  15.0  16.0  17.0  18.0  19.0
```

很多用于算术运算的操作都有对应的 Series 和 DataFrame 方法和以子母 r 开头的反向运算方法。

```python
1 / df1
#        a         b         c         d
# 0    inf  1.000000  0.500000  0.333333
# 1  0.250  0.200000  0.166667  0.142857
# 2  0.125  0.111111  0.100000  0.090909
df1.rdiv(1)
#        a         b         c         d
# 0    inf  1.000000  0.500000  0.333333
# 1  0.250  0.200000  0.166667  0.142857
# 2  0.125  0.111111  0.100000  0.090909
```

| Method              | Description                                |
| :------------------ | :----------------------------------------- |
| `add`               | 用于加法 (`+`)                             |
| `sub`               | 用于减法 (`-`)                             |
| `div`               | 用于除法 (`/`)                             |
| `floordiv`          | 用于整除 (`//`)                            |
| `mul`               | 用于乘法 (`*`)                             |
| `pow`               | 用于幂运算 (`**`)                          |
| `radd`, `rsub`, ... | 反向版本的算术方法（例如，`other + df`）。 |

与不同维度的 NumPy 数组一样，DataFrame 和 Series 之间的算术运算也是可行的。

首先，作为一个示例，考虑二维数组与其行之一之间的差异：

```python
arr = np.arange(12.).reshape((3, 4))
arr
# array([[ 0.,  1.,  2.,  3.],
#        [ 4.,  5.,  6.,  7.],
#        [ 8.,  9., 10., 11.]])
arr[0]
# array([0., 1., 2., 3.])
arr - arr[0]
# array([[0., 0., 0., 0.],
#        [4., 4., 4., 4.],
#        [8., 8., 8., 8.]])
```

当我们从 `arr` 中减去 `arr[0]` 时，每行执行一次减法。这称为广播。 DataFrame 和 Series 之间的操作是类似的：

```python
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list("bde"),
                     index=["Utah", "Ohio", "Texas", "Oregon"])
series = frame.iloc[0]
frame
#           b     d     e
# Utah    0.0   1.0   2.0
# Ohio    3.0   4.0   5.0
# Texas   6.0   7.0   8.0
# Oregon  9.0  10.0  11.0
series
# b    0.0
# d    1.0
# e    2.0
# Name: Utah, dtype: float64
frame - series
#           b    d    e
# Utah    0.0  0.0  0.0
# Ohio    3.0  3.0  3.0
# Texas   6.0  6.0  6.0
# Oregon  9.0  9.0  9.0
```

如果在 DataFrame 的列或 Series 的索引中都找不到索引值，则对象将被重新索引以形成并集：

```python
series2 = pd.Series(np.arange(3), index=["b", "e", "f"])
series2
# b    0
# e    1
# f    2
# dtype: int64
frame + series2
#           b   d     e   f
# Utah    0.0 NaN   3.0 NaN
# Ohio    3.0 NaN   6.0 NaN
# Texas   6.0 NaN   9.0 NaN
# Oregon  9.0 NaN  12.0 NaN
```

如果您想在列上进行广播，在行上进行匹配，则必须使用一种算术方法并指定在索引上进行匹配。例如：

```python
series3 = frame["d"]
frame
#           b     d     e
# Utah    0.0   1.0   2.0
# Ohio    3.0   4.0   5.0
# Texas   6.0   7.0   8.0
# Oregon  9.0  10.0  11.0
series3
# Utah       1.0
# Ohio       4.0
# Texas      7.0
# Oregon    10.0
# Name: d, dtype: float64
frame.sub(series3, axis="index")
#           b    d    e
# Utah   -1.0  0.0  1.0
# Ohio   -1.0  0.0  1.0
# Texas  -1.0  0.0  1.0
# Oregon -1.0  0.0  1.0
```

用 `axis` 参数传递要匹配的轴。在本例中，匹配了 DataFrame 的行索引 (`axis="index"`) 并跨列广播。

#### 函数应用与映射

NumPy ufuncs（逐元素数组方法）也适用于 pandas 对象：

```python
frame = pd.DataFrame(np.random.standard_normal((4, 3)),
                     columns=list("bde"),
                     index=["Utah", "Ohio", "Texas", "Oregon"])
frame
#                b         d         e
# Utah   -0.204708  0.478943 -0.519439
# Ohio   -0.555730  1.965781  1.393406
# Texas   0.092908  0.281746  0.769023
# Oregon  1.246435  1.007189 -1.296221
np.abs(frame)
#                b         d         e
# Utah    0.204708  0.478943  0.519439
# Ohio    0.555730  1.965781  1.393406
# Texas   0.092908  0.281746  0.769023
# Oregon  1.246435  1.007189  1.296221
```

另一种常见的操作是将一维数组上的函数应用于每一列或行。 DataFrame 的 `apply` 方法正是这样做的：

```python
def f1(x):
    return x.max() - x.min()
frame.apply(f1)
# b    1.802165
# d    1.684034
# e    2.689627
# dtype: float64
```

函数 `f` 计算 Series 的最大值和最小值之间的差，在框架中的每一列上调用一次。结果是一个以 `frame` 列作为索引的 Series。

如果传递 `axis="columns"` 来应用，则该函数将每行调用一次。思考这个问题的一个有用方法是 “跨列应用”：

```python
frame.apply(f1, axis="columns")
# Utah      0.998382
# Ohio      2.521511
# Texas     0.676115
# Oregon    2.542656
# dtype: float64
```

许多最常见的数组统计数据（例如 `sum` 和 `mean`）都是 DataFrame 方法，因此没有必要使用 `apply`。

传递给 `apply` 的函数不仅可以返回标量值；还可以返回具有多个值的 Series：

```python
def f2(x):
    return pd.Series([x.min(), x.max()], index=["min", "max"])
frame.apply(f2)
#             b         d         e
# min -0.555730  0.281746 -1.296221
# max  1.246435  1.965781  1.393406
```

也可以使用逐元素 Python 函数。假设你想根据 `frame` 中的每个浮点值计算格式化字符串。可以使用 `applymap` 执行此操作：

```python
def my_format(x):
    return f"{x:.2f}"
frame.applymap(my_format)
#             b     d      e
# Utah    -0.20  0.48  -0.52
# Ohio    -0.56  1.97   1.39
# Texas    0.09  0.28   0.77
# Oregon   1.25  1.01  -1.30
```

名称 `applymap` 的原因是 Series 有一个用于应用逐元素函数的 `map` 方法：

```python
frame["e"].map(my_format)
# Utah      -0.52
# Ohio       1.39
# Texas      0.77
# Oregon    -1.30
# Name: e, dtype: object
```

#### 排序和排名

按某种标准对数据集进行排序是另一个重要的内置操作。要按行或列标签按字典顺序排序，请使用 `sort_index` 方法，该方法返回一个新的排序对象：

```python
obj = pd.Series(np.arange(4), index=["d", "a", "b", "c"])
obj
# d    0
# a    1
# b    2
# c    3
# dtype: int64
obj.sort_index()
# a    1
# b    2
# c    3
# d    0
# dtype: int64
```

使用 DataFrame，可以按任一轴上的索引进行排序：

```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=["three", "one"],
                     columns=["d", "a", "b", "c"])
frame
#        d  a  b  c
# three  0  1  2  3
# one    4  5  6  7
frame.sort_index()
#        d  a  b  c
# one    4  5  6  7
# three  0  1  2  3
frame.sort_index(axis="columns")
#        a  b  c  d
# three  1  2  3  0
# one    5  6  7  4
```

数据默认按升序排序，但也可以按降序排序：

```python
frame.sort_index(axis="columns", ascending=False)
#        d  c  b  a
# three  0  3  2  1
# one    4  7  6  5
```

要按 Series 值对系列进行排序，请使用其 `sort_values` 方法：

```python
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
# 2   -3
# 3    2
# 0    4
# 1    7
# dtype: int64
```

默认情况下，所有缺失值都会排序到 Series 末尾：

```python
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
# 4   -3.0
# 5    2.0
# 0    4.0
# 2    7.0
# 1    NaN
# 3    NaN
# dtype: float64
```

可以使用 `na_position` 选项将缺失值排序到开头：

```python
obj.sort_values(na_position="first")
# 1    NaN
# 3    NaN
# 4   -3.0
# 5    2.0
# 0    4.0
# 2    7.0
# dtype: float64
```

对 DataFrame 进行排序时，可以使用一列或多列中的数据作为排序键。为此，请将一个或多个列名称传递给 `sort_values`：

```python
frame = pd.DataFrame({"b": [4, 7, -3, 2], "a": [0, 1, 0, 1]})
frame
#    b  a
# 0  4  0
# 1  7  1
# 2 -3  0
# 3  2  1
frame.sort_values("b")
#    b  a
# 2 -3  0
# 3  2  1
# 0  4  0
# 1  7  1
frame.sort_values(["a", "b"])
#    b  a
# 2 -3  0
# 0  4  0
# 3  2  1
# 1  7  1
```

排名从最低值开始，从 1 到数组中有效数据点的数量分配排名。Series 和 DataFrame 的排名方法是值得关注的地方；默认情况下，排名通过为每个组分配平均排名来打破平局：

```python
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
# 0    6.5 # 两个 7 占据了第 6 第 7 名，这里分配均值 6.5 作为最终排名
# 1    1.0
# 2    6.5
# 3    4.5
# 4    3.0
# 5    2.0
# 6    4.5
# dtype: float64
obj.rank(method="first")
# 0    6.0 # 两个 7 中先出现的是第 6，后出现的是第 7
# 1    1.0
# 2    7.0
# 3    4.0
# 4    3.0
# 5    2.0
# 6    5.0
# dtype: float64
obj.rank(ascending=False)
# 0    1.5
# 1    7.0
# 2    1.5
# 3    3.5
# 4    5.0
# 5    6.0
# 6    3.5
# dtype: float64
```

DataFrame 也可以计算行或列的排名：

```python
frame = pd.DataFrame({"b": [4.3, 7, -3, 2], "a": [0, 1, 0, 1],
                      "c": [-2, 5, 8, -2.5]})
frame
#      b  a    c
# 0  4.3  0 -2.0
# 1  7.0  1  5.0
# 2 -3.0  0  8.0
# 3  2.0  1 -2.5
frame.rank(axis="columns")
#      b    a    c
# 0  3.0  2.0  1.0
# 1  3.0  1.0  2.0
# 2  1.0  2.0  3.0
# 3  3.0  2.0  1.0
```

排名时打破平级的方法如下表。

| 方法 (Method) | 说明 (Description)                                           |
| :------------ | :----------------------------------------------------------- |
| `'average'`   | 默认值：为相等分组中的每个条目分配平均排名。                 |
| `'min'`       | 对整个分组使用最小排名。                                     |
| `'max'`       | 对整个分组使用最大排名。                                     |
| `'first'`     | 按照值在数据中出现的顺序分配排名。                           |
| `'dense'`     | 类似于 `'min'`，但组与组之间的排名总是增加 1，而不是增加相等元素的数量。 |

#### 具有重复标签的轴索引

到目前为止，我们看过的几乎所有示例都有唯一的轴标签（索引值）。虽然许多 pandas 函数（如 `reindex`）要求标签是唯一的，但这不是强制性的。让我们考虑一个具有重复索引的小 Series：

```python
obj = pd.Series(np.arange(5), index=["a", "a", "b", "b", "c"])
obj
# a    0
# a    1
# b    2
# b    3
# c    4
# dtype: int64
obj.index.is_unique
# False
```

有重复项出现时，数据选择的行为会变得不一样。对具有多个条目的标签进行索引会返回一个 Series，而单个条目则返回一个标量值：

```python
obj["a"]
# a    0
# a    1
# dtype: int64
obj["c"]
# 4
```

这可能会使您的代码更加复杂，因为索引的输出类型可能会根据标签是否重复而有所不同。

相同的逻辑扩展到 DataFrame 中的索引行（或列）：

```python
df = pd.DataFrame(np.random.standard_normal((5, 3)),
                  index=["a", "a", "b", "b", "c"])
df
#           0         1         2
# a  0.274992  0.228913  1.352917
# a  0.886429 -2.001637 -0.371843
# b  1.669025 -0.438570 -0.539741
# b  0.476985  3.248944 -1.021228
# c -0.577087  0.124121  0.302614
df.loc["b"]
#           0         1         2
# b  1.669025 -0.438570 -0.539741
# b  0.476985  3.248944 -1.021228
df.loc["c"]
# 0   -0.577087
# 1    0.124121
# 2    0.302614
# Name: c, dtype: float64
```

### 5.3 总结和计算描述性统计

pandas 对象配备了一套常见的数学和统计方法。其中大多数属于归约或汇总统计的类别，从 Series 中提取单个值（如总和或平均值）的方法，或从 DataFrame 的行或列中提取一系列值的方法。与 NumPy 数组上的类似方法相比，它们具有针对丢失数据的内置处理。考虑一个小的 DataFrame：

```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=["a", "b", "c", "d"],
                  columns=["one", "two"])
df
#     one  two
# a  1.40  NaN
# b  7.10 -4.5
# c   NaN  NaN
# d  0.75 -1.3
df.sum()
# one    9.25
# two   -5.80
# dtype: float64
df.sum(axis="columns")
# a    1.40
# b    2.60
# c    0.00
# d   -0.55
# dtype: float64

通过控制 skipna 标志，决定对 NA 值的处理，是直接跳过还是传播 NA
---
df.sum(axis="index", skipna=False)
# one   NaN
# two   NaN
# dtype: float64
df.sum(axis="columns", skipna=False)
# a     NaN
# b    2.60
# c     NaN
# d   -0.55
# dtype: float64
```

一些聚合（例如平均值）需要至少一个非 NA 值才能产生值结果，因此这里我们有：

```python
df.mean(axis="columns")
# a    1.400
# b    1.300
# c      NaN
# d   -0.275
# dtype: float64
```

别的参数还包括

| Method   | Description                                      |
| :------- | :----------------------------------------------- |
| `axis`   | 要计算的轴                                       |
| `skipna` | 是否跳过缺失值。默认是 `True`                    |
| `level`  | 在指定层级计算，如果轴是分层索引的（MultiIndex） |

某些方法（例如 `idxmin` 和 `idxmax`）返回间接统计信息，例如达到最小值或最大值的索引值：

```python
df.idxmax()
# one    b
# two    d
# dtype: object
df.cumsum()
#     one  two
# a  1.40  NaN
# b  8.50 -4.5
# c   NaN  NaN
# d  9.25 -5.8
```

有些方法既不是减法，也不是累加。`describe` 就是这样一个例子，一次性生成多个汇总统计数据：

```python
df.describe()
#             one       two
# count  3.000000  2.000000
# mean   3.083333 -2.900000
# std    3.493685  2.262742
# min    0.750000 -4.500000
# 25%    1.075000 -3.700000
# 50%    1.400000 -2.900000
# 75%    4.250000 -2.100000
# max    7.100000 -1.300000
```

对于非数字数据，`describe` 会生成替代的汇总统计数据：

```python
obj = pd.Series(["a", "a", "b", "c"] * 4)
obj.describe()
# count     16
# unique     3
# top        a
# freq       8
# dtype: object
```

有关汇总统计数据和相关方法的完整列表，请参阅下表。

| Method             | Description                                             |
| :----------------- | :------------------------------------------------------ |
| `count`            | 非 NA（非缺失）值的数量。                               |
| `describe`         | 计算 Series 或 DataFrame 各列的汇总统计信息集合。       |
| `min`, `max`       | 计算最小值和最大值。                                    |
| `argmin`, `argmax` | 计算最小值或最大值所在的整数索引位置（integers）。      |
| `idxmin`, `idxmax` | 计算最小值或最大值所在的索引标签（labels）。            |
| `quantile`         | 计算样本的分位数（0 到 1 之间）。                       |
| `sum`              | 值的总和。                                              |
| `mean`             | 值的平均数。                                            |
| `median`           | 值的算术中位数（50% 分位数）。                          |
| `mad`              | 根据平均值计算平均绝对偏差（Mean absolute deviation）。 |
| `prod`             | 所有值的乘积。                                          |
| `var`              | 值的样本方差。                                          |
| `std`              | 值的样本标准差。                                        |
| `skew`             | 值的样本偏度（三阶矩）。                                |
| `kurt`             | 值的样本峰度（四阶矩）。                                |
| `cumsum`           | 值的累计和。                                            |
| `cummin`, `cummax` | 值的累计最小值或累计最大值。                            |
| `cumprod`          | 值的累计乘积。                                          |
| `diff`             | 计算一阶算术差分（对时间序列有用）。                    |
| `pct_change`       | 计算百分比变化。                                        |

#### 相关性和协方差

有一些统计指标（比如相关系数和协方差）是通过成对的数据计算出来的。来看几个股票价格和成交量的 DataFrame 例子，这些数据最初是从雅虎财经（Yahoo! Finance）获取的。作者已经把它们存成了 Python 特有的二进制文件（pickle 格式），你可以在本书配套的数据集里找到它们：

```python
price = pd.read_pickle("examples/yahoo_price.pkl")
volume = pd.read_pickle("examples/yahoo_volume.pkl")
```

现在计算价格的百分比变化，这是一种时间序列运算，将在第 11 章：时间序列中进一步探讨：

```python
returns = price.pct_change()
returns.tail()
#                 AAPL      GOOG       IBM      MSFT
# Date                                              
# 2016-10-17 -0.000680  0.001837  0.002072 -0.003483
# 2016-10-18 -0.000681  0.019616 -0.026168  0.007690
# 2016-10-19 -0.002979  0.007846  0.003583 -0.002255
# 2016-10-20 -0.000512 -0.005652  0.001719 -0.004867
# 2016-10-21 -0.003930  0.003011 -0.012474  0.042096
```

Series 的 `corr` 方法计算两个 Series 中重叠的、非 NA 的、按索引对齐的值的相关性。相关地，`cov` 计算协方差：

```python
returns["MSFT"].corr(returns["IBM"])
# 0.49976361144151166
returns["MSFT"].cov(returns["IBM"])
# 8.870655479703549e-05
```

另一方面，DataFrame 的 `corr` 和 `cov` 方法分别返回完整的相关或协方差矩阵作为 DataFrame：

```python
returns.corr()
#           AAPL      GOOG       IBM      MSFT
# AAPL  1.000000  0.407919  0.386817  0.389695
# GOOG  0.407919  1.000000  0.405099  0.465919
# IBM   0.386817  0.405099  1.000000  0.499764
# MSFT  0.389695  0.465919  0.499764  1.000000
returns.cov()
#           AAPL      GOOG       IBM      MSFT
# AAPL  0.000277  0.000107  0.000078  0.000095
# GOOG  0.000107  0.000251  0.000078  0.000108
# IBM   0.000078  0.000078  0.000146  0.000089
# MSFT  0.000095  0.000108  0.000089  0.000215
```

使用 DataFrame 的 `corrwith` 方法可以计算 DataFrame 的列或行与另一个 Series 或 DataFrame 之间的成对相关性。

传递 Series 返回一个 Series，其中包含为每列计算的相关值：

```python
returns.corrwith(returns["IBM"])
# AAPL    0.386817
# GOOG    0.405099
# IBM     1.000000
# MSFT    0.499764
# dtype: float64
```

传递 DataFrame 会计算匹配列名称的相关性。在这里，我计算百分比变化与交易量的相关性：

```python
returns.corrwith(volume)
# AAPL   -0.075565
# GOOG   -0.007067
# IBM    -0.204849
# MSFT   -0.092950
# dtype: float64
```

传递 `axis="columns"` 会逐行执行操作。在所有情况下，在计算相关性之前，数据点都会按标签对齐。

#### 唯一值、值计数和成员关系

另一类相关方法提取有关一维 Series 中包含的值的信息。为了说明这些，请考虑以下示例：

```python
obj = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])
uniques = obj.unique()
uniques
# array(['c', 'a', 'd', 'b'], dtype=object)
```

第一个函数是 `unique`，它为你提供 Series 中唯一值的数组。

唯一值不一定按它们首次出现的顺序返回，也不是按排序顺序返回，但如果需要，可以在事后对它们进行排序 (`uniques.sort()`)。相关地， `value_counts` 计算包含值频率的 Series：

```python
obj.value_counts()
# c    3
# a    3
# b    2
# d    1
# Name: count, dtype: int64
```

为了方便起见，该系列按值降序排列。 `value_counts` 也可用作顶级 pandas 方法，可与 NumPy 数组或其他 Python 序列一起使用：

```python
pd.value_counts(obj.to_numpy(), sort=False)
# c    3
# a    3
# d    1
# b    2
# Name: count, dtype: int64
```

`isin` 执行向量化集成员关系检查，可用于将数据集过滤为 DataFrame 中的 Series 或列中的值的子集：

```python
obj
# 0    c
# 1    a
# 2    d
# 3    a
# 4    a
# 5    b
# 6    b
# 7    c
# 8    c
# dtype: object
mask = obj.isin(["b", "c"])
mask
# 0     True
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6     True
# 7     True
# 8     True
# dtype: bool
obj[mask]
# 0    c
# 5    b
# 6    b
# 7    c
# 8    c
# dtype: object
```

与 `isin` 相关的是 `Index.get_indexer` 方法，它提供一个索引数组，每个索引代表了当前元素在目标集合的位置：

```python
to_match = pd.Series(["c", "a", "b", "b", "c", "a"])
unique_vals = pd.Series(["c", "b", "a"])
indices = pd.Index(unique_vals).get_indexer(to_match)
indices
# array([0, 2, 1, 1, 0, 2])
```

有关这些方法的参考，请参阅下表。

| Method         | Description                                                  |
| :------------- | :----------------------------------------------------------- |
| `isin`         | 计算一个布尔数组，指示每个 Series 或 DataFrame 值是否包含在传递的值序列中 |
| `get_indexer`  | 将数组中每个值的整数索引计算到另一个不同值的数组中；有助于数据对齐和连接类型操作 |
| `unique`       | 计算 Series 中唯一值的数组，按观察到的顺序返回               |
| `value_counts` | 返回一个包含唯一值作为其索引和频率作为其值的系列，按降序排列计数 |

在某些情况下，您可能需要计算 DataFrame 中多个相关列的直方图。这是一个例子：

```python
data = pd.DataFrame({"Qu1": [1, 3, 4, 3, 4],
                     "Qu2": [2, 3, 1, 2, 3],
                     "Qu3": [1, 5, 2, 4, 4]})
data
#    Qu1  Qu2  Qu3
# 0    1    2    1
# 1    3    3    5
# 2    4    1    2
# 3    3    2    4
# 4    4    3    4
```

我们可以计算单个列的值计数，如下所示：

```python
data["Qu1"].value_counts().sort_index()
# Qu1
# 1    1
# 3    2
# 4    2
# Name: count, dtype: int64
```

要计算所有列的值，请将 `pandas.value_counts` 传递给 DataFrame 的 `apply` 方法：

```python
result = data.apply(pd.value_counts).fillna(0)
result
#    Qu1  Qu2  Qu3
# 1  1.0  1.0  1.0
# 2  0.0  2.0  1.0
# 3  2.0  2.0  0.0
# 4  2.0  0.0  2.0
# 5  0.0  0.0  1.0
```

结果中的行标签是所有列中出现的不同值。这些值是每列中这些值的相应计数。

还有一个 `DataFrame.value_counts` 方法，但它将 DataFrame 的每一行视为一个元组来计算计数，以确定每个不同行的出现次数：

```python
data = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [0, 0, 1, 0, 0]})
data
#    a  b
# 0  1  0
# 1  1  0
# 2  1  1
# 3  2  0
# 4  2  0
data.value_counts()
# a  b
# 1  0    2
# 2  0    2
# 1  1    1
# Name: count, dtype: int64
```

在这种情况下，结果有一个索引，将不同的行表示为分层索引，我们将在第 8 章：数据整理：联接、组合和重塑中更详细地探讨该主题。
