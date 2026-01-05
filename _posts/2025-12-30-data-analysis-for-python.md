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

## 第 5 章 pandas 入门

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



