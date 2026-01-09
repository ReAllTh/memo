---
layout: post
title: 《Python for Data Analysis, 3E》 阅读笔记
date: 2025-12-30 09:02:08 +0000
tags: [Python, Data Analysis, 阅读笔记]
---

**更新**：感觉像第 4 章、第 5 章那样基本直接翻译原书，没啥意义，LC 上遇到题还是睁眼瞎，不如只记录常用 API，然后多实践，不会用 API 再去翻原书或者 Ref。

**更新**：做些实际的例子会比较好。后面应该会发一些数据分析的案例。

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

- **略过**：第 9 章、第 11 章、第 13 章、附录 A & B
- **浏览**：第 1 章、第 2 章、第 3 章
- **摘录**：第 4 章、第 6 章、第 12 章
- **精读**：第 5 章、第 7 章、第 8 章、第 10 章

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

## 第 6 章：数据加载、存储和文件格式

加载数据是使用书中大多数工具的必要的第一步。

pandas 具有许多用于将表格数据读取为 DataFrame 对象的函数。下表总结了其中一些； `pandas.read_csv` 是最常用的之一。

| 函数             | 描述                                                         |
| :--------------- | :----------------------------------------------------------- |
| `read_csv`       | 从文件、URL 或文件型对象（file-like object）读取分隔数据；默认使用逗号作为分隔符。 |
| `read_table`     | 从文件、URL 或文件型对象读取分隔数据；默认使用制表符 (`'\t'`) 作为分隔符。 |
| `read_fwf`       | 读取固定宽度列格式（fixed-width column format）的数据（即没有分隔符）。 |
| `read_clipboard` | `read_table` 的剪贴板版本，用于从系统剪贴板读取数据；在将网页中的表格转换为数据时非常有用。 |
| `read_excel`     | 从 Excel XLS 或 XLSX 文件读取表格数据。                      |
| `read_hdf`       | 读取由 pandas 编写的 HDF5 文件。                             |
| `read_html`      | 读取给定 HTML 文档中发现的所有表格。                         |
| `read_json`      | 从 JSON（JavaScript 对象表示法）字符串表示、文件、URL 或文件型对象读取数据。 |
| `read_feather`   | 读取 Feather 二进制文件格式。                                |
| `read_parquet`   | 读取 Apache Parquet 二进制文件格式。                         |
| `read_orc`       | 读取 Apache ORC 二进制文件格式。                             |
| `read_pickle`    | 读取以 Python pickle 格式存储的任意对象。                    |
| `read_sas`       | 读取存储为 SAS 自定义存储格式之一的 SAS 数据集。             |
| `read_spss`      | 读取由 SPSS 创建的数据文件。                                 |
| `read_sql`       | 将 SQL 查询的结果（使用 SQLAlchemy）读取为 pandas DataFrame。 |
| `read_stata`     | 读取 Stata 文件格式的数据集。                                |
| `read_xml`       | 读取 XML 文档。                                              |

这些函数提供了非常丰富的的可选参数，像是最常用 `read_csv` 有 50 多个可选参数，在不知道怎么按照自己想要的方式读取文件时，建议参考 pandas 官方在线文档或者咨询 AI。

## 第 7 章：数据清洗与准备

在进行数据分析和建模的过程中，分析师 80% 或更多的时间花费在数据准备上：加载、清洗、转换和重新排列。

这章将讨论用于丢失数据、重复数据、字符串操作和其他一些分析数据转换的工具。

### 7.1 处理缺失值

pandas 采用了 R 编程语言中使用的约定，将缺失数据称为 NA，代表不可用（Not Available）。

在统计应用中，NA 数据可能是不存在的数据，也可能是存在但未被观察到的数据（例如，由于数据收集问题）。

在清理数据进行分析时，对缺失数据本身进行分析通常很重要，以识别数据收集问题或由缺失数据引起的数据潜在偏差。

pandas 提供了一些用于处理缺失值的方法，如下表。

| Method    | Description                                                  |
| :-------- | :----------------------------------------------------------- |
| `dropna`  | 根据数值是否存在缺失来过滤轴标签，并且可以针对允许缺失的数据量设定不同的阈值。 |
| `fillna`  | 使用指定的值或插值方法（例如 `'ffill'` 或 `'bfill'`）来填充缺失数据。 |
| `isnull`  | 返回一个含有布尔值的对象，这些布尔值表示哪些值是缺失值 / NA。 |
| `notnull` | `isnull` 的否定形式（即：如果值不缺失，返回 True）。         |

### 7.2 转换数据

pandas 提供了一些用于数值转换的方法，如下表。

| Method            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `drop_duplicates` | 删除重复值，默认保留第一个不重复的值。                       |
| `map`             | 根据传入的字典或者方法，将已有的值映射为新的值。也可以用在轴的重命名上。 |
| `replace`         | 把表中指定值统一替换成新值                                   |
| `cut`             | 根据传入的 list，或者要期望的分箱数量，返回一个 Categorical 对象，用于对数据做后续的分箱操作 |
| `qcut`            | 不同于 `cut` 的按区间等宽划分，这个方法会按照样本等频划分，保证每个区间的样本数量基本一致 |
| `describe`        | 描述当前表的统计信息，包括计数、均值、标准差、分位数等       |
| `sample`          | 随机采样（默认不放回）                                       |
| `get_dummies`     | 把分类变量转换成数值矩阵（默认布尔矩阵），一般用在机器学习场景的独热编码 |

### 7.3 扩展数据类型

pandas 开发了一个扩展类型系统，允许添加新的数据类型，即使 NumPy 本身不支持它们。

这些新数据类型可以与来自 NumPy 数组的数据一起视为同级的数据类型。



| Extension Type                        | Description                                                  | Type Code / Alias                             |
| :------------------------------------ | :----------------------------------------------------------- | :-------------------------------------------- |
| `CategoricalDtype`                    | Categorical data (分类型数据/类别数据)                       | `'category'`                                  |
| `Int64`, `Int32`, `Int16`, `Int8`     | Nullable signed integer types (支持缺失值的有符号整数类型)   | `'Int64'`, `'Int32'`, `'Int16'`, `'Int8'`     |
| `UInt64`, `UInt32`, `UInt16`, `UInt8` | Nullable unsigned integer types (支持缺失值的无符号整数类型) | `'UInt64'`, `'UInt32'`, `'UInt16'`, `'UInt8'` |
| `Float64`, `Float32`                  | Nullable floating-point types (支持缺失值的浮点类型，主要用于与整数类型保持一致) | `'Float64'`, `'Float32'`                      |
| `StringDtype`                         | Dedicated string type (专用字符串类型，解决了 object 类型的性能和混合类型问题) | `'string'`                                    |
| `BooleanDtype`                        | Nullable boolean type (支持缺失值的布尔类型)                 | `'boolean'`                                   |
| `DatetimeTZDtype`                     | Time zone aware datetime (带时区信息的日期时间类型)          | `'datetime64[ns, <tz>]'`                      |
| `PeriodDtype`                         | Time spans / periods (时间段/周期类型)                       | `'period[<freq>]'`                            |
| `IntervalDtype`                       | Intervals (区间类型)                                         | `'interval'`                                  |

这些扩展数据类型主要有以下好处：

*   支持缺失值 (Nullable)：传统的 NumPy 整数和布尔数组不支持 `NaN`（缺失值），如果你在整数列中引入 `NaN`，它会被强制转换为浮点数（float）。Pandas 的扩展类型（如 `Int64` 和 `boolean`）使用专门的 `pd.NA` 来表示缺失值，从而保持数据原本的整数或布尔性质。
*   专用字符串 (`string`)：传统上 pandas 使用 `object` 类型存储字符串，这效率较低且容易混入非字符串对象。`StringDtype` 是专门为文本数据设计的。

需要注意扩展类型是大小写敏感的，像 `'Int64'` 这样的类型别名是大写的，这与 NumPy 的 `'int64'`（全小写，不支持原生缺失值）是区分开的。

### 7.4 字符串操作

Python 长期以来一直是一种流行的原始数据操作语言，部分原因是它易于用于字符串和文本处理。使用字符串对象的内置方法可以使大多数文本操作变得简单。对于更复杂的模式匹配和文本操作，可能需要正则表达式。 pandas 使您能够在整个数据数组上简洁地应用字符串和正则表达式，此外还可以处理缺失数据。

#### Python 内置的字符串操作

| Method                      | Description                                                  |
| :-------------------------- | :----------------------------------------------------------- |
| `count`                     | 返回子串在字符串中非重叠出现的次数。                         |
| `endswith`                  | 如果字符串以指定后缀结尾，则返回 True。                      |
| `startswith`                | 如果字符串以指定前缀开头，则返回 True。                      |
| `join`                      | 将字符串作为分隔符，用于连接其他字符串序列。                 |
| `index`                     | 如果在字符串中找到子串，则返回子串第一个字符的位置。如果未找到，则引发 ValueError 异常。 |
| `find`                      | 如果在字符串中找到子串，则返回子串第一个字符的位置。类似于 index，但如果未找到则返回 -1。 |
| `rfind`                     | 如果在字符串中找到子串，则返回子串第一个字符的位置。类似于 find，但从字符串末尾开始搜索。 |
| `replace`                   | 将指定子串的所有出现替换为另一个字符串。                     |
| `strip`, `rstrip`, `lstrip` | 去除空白字符（包括换行符）；rstrip 从右侧去除，lstrip 从左侧去除。 |
| `split`                     | 使用指定分隔符将字符串拆分为子串列表。                       |
| `lower`                     | 将字母字符转换为小写。                                       |
| `upper`                     | 将字母字符转换为大写。                                       |
| `casefold`                  | 将字符转换为小写，常用于不区分大小写的匹配（比 lower 更彻底，例如处理德语字符 'ß'）。 |
| `ljust`, `rjust`            | 左对齐或右对齐；使用空格（或指定字符）将字符串填充至指定宽度。 |

#### 正则表达式

| Method        | Description                                                  |
| :------------ | :----------------------------------------------------------- |
| `findall`     | 将字符串中所有的非重叠匹配项作为列表返回。                   |
| `finditer`    | 与 `findall` 类似，但返回的是一个迭代器。                    |
| `match`       | 从字符串的起始位置匹配模式，并可选择将模式组件分段成组；如果匹配成功，返回一个匹配对象，否则返回 `None`。 |
| `search`      | 扫描字符串以查找匹配模式；如果找到则返回一个匹配对象。与 `match` 不同，`search` 可以在字符串的任意位置匹配，而不仅仅是开头。 |
| `split`       | 根据模式出现的每一处将字符串拆分为片段。                     |
| `sub`, `subn` | 将字符串中出现的所有模式替换为指定内容（`subn` 还会返回替换发生的次数）。 |

#### pandas 的字符串操作

| 方法 (Method)               | 描述 (Description)                                           |
| :-------------------------- | :----------------------------------------------------------- |
| `cat`                       | 实现元素级的字符串连接，可指定分隔符 (delimiter)。           |
| `contains`                  | 返回布尔数组，表示每个字符串是否包含指定的模式 (pattern) 或正则表达式。 |
| `count`                     | 统计模式 (pattern) 出现的次数。                              |
| `extract`                   | 使用正则表达式的分组捕获功能，从字符串中提取一个或多个子字符串；结果通常为 DataFrame。 |
| `endswith`                  | 对每个元素执行 `x.endswith(pattern)`（检查是否以指定模式结尾）。 |
| `startswith`                | 对每个元素执行 `x.startswith(pattern)`（检查是否以指定模式开头）。 |
| `findall`                   | 计算每个字符串中所有匹配该模式/正则表达式的列表。            |
| `get`                       | 获取各元素中第 `i` 个字符（索引访问）。                      |
| `isalnum`                   | 等同于内置的 `str.isalnum`（检查是否全为字母或数字）。       |
| `isalpha`                   | 等同于内置的 `str.isalpha`（检查是否全为字母）。             |
| `isdecimal`                 | 等同于内置的 `str.isdecimal`（检查是否全为十进制数字）。     |
| `isdigit`                   | 等同于内置的 `str.isdigit`（检查是否全为数字）。             |
| `islower`                   | 等同于内置的 `str.islower`（检查是否全为小写）。             |
| `isnumeric`                 | 等同于内置的 `str.isnumeric`（检查是否全为数值字符）。       |
| `isupper`                   | 等同于内置的 `str.isupper`（检查是否全为大写）。             |
| `join`                      | 根据指定的分隔符，将 Series 中每个元素的字符串连接起来。     |
| `len`                       | 计算每个字符串的长度。                                       |
| `lower`, `upper`            | 转换大小写；分别等同于 `x.lower()` 和 `x.upper()`。          |
| `match`                     | 根据指定的正则表达式对每个元素执行 `re.match`，返回匹配的分组列表或布尔值。 |
| `pad`                       | 在字符串的左侧、右侧或两侧添加空白符。                       |
| `center`                    | 等同于 `pad(side='both')`，将字符串居中。                    |
| `repeat`                    | 重复字符串值（例如 `s.str.repeat(3)`）。                     |
| `replace`                   | 将匹配到的模式/正则表达式替换为其他字符串。                  |
| `slice`                     | 对 Series 中的每个字符串进行切片操作。                       |
| `split`                     | 根据分隔符或正则表达式将字符串拆分为列表。                   |
| `strip`, `rstrip`, `lstrip` | 去除空白符（包括换行符）；分别为去除两端、右端或左端的空白。 |

**补充说明：**

*   这些方法通过 Pandas Series 的 `.str` 属性调用（例如 `data.str.contains('gmail')`）。
*   它们会自动处理缺失值（NA/NaN），这是相比于 Python 原生循环处理的一大优势。
*   大部分接受 `pattern` 参数的方法默认支持正则表达式（RegEx）。

### 7.5 数据分类（Categorical）

在 Pandas 中，`Categorical` 是一种专门用于处理类别型变量（Categorical variables）的数据类型。它在统计学中对应那些取值范围有限且通常固定的变量。

`Categorical` 主要解决以下三个问题：

- 节省内存：如果一个字符串列包含大量重复值（如“男/女”、“省份”），使用 `category` 类型会比 `object`（字符串）类型节省大量空间。它内部使用整数（codes）来存储数据，而将实际的字符串值（categories）只存储一份。
- 逻辑排序（非字母表排序）：你可以定义类别的逻辑顺序。例如，“低”、“中”、“高”如果按字母排序是“高、低、中”，但定义为 `Categorical` 后，可以按“低 < 中 < 高”排序。
- 性能优化：在进行分组（Groupby）、排序或某些字符串操作时，由于是对整数代码进行操作，速度通常比处理原始字符串快得多。

`Categorical` 数据由两部分组成：

1. Categories（类别）：唯一值的集合（如 `['A', 'B', 'C']`）。
2. Codes（代码）：一个整数数组，记录每个位置对应哪个类别（如 `[0, 1, 0, 2]` 代表 `['A', 'B', 'A', 'C']`）。

#### 创建 Categorical 对象

你可以直接转换现有列，或手动创建。

```python
import pandas as pd
import numpy as np

# 方式 1：转换现有 Series
df = pd.DataFrame({"fruit": ["apple", "banana", "apple", "apple"]})
df["fruit"] = df["fruit"].astype("category")

# 方式 2：手动创建并指定顺序
s = pd.Series(["low", "high", "medium", "low"], dtype="category")
s = s.cat.set_categories(["low", "medium", "high"], ordered=True)
```

#### 排序示例

如果不使用 `category`，排序将按字母顺序。

```python
# 定义有顺序的类别
levels = ["Junior", "Senior", "Manager"]
df = pd.DataFrame({"rank": ["Manager", "Junior", "Senior", "Junior"]})

# 转换为有序类别
df["rank"] = pd.Categorical(df["rank"], categories=levels, ordered=True)

# 排序时会遵循 Junior -> Senior -> Manager
print(df.sort_values("rank"))
```

#### 使用 `.cat` 访问器

类似于字符串的 `.str` 或日期的 `.dt`，类别型数据有专门的 `.cat` 属性来管理类别：

- `df['col'].cat.categories`：查看所有类别。
- `df['col'].cat.codes`：查看底层的整数代码。
- `df['col'].cat.rename_categories([...])`：重命名类别名称。
- `df['col'].cat.add_categories([...])`：添加新类别。

#### 什么时候不该用

- 高基数数据：如果一列中几乎每个值都是唯一的（如 ID、身份证号），转换为 `category` 反而会增加开销，因为它需要额外维护一份类别映射表。
- 频繁修改值：如果你需要给这一列赋值一个“不在已有类别中”的新值，Pandas 会报错（必须先用 `add_categories` 添加该值）。

这张表列出了可以通过 `series.cat` 访问的用于操作分类数据的方法。

| Method                     | Description                                                  |
| :------------------------- | :----------------------------------------------------------- |
| `add_categories`           | 在现有类别的末尾追加新的（未使用的）类别。                   |
| `remove_categories`        | 移除指定的类别；若数据中存在属于该类别的值，这些值将被设置为 null (NaN)。 |
| `remove_unused_categories` | 移除数据中未出现（未使用）的类别。                           |
| `rename_categories`        | 用指定的新名称替换类别（不改变类别的数量，只改变名称）。     |
| `reorder_categories`       | 改变类别的顺序（不改变数据本身的顺序，只改变类别索引的逻辑顺序）。 |
| `set_categories`           | 将类别替换为一组新的指定类别；此操作可以添加新类别，也可以移除旧类别（取决于新列表的内容）。 |
| `as_ordered`               | 将分类数据设置为有序（Ordered）。                            |
| `as_unordered`             | 将分类数据设置为无序（Unordered）。                          |

## 第 8 章：数据规整：连接、合并与重塑

在许多应用中，数据可能分布在多个文件或数据库中，或者以不便于分析的形式排列。本章重点介绍帮助组合、连接和重新排列数据的工具。

### 8.1 分层索引

分层索引是 pandas 的一个重要功能，它使你能够在一个轴上拥有多个（两个或更多）索引级别。

另一种思考方式是，它提供了一种以较低维度形式处理较高维度数据的方法。

让我们从一个简单的示例开始：创建一个以列表（或数组）列表作为索引的 Series：

```python
data = pd.Series(np.random.uniform(size=9),
                 index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data
# a  1    0.929616
#    2    0.316376
#    3    0.183919
# b  1    0.204560
#    3    0.567725
# c  1    0.595545
#    2    0.964515
# d  2    0.653177
#    3    0.748907
# dtype: float64
```

你看到的是一个以 `MultiIndex` 作为索引的系列的美化视图。索引显示中的 “间隙” 意味着 “使用正上方的标签”：

```python
data.index
# MultiIndex([('a', 1),
#             ('a', 2),
#             ('a', 3),
#             ('b', 1),
#             ('b', 3),
#             ('c', 1),
#             ('c', 2),
#             ('d', 2),
#             ('d', 3)],
#            )
```

使用分层索引对象，可以实现所谓的部分索引，使您能够简洁地选择数据子集：

```python
data["b"]
# 1    0.204560
# 3    0.567725
# dtype: float64
data["b":"c"]
# b  1    0.204560
#    3    0.567725
# c  1    0.595545
#    2    0.964515
# dtype: float64
data.loc[["b", "d"]]
# b  1    0.204560
#    3    0.567725
# d  2    0.653177
#    3    0.748907
# dtype: float64
```

甚至可以从“内部”层面进行选择。这里我从第二个索引级别选择所有值为 2 的值：

```python
data.loc[:, 2]
# a    0.316376
# c    0.964515
# d    0.653177
# dtype: float64
```

分层索引在重塑数据和基于组的操作（例如形成数据透视表）中发挥着重要作用。例如，您可以使用其 `unstack` 方法将此数据重新排列到 DataFrame 中：

```python
data.unstack()
#           1         2         3
# a  0.929616  0.316376  0.183919
# b  0.204560       NaN  0.567725
# c  0.595545  0.964515       NaN
# d       NaN  0.653177  0.748907
```

`unstack` 的逆操作是 `stack`：

```python
data.unstack().stack()
# a  1    0.929616
#    2    0.316376
#    3    0.183919
# b  1    0.204560
#    3    0.567725
# c  1    0.595545
#    2    0.964515
# d  2    0.653177
#    3    0.748907
# dtype: float64
```

对于 DataFrame，任一轴都可以有分层索引：

```python
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame
#      Ohio     Colorado
#     Green Red    Green
# a 1     0   1        2
#   2     3   4        5
# b 1     6   7        8
#   2     9  10       11
```

层次结构级别可以有名称（作为字符串或任何 Python 对象）。如果是这样，这些将显示在控制台输出中：

```python
frame.index.names = ["key1", "key2"]
frame.columns.names = ["state", "color"]
frame
# state      Ohio     Colorado
# color     Green Red    Green
# key1 key2                   
# a    1        0   1        2
#      2        3   4        5
# b    1        6   7        8
#      2        9  10       11
```

通过部分列索引，你可以类似地选择列组：

```python
frame["Ohio"]
# color      Green  Red
# key1 key2            
# a    1         0    1
#      2         3    4
# b    1         6    7
#      2         9   10
```

有时，您可能需要重新排列轴上级别的顺序或按某一特定级别中的值对数据进行排序。 `swaplevel` 方法采用两个级别编号或名称，并返回一个级别互换的新对象（但数据未改变）：

```python
frame.swaplevel("key1", "key2")
# state      Ohio     Colorado
# color     Green Red    Green
# key2 key1                   
# 1    a        0   1        2
# 2    a        3   4        5
# 1    b        6   7        8
# 2    b        9  10       11
```

`sort_index` 默认情况下使用所有索引级别按字典顺序对数据进行排序，但您可以通过传递 level 参数选择仅使用单个级别或级别的子集进行排序。例如：

```python
frame.sort_index(level=1)
# state      Ohio     Colorado
# color     Green Red    Green
# key1 key2                   
# a    1        0   1        2
# b    1        6   7        8
# a    2        3   4        5
# b    2        9  10       11
frame.swaplevel(0, 1).sort_index(level=0)
# state      Ohio     Colorado
# color     Green Red    Green
# key2 key1                   
# 1    a        0   1        2
#      b        6   7        8
# 2    a        3   4        5
#      b        9  10       11
```

DataFrame 和 Series 上的许多描述性和摘要统计数据都有一个级别选项，您可以在其中指定要在特定轴上聚合的级别。考虑上面的 DataFrame；我们可以按行或列的级别进行聚合，如下所示：

```python
frame.groupby(level="key2").sum()
# state  Ohio     Colorado
# color Green Red    Green
# key2                    
# 1         6   8       10
# 2        12  14       16
frame.groupby(level="color", axis="columns").sum()
# color      Green  Red
# key1 key2            
# a    1         2    1
#      2         8    4
# b    1        14    7
#      2        20   10
```

使用 DataFrame 中的一列或多列作为行索引并不罕见；或者，您可能希望将行索引移动到 DataFrame 的列中。这是一个数据框示例：

```python
frame = pd.DataFrame({"a": range(7), "b": range(7, 0, -1),
                      "c": ["one", "one", "one", "two", "two",
                            "two", "two"],
                      "d": [0, 1, 2, 0, 1, 2, 3]})
frame
#    a  b    c  d
# 0  0  7  one  0
# 1  1  6  one  1
# 2  2  5  one  2
# 3  3  4  two  0
# 4  4  3  two  1
# 5  5  2  two  2
# 6  6  1  two  3
```

DataFrame 的 `set_index` 函数将使用其一列或多列作为索引创建一个新的 DataFrame：

```python
frame2 = frame.set_index(["c", "d"])
frame2
#        a  b
# c   d      
# one 0  0  7
#     1  1  6
#     2  2  5
# two 0  3  4
#     1  4  3
#     2  5  2
#     3  6  1
```

默认情况下，列将从 DataFrame 中删除，但您可以通过将 drop=False 传递给 set_index 将它们保留在其中：

```python
frame.set_index(["c", "d"], drop=False)
#        a  b    c  d
# c   d              
# one 0  0  7  one  0
#     1  1  6  one  1
#     2  2  5  one  2
# two 0  3  4  two  0
#     1  4  3  two  1
#     2  5  2  two  2
#     3  6  1  two  3
```

另一方面，`reset_index` 的作用与 `set_index` 相反；分层索引级别移至列中：

```python
frame2.reset_index()
#      c  d  a  b
# 0  one  0  0  7
# 1  one  1  1  6
# 2  one  2  2  5
# 3  two  0  3  4
# 4  two  1  4  3
# 5  two  2  5  2
# 6  two  3  6  1
```

### 8.2 组合和合并数据集

pandas 对象中包含的数据可以通过多种方式组合：

- `pandas.merge`
  基于一个或多个键连接 DataFrame 中的行。SQL 或其他关系数据库的用户对此会很熟悉，因为它实现了数据库连接操作。
- `pandas.concat`
  沿轴将对象连接或 “堆叠” 在一起。
- `combine_first`
  将重叠数据拼接在一起，用另一个对象的值填充一个对象中的缺失值。

详细参考：https://wesmckinney.com/book/data-wrangling#prep_merge_join

### 8.3 重塑和旋转

有许多用于重新排列表格数据的基本操作。这些被称为重塑或旋转操作。

分层索引提供了一种在 DataFrame 中重新排列数据的一致方法。有两个主要操作：

- `stack`

  这从数据中的列“旋转”或旋转到行。这会让 DataFrame 变“窄”变“长”。

- `unstack`

  这从行转向列。这会让 DataFrame 变“宽”变“短”。

详细参考：https://wesmckinney.com/book/data-wrangling#prep_reshape

## 第 10 章：数据聚合与分组运算

对数据集进行分类并向每个组应用函数（无论是聚合还是转换）都可能是数据分析工作流程的关键组成部分。加载、合并和准备数据集后，你可能需要计算组统计数据或可能的数据透视表以用于报告或可视化目的。 pandas 提供了一个多功能的 `groupby` 接口，能够以自然的方式对数据集进行切片、切块和汇总。

![how to think group by op](https://wesmckinney.com/book/images/pda3_1001.png)

```python
df = pd.DataFrame({"key1" : ["a", "a", None, "b", "b", "a", None],
                   "key2" : pd.Series([1, 2, 1, 2, 1, None, 1],
                                      dtype="Int64"),
                   "data1" : np.random.standard_normal(7),
                   "data2" : np.random.standard_normal(7)})
df
#    key1  key2     data1     data2
# 0     a     1 -0.204708  0.281746
# 1     a     2  0.478943  0.769023
# 2  None     1 -0.519439  1.246435
# 3     b     2 -0.555730  1.007189
# 4     b     1  1.965781 -1.296221
# 5     a  <NA>  1.393406  0.274992
# 6  None     1  0.092908  0.228913
grouped = df["data1"].groupby(df["key1"])
grouped
# <pandas.core.groupby.generic.SeriesGroupBy object at 0x17b7913f0>
grouped.mean()
# key1
# a    0.555881
# b    0.705025
# Name: data1, dtype: float64
means = df["data1"].groupby([df["key1"], df["key2"]]).mean()
means
# key1  key2
# a     1      -0.204708
#       2       0.478943
# b     1       1.965781
#       2      -0.555730
# Name: data1, dtype: float64
means.unstack()
# key2         1         2
# key1                    
# a    -0.204708  0.478943
# b     1.965781 -0.555730
```

### 10.1 数据聚合

聚合是指从数组生成标量值的任何数据转换。前面的示例使用了其中的平均值。许多常见的聚合（如下表中的聚合）都有优化的实现。

| Function           | Description                                                  |
| :----------------- | :----------------------------------------------------------- |
| `any`, `all`       | 如果组内任何（一个或多个）或所有非 NA 值分别为“真”（truthy），则返回 True |
| `count`            | 组内非 NA 值的数量                                           |
| `cummin`, `cummax` | 非 NA 值的累积最小值和累积最大值                             |
| `cumsum`           | 非 NA 值的累积和                                             |
| `cumprod`          | 非 NA 值的累积积                                             |
| `first`, `last`    | 组内第一个和最后一个非 NA 值                                 |
| `mean`             | 非 NA 值的平均值                                             |
| `median`           | 非 NA 值的算术中位数                                         |
| `min`, `max`       | 非 NA 值的最小值和最大值                                     |
| `prod`             | 非 NA 值的乘积                                               |
| `std`, `var`       | 无偏（分母为 n-1）标准差和方差                               |
| `sum`              | 非 NA 值的总和                                               |

**补充说明：**

*   非 NA 值 (non-NA values)：指的是不包含缺失值（如 `NaN` 或 `None`）的数据。
*   无偏 (Unbiased)：在计算标准差和方差时，分母使用 $n-1$ 而不是 $n$，这在统计学中用于样本估计总体的偏差修正。

`GroupBy` 具有极强的扩展性。它不仅仅局限于自带的那几个聚合函数（如 `sum`, `max`），它可以作为一种 “容器”，让你在每个分组上运行任何该数据类型支持的方法，代价仅仅是计算速度比那些经过专门优化的函数慢一些：

```python
df
#    key1  key2     data1     data2
# 0     a     1 -0.204708  0.281746
# 1     a     2  0.478943  0.769023
# 2  None     1 -0.519439  1.246435
# 3     b     2 -0.555730  1.007189
# 4     b     1  1.965781 -1.296221
# 5     a  <NA>  1.393406  0.274992
# 6  None     1  0.092908  0.228913
grouped = df.groupby("key1")

# 即使 GroupBy 没有直接定义 nsmallest，你依然可以这样写：
result = grouped['data1'].nsmallest(2)
# key1   
# a     0   -0.204708
#       1    0.478943
# b     3   -0.555730
#       4    1.965781
# Name: data1, dtype: float64
```

你还可以自定义聚合函数，然后通过 `aggregate` 方法的短别名 `agg` 调用它：

```python
def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)
#       key2     data1     data2
# key1                          
# a        1  1.598113  0.494031
# b        1  2.521511  2.303410
```

### 10.2 Apply：通用的分组-计算-组合函数

最通用的 `GroupBy` 方法是 `apply`，它将正在操作的对象拆分为多个片段，在每个片段上调用传递的函数，然后尝试连接这些片段。

```python
def top(df, n=5, column="tip_pct"):
    return df.sort_values(column, ascending=False)[:n]
top(tips, n=6)
#      total_bill   tip smoker  day    time  size   tip_pct
# 172        7.25  5.15    Yes  Sun  Dinner     2  0.710345
# 178        9.60  4.00    Yes  Sun  Dinner     2  0.416667
# 67         3.07  1.00    Yes  Sat  Dinner     1  0.325733
# 232       11.61  3.39     No  Sat  Dinner     2  0.291990
# 183       23.17  6.50    Yes  Sun  Dinner     4  0.280535
# 109       14.31  4.00    Yes  Sat  Dinner     2  0.279525
tips.groupby("smoker").apply(top)
#             total_bill   tip smoker   day    time  size   tip_pct
# smoker                                                           
# No     232       11.61  3.39     No   Sat  Dinner     2  0.291990
#        149        7.51  2.00     No  Thur   Lunch     2  0.266312
#        51        10.29  2.60     No   Sun  Dinner     2  0.252672
#        185       20.69  5.00     No   Sun  Dinner     5  0.241663
#        88        24.71  5.85     No  Thur   Lunch     2  0.236746
# Yes    172        7.25  5.15    Yes   Sun  Dinner     2  0.710345
#        178        9.60  4.00    Yes   Sun  Dinner     2  0.416667
#        67         3.07  1.00    Yes   Sat  Dinner     1  0.325733
#        183       23.17  6.50    Yes   Sun  Dinner     4  0.280535
#        109       14.31  4.00    Yes   Sat  Dinner     2  0.279525
```

发生了什么？首先，`tips` DataFrame 根据 `smoker` 的值分为几组。然后对每个组调用 `top` 函数，并使用 `pandas.concat` 将每个函数调用的结果粘合在一起，并用组名称标记各个部分。因此，结果具有一个分层索引，其内部级别包含来自原始 DataFrame 的索引值。

### 10.3 分组 `transform` 和 “解包” `GroupBy`

在 pandas 中，`transform` 是一个非常强大的函数，它主要用于对 DataFrame 或 Series 执行操作，并返回一个与原始对象形状相同（即行数相同）的结果。

它最常与 `groupby` 结合使用，用于在不改变数据结构的情况下进行特征工程或数值转换。

`transform` 的三个核心特点是：

1. 保持维度：输出的行数必须与输入的行数完全一致。
2. 广播（Broadcasting）：它可以将聚合结果（如平均值）广播回原始数据的每一行。
3. 支持多种输入：可以接受函数、字符串（内置函数名）、列表或字典。

常结合 `Groupby` 进行数据标准化。这是 `transform` 最典型的用法。假设你有一组销售数据，你想计算每个销售员的每笔订单占其个人总销售额的比例。

如果使用 `sum()`，结果会缩减为每个销售员一行；而使用 `transform('sum')`，结果会保持原来的行数，方便直接计算。

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Alice', 'Bob', 'Bob'],
    'Sales': [100, 200, 300, 400]
})

# 计算每个人的总销售额并映射回原表
df['Total_Sales'] = df.groupby('Name')['Sales'].transform('sum')
#     Name  Sales  Total_Sales
# 0  Alice    100          300
# 1  Alice    200          300
# 2    Bob    300          700
# 3    Bob    400          700
```

另外一个典型场景是填充缺失值 (Imputation)，你可以根据分组的平均值来填充该组内的缺失值，而不是使用全局平均值。

```python
# 根据分组均值填充 NaN
df['Sales'] = df.groupby('Name')['Sales'].transform(lambda x: x.fillna(x.mean()))
```

`transform` 也可以直接作用于 Series，对每个元素执行函数（类似于 `map` 或 `apply`），但其限制是必须返回相同长度的数据。

#### transform 与 apply 的区别

| 特性     | `transform`                                            | `apply`                            |
| -------- | ------------------------------------------------------ | ---------------------------------- |
| 输出形状 | 必须与输入行数一致                                     | 可以是标量、减少行数或改变形状     |
| 操作范围 | 通常在单列上执行或分组后广播                           | 可以处理多列之间的交互（跨列计算） |
| 性能     | 对内置聚合函数（如 `sum`, `mean`）有高度优化，速度极快 | 灵活性更高，但通常速度较慢         |

当你需要改变数值但不改变数据框的形状（行数）时，应该优先选择 `transform`。它最适合用于：

- 计算组内百分比。
- 减去组内均值（去中心化）。
- 在原表中新增一列基于分组统计的参考指标。

### 10.4 数据透视表和交叉表

数据透视表是电子表格程序和其他数据分析软件中常见的数据汇总工具。它通过一个或多个键聚合数据表，将数据排列在一个矩形中，其中一些组键沿行，一些沿列。 Python 中使用 pandas 的数据透视表可以通过本章中描述的 `groupby` 工具，结合利用分层索引的重塑操作来实现。 DataFrame 还有一个 `pivot_table` 方法，还有一个顶级 `pandas.pivot_table` 函数。除了为 `groupby` 提供方便的接口之外，`pivot_table` 还可以添加部分总计。

返回到 `tips` 数据集，假设你想计算按行上的 `day` 和 `smoker` 排列的组均值表（默认的 `pivot_table` 聚合类型）：

```python
tips.head()
#    total_bill   tip smoker  day    time  size   tip_pct
# 0       16.99  1.01     No  Sun  Dinner     2  0.059447
# 1       10.34  1.66     No  Sun  Dinner     3  0.160542
# 2       21.01  3.50     No  Sun  Dinner     3  0.166587
# 3       23.68  3.31     No  Sun  Dinner     2  0.139780
# 4       24.59  3.61     No  Sun  Dinner     4  0.146808
tips.pivot_table(index=["day", "smoker"],
                 values=["size", "tip", "tip_pct", "total_bill"])
#                  size       tip   tip_pct  total_bill
# day  smoker                                          
# Fri  No      2.250000  2.812500  0.151650   18.420000
#      Yes     2.066667  2.714000  0.174783   16.813333
# Sat  No      2.555556  3.102889  0.158048   19.661778
#      Yes     2.476190  2.875476  0.147906   21.276667
# Sun  No      2.929825  3.167895  0.160113   20.506667
#      Yes     2.578947  3.516842  0.187250   24.120000
# Thur No      2.488889  2.673778  0.160298   17.113111
#      Yes     2.352941  3.030000  0.163863   19.190588
```

