---
layout: post
title: 数据分析案例：Olist 巴西电商公共数据集
date: 2026-01-11 19:30:08 +0000
tags: [Data Analysis, case, 阅读笔记]
mermaid: true
---

**数据集来源**：

- [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist)

> 这是真实的商业数据，所有商店和合作伙伴的文本都已替换为《权力的游戏》中的家族名称。
{: .prompt-tip }

**分析背景假设**：

[Olist](https://olist.com/) 作为一个连接商家和买家的平台，近期面临**营销转化率不稳定且物流差评增多**的情况。领导要我通过分析**找出高质量商家的特征，并优化用户的购物体验**。

## 整体分析框架

为保证分析的深度与可落地性，这个案例在**[描述性统计](https://www.spsspro.com/help/descriptive/)**的基础上，还基于**[跨行业数据挖掘标准流程（CRISP-DM）](https://www.ibm.com/docs/zh/spss-modeler/saas?topic=dm-crisp-help-overview)**的思想和 Olist 的 **[Marketplace 业务模式](https://zhuanlan.zhihu.com/p/565026406)**，构建了从 “营销获客” 到 “履约交付” 的全链路分析闭环。

整个分析框架主要分为四个阶段：

1. **数据架构与全貌认知**

   这一阶段的目标是理清 11 张数据表的 **[ER 关系](https://www.visual-paradigm.com/cn/guide/data-modeling/what-is-entity-relationship-diagram/)**，构建可供分析的宽表模型。

   - 语义标准化：数据源的网站已经给出了所有字段的准确语义。
   - 业务逻辑映射：数据源的网站已经给出了现成的 Data Schema。

   在二者的基础上，结合 Olist 的业务模式，画出 ER 图即可。

2. **数据清洗**

   这一阶段的目标是将原始脏数据转化为高质量的分析资产。

   - 数据质量审计：检查数据的完整性（`Null` 值分布）、一致性（价格异常值、状态冲突）、唯一性（主键查重）和分布合理性（偏度和峰度），并以合适的方式处理非法数据（剔除或填充）。

   - 时效特征构造：计算 `delta_delivery_time`（实际送达 - 预计送达），用于量化物流延迟程度。
   - 地理特征处理：利用地理坐标计算商家与买家间的物理距离，为物流分析提供归因依据。
   - 多语言处理：将葡萄牙语的 `product_category` 映射为英语，确保可读性。

3. **指标体系构建与假设提出**

   这一阶段的目标是采用**[目标-策略-衡量（OSM）模型](https://zhuanlan.zhihu.com/p/1945238646325249909)**，将业务问题转化为可量化的数据指标。

   - 营销侧指标体系：建立从[营销合格线索（MQL）](https://www.shuziqianzhan.com/article/2383.html)到活跃商家中间各环节的[转化漏斗（Conversion Funnel）](https://www.shopify.com/zh/blog/ecommerce-funnel)，重点关注**[渠道转化率（CVR）](https://advertising.amazon.com/zh-cn/library/guides/conversion-rate#1)**与**转化耗时（Time To Conversion）**。
   - 运营侧指标体系：
     - 北极星指标：[成交总额（GMV）](https://wiki.mbalib.com/wiki/GMV)、订单量。
     - 效率指标：订单核销率、[物流按时交付率（OTD）](https://mbb.eet-china.com/blog/642052-398254.html)。
     - 体验指标：[净推荐值（NPS）](https://www.ibm.com/cn-zh/think/topics/net-promoter-score)替代指标（Review Score 平均分）。
   - 提出假设：
     - *H1*：来自特定渠道（如 Paid Search）的商家具有更高的[生命周期价值（LTV）](https://zhuanlan.zhihu.com/p/51914694)。
     - *H2*：物流延迟天数与客户评分之间存在显著的负相关非线性关系。

4. **归因洞察与可视化叙事**

   这一阶段的目标是通过可视化验证假设，并输出可执行的商业建议。

   - 全链路漏斗分析：打通营销表与订单表，评估不同来源商家的后续销售表现，优化市场投放策略。
   - RFM 用户/商家分层：基于 **[Recency-Frequency-Monetary 模型](https://blog.ocard.co/knowhow/crm-rfm-analysis/)**对商家进行价值分层，识别 “金牌商家” 特征。
   - 地理空间分析：绘制巴西物流热力图，识别 “高延迟、低评分” 的物流黑洞区域，提出仓储布局建议。
   - 归因分析：探究影响差评（Review Score < 3）的核心因子（是产品质量还是物流速度？）。

## 数据架构与全貌认知

**目标**：目标是理清 11 张数据表的 ER 关系，构建可供分析的宽表模型。

数据源网站已经给出了所有字段的准确语义和 Data Schema，这里根据已有信息画出 ER 图即可。

```mermaid
---
config:
  theme: neutral
---
erDiagram
	olist_customers {
		string customer_id PK
		string customer_unique_id UK
		string customer_zip_code_prefix
		string customer_city
		string customer_state
	}

	olist_orders {
		string order_id PK
		string customer_id FK
		string order_status
		timestamp order_purchase_timestamp
		timestamp order_approved_at
		timestamp order_delivered_carrier_date
		timestamp order_delivered_customer_date
		timestamp order_estimated_delivery_date
	}

	olist_order_items {
		string order_id FK
		int order_item_id
		string product_id FK
		string seller_id FK
		timestamp shipping_limit_date
		float price
		float freight_value
	}

	olist_products {
		string product_id PK
		string product_category_name 
		int product_name_lenght 
		int product_description_lenght 
		int product_photos_qty 
		float product_weight_g 
		float product_length_cm 
		float product_height_cm 
		float product_width_cm 
	}

	olist_sellers {
		string seller_id PK
		string seller_zip_code_prefix 
		string seller_city 
		string seller_state 
	}

    olist_order_reviews {
		string review_id PK
		string order_id FK
		int review_score 
		string review_comment_title 
		string review_comment_message 
		timestamp review_creation_date 
		timestamp review_answer_timestamp 
	}

	olist_order_payments {
		string order_id FK
		int payment_sequential 
		string payment_type 
		int payment_installments 
		float payment_value 
	}



	olist_geolocation {
		string geolocation_zip_code_prefix 
		float geolocation_lat 
		float geolocation_lng 
		string geolocation_city 
		string geolocation_state 
	}

	product_category_name_translation {
		string product_category_name PK
		string product_category_name_english UK
	}

	olist_marketing_qualified_leads {
		string mql_id PK
		date first_contact_date 
		string landing_page_id 
		string origin 
	}

	olist_closed_deals {
		string mql_id FK
		string seller_id FK
		string sdr_id 
		string sr_id 
		timestamp won_date 
		string business_segment 
		string lead_type 
		string lead_behaviour_profile 
		bool has_company 
		bool has_gtin 
		string average_stock 
		string business_type 
		string declared_product_catalog_size 
		string declared_monthly_revenue 
	}

	olist_customers||--o{olist_orders:"places"
	olist_geolocation||--o{olist_customers:"locates"
	olist_orders||--|{olist_order_items:"contains"
	olist_orders||--|{olist_order_payments:"paid_via"
	olist_orders||--o{olist_order_reviews:"receives"
	olist_products||--o{olist_order_items:"defines"
	olist_sellers||--o{olist_order_items:"fulfills"
	olist_geolocation||--o{olist_sellers:"locates"
	product_category_name_translation||--o{olist_products:"translates_category"
	olist_marketing_qualified_leads||--o|olist_closed_deals:"converts_to"
	olist_closed_deals|o--||olist_sellers:"becomes_seller"
```

## 数据清洗

**目标**：将原始脏数据转化为高质量的分析资产。

### 数据质量审计

首先需要明确，质量审计并不是要把所有字段的值全部清洗到一尘不染，对于大数据量的分析来说，有偶尔的脏数据是可以容忍的，况且现实世界的业务是很复杂的，有时候根本无法判断某个字段的值到底是不是合法的，这里只需要让数据绝大部分可用即可。

面对 Olist 数据集复杂的 11 张表结构，手动逐个检查字段不仅效率低下，而且很容易遗漏异常。为此，我构建了一个基于 Pandas 的通用数据审计脚本。该工具能自动计算当前目录的 `dataset` 文件夹下所有 `csv` 文件数据集的缺失率、基数分布、偏度峰度等 10+ 项关键指标，并把最终的审计结果保存在当前目录下的 `audit_report.xlsx` 文件中。

```python
import glob
import os.path

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_any_dtype, is_integer_dtype, is_object_dtype
from tqdm import tqdm

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def alert_row(row):
    """
    对单行数据进行规则告警
    有告警只以为着需要关注，并不意味着一定有问题

    :param row: 要告警的行
    :return: None
    """

    alerts = []
    is_integer = is_integer_dtype(row['type'])
    is_obj = is_object_dtype(row['type'])
    # 完整性检查
    if row['missing_pct'] > 0.0:
        alerts.append('missing_need_handle')
    # 一致性检查
    if is_integer and row['zero_pct'] > 0.0:
        alerts.append('if_zero_valid')
    if is_integer and row['negative_pct'] > 0.0:
        alerts.append('if_negative_valid')
    if is_integer and row['max'] > (row['mean'] + 3 * row['std']):
        alerts.append('check_polar_valid')
    # 唯一性检查
    if row['unique'] == 1:
        alerts.append('single_no_need')
    if row['top_pct'] > 0.95:
        alerts.append('too_many_top')
    if row['unique_pct'] > 0.95:
        alerts.append('may_pk_or_uk')
    if is_obj and row['unique'] / row['count'] > 0.9:
        alerts.append('may_need_nlp')
    # 分布合理性检查
    if is_integer and abs(row['skewness']) > 3:
        alerts.append('high_skewed')
    if is_integer and abs(row['kurtosis']) > 2:
        alerts.append('may_high_kurt')
    if is_integer and row['mean'] / row['median'] >= 100:
        alerts.append('mean_shifted')

    return ' | '.join(alerts)


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 数据质量审计脚本

    :param df: 要审计的 DataFrame
    :return: 审计结果
    """

    row_cnt, col_cnt = df.shape
    audit_stat = []

    # 按列统计数据
    for col in df.columns:
        col_stat = {}
        # 列基本信息
        col_series = df.loc[:, col]
        col_type = col_series.dtype
        col_stat.update({
            'column': col,
            'type': col_type,
            'count': col_cnt,
        })
        # 基础描述性统计
        col_missing_cnt = col_series.isna().sum()
        col_missing_pct = col_missing_cnt / row_cnt
        col_unique_cnt = col_series.nunique()
        col_unique_pct = col_unique_cnt / row_cnt
        col_stat.update({
            'missing': col_missing_cnt,
            'missing_pct': col_missing_pct,
            'unique': col_unique_cnt,
            'unique_pct': col_unique_pct,
        })
        # 数值型统计
        if is_numeric_dtype(col_series):
            col_stat.update({
                'min': col_series.min(),
                'max': col_series.max(),
                'mean': col_series.mean(),
                'median': col_series.median(),
                'std': col_series.std(),
                'zero': (col_series == 0).sum(),
                'zero_pct': (col_series == 0).sum() / row_cnt,
                'negative': (col_series < 0).sum(),
                'negative_pct': (col_series < 0).sum() / row_cnt,
                'skewness': col_series.skew(),
                'kurtosis': col_series.kurt(),
            })
        # 日期型统计
        elif is_datetime64_any_dtype(col_series):
            col_stat.update({
                'min': col_series.min(),
                'max': col_series.max(),
                'range': (col_series.max() - col_series.min()).days if not col_series.isna().all() else None
            })
        else:
            col_stat.update({
                'top': str(col_series.mode().iloc[0])[:20],
                'top_cnt': col_series.value_counts().iloc[0],
                'top_pct': col_series.value_counts().iloc[0] / row_cnt,
            })
        audit_stat.append(col_stat)

    audit_df = pd.DataFrame(audit_stat).set_index('column')
    audit_df.insert(0, 'alerts', audit_df.apply(alert_row, axis=1))

    return audit_df


def audit_all_dataset(folder_path):
    """
    遍历并审计文件夹下的所有 csv 文件

    :param folder_path: 目标文件夹
    :return: None
    """

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f'未能在 {folder_path} 中发现 csv 文件')
        return

    print(f'找到 {len(csv_files)} 个 csv 文件，开始审计')
    writer = pd.ExcelWriter('audit_report.xlsx', engine="xlsxwriter")

    for file_path in tqdm(csv_files, desc="Processing Files"):
        file_name = os.path.basename(file_path).replace('.csv', '')
        try:
            dataset = pd.read_csv(file_path)
            audit_df = audit_dataframe(dataset)
            audit_df.to_excel(writer, sheet_name=file_name[:30])
        except Exception as e:
            print(f'处理 {file_name} 时出错：{e}')

    writer.close()
    print('审计结束，审计报告保存在 audit_report.xlsx}')


if __name__ == '__main__':
    audit_all_dataset('./dataset')

```

例如，这是对 `olist_closed_deals_dataset.csv` 的审计数据结果，`alerts` 不代表这个字段一定是有问题的，只是提示需要关注哪些方面：

```
                                                           alerts     type  count  missing  missing_pct  unique  unique_pct                   top  top_cnt   top_pct  min         max          mean  median           std   zero  zero_pct  negative  negative_pct   skewness    kurtosis
column                                                                                                                                                                                                                                                                                   
mql_id                                may_pk_or_uk | may_need_nlp   object     14        0     0.000000     842    1.000000  000dd3543ac84d906eae      1.0  0.001188  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
seller_id                             may_pk_or_uk | may_need_nlp   object     14        0     0.000000     842    1.000000  00065220becb8785e2cf      1.0  0.001188  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
sdr_id                                               may_need_nlp   object     14        0     0.000000      32    0.038005  4b339f9567d060bcea4f    140.0  0.166271  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
sr_id                                                may_need_nlp   object     14        0     0.000000      22    0.026128  4ef15afb4b2723d8f3d8    133.0  0.157957  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
won_date                              may_pk_or_uk | may_need_nlp   object     14        0     0.000000     824    0.978622   2018-05-04 03:00:00      6.0  0.007126  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
business_segment               missing_need_handle | may_need_nlp   object     14        1     0.001188      33    0.039192            home_decor    105.0  0.124703  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
lead_type                                     missing_need_handle   object     14        6     0.007126       8    0.009501         online_medium    332.0  0.394299  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
lead_behaviour_profile                        missing_need_handle   object     14      177     0.210214       9    0.010689                   cat    407.0  0.483373  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
has_company                                   missing_need_handle   object     14      779     0.925178       2    0.002375                  True     58.0  0.068884  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
has_gtin                                      missing_need_handle   object     14      778     0.923990       2    0.002375                  True     54.0  0.064133  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
average_stock                                 missing_need_handle   object     14      776     0.921615       6    0.007126                  5-20     22.0  0.026128  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
business_type                                 missing_need_handle   object     14       10     0.011876       3    0.003563              reseller    587.0  0.697150  NaN         NaN           NaN     NaN           NaN    NaN       NaN       NaN           NaN        NaN         NaN
declared_product_catalog_size                 missing_need_handle  float64     14      773     0.918052      33    0.039192                   NaN      NaN       NaN  1.0      2000.0    233.028986   100.0  3.523806e+02    0.0  0.000000       0.0           0.0   2.731955    9.274852
declared_monthly_revenue                                           float64     14        0     0.000000      27    0.032067                   NaN      NaN       NaN  0.0  50000000.0  73377.679335     0.0  1.744799e+06  797.0  0.946556       0.0           0.0  28.036956  800.377062

```

一般来讲，我们解读审计报告，主要是检查数据的完整性、一致性、唯一性还有分布合理性：

#### 完整性检查

关注指标：`missing_pct`

- 完美：`missing_pct == 0`
- 可容忍：`0 <= missing_pct <= 0.3`，通常可以用均值或者众数填充，或者直接删除对应的行
- 无法容忍：`missing_pct > 0.3`，除非这个字段本身就是 “稀疏” 的，比如 “备注” 字段，否则这列没有分析价值
- 直接删除这列：`missing_pct > 0.8`

**检查结果**：

- `closed_deals`
  - `business_segment`、`lead_type`、`business_type` 缺失小于 1%，删除对应的行
  - `lead_behaviour_profile` 缺失 21%，这个字段是销售代表主观判断的客户类型，它的缺失本身就代表了 “暂时无法判断” 的信息，可以用 `unknown` 填充
  - `has_company`、`has_gtin`、`average_stock`、`declared_product_catalog_size` 缺失 90+%，删除这四列
- `marketing_qualified_leads`
  - `origin` 缺失 0.8%，删除对应的行
- `orders`
  - `order_approved_at`、`order_delivered_carrier_date`、`order_delivered_customer_date` 缺失率低于 3%，删除对应的行
- `order_reviews`
  - `review_comment_title` 和 `review_comment_message` 缺失分别超过 10% 和 40%，但是这两个是用户的评论，业务价值比较高，不适合直接删除，可以用空字符串填充
- `products`
  - 除了 `product_id` 之外，其余字段都有少量缺失（<2%），删除对应的行

#### 一致性检查

关注指标：`min`、`max`、`zero_pct`、`negative_pct`、`negative_pct`

- 异常零值：对于某些字段（如“身高”、“价格”、“耗时”），0 是没有物理意义的。如果 `zero_pct > 0`，说明存在脏数据。
- 异常负值：对于“年龄”、“销售额”、“库存”等字段，`min < 0` 或 `negative_pct > 0`，说明存在逻辑错误。
- 极值检查：如果 `max` 远大于 `mean + 3 * std`，或者 `max` 是 `median` 的数百倍，说明存在极端离群值。

**检查结果**：

- `order_items`
  - `order_item_id` 提示极值异常，但是这个字段是用来在同一个订单的不同商品排序的，大部分人一个订单只买少量东西，偶尔有人买了很多东西很正常，可以忽略处理。
- `order_payments`
  - `payment_sequential` 和 `payment_installments` 提示极值异常和零值异常，可以忽略处理，这是一个序列字段，并且部分人买东西不走分期付款。

#### 唯一性检查

关注指标：`unique_pct`、`unique` 、`top_pct` 

- 单一值：`unique == 1`，此列没有额外的信息贡献，建议直接删除
- 低方差：`top_pct > 0.95`，此列 95% 的数据都是同一个值，可能冗余太大，分析价值不高
- 主键判定：`unique_pct == 1`，如果此列是主键列，但是不满足条件，那么这列数据可能不干净
- 高基数：`type` 是 `object/category`，但 `unique_cnt` 非常大（接近行数）。说明这可能是一个杂乱的文本字段（如用户评论），而不是分类字段，需要 NLP 处理。

**检查结果**：

- `orders`
  - `order_status` 有 97% 都是 `delivered`，属于低方差，但是这个字段业务价值很高，暂且保留

#### 分布合理性检查

关注指标：`mean` vs `median`、`skewness`、`kurtosis`

- 偏态分布：如果 `abs(skewness) > 3`，说明数据分布严重倾斜（长尾）。虽然不一定是脏数据，但在建模前通常需要做对数变换。

- 均值偏移：如果 `mean` 和 `median` 差异巨大（例如 `mean=1000`、`median=10`），通常暗示有巨大离群值拉高了均值。


**检查结果**：只有一些代表序列的字段发生了偏态和均值偏移，这是正常的，不做处理。

### 时效特征构造

