import akshare as ak
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import numpy as np

from core.utils.config_setting import Config
from core.llms.llm_factory import LLMFactory
llm_factory = LLMFactory()
llm_client = llm_factory.get_instance()

start_date = "20240101"  # 从2024年1月1日开始
def get_futures_data(symbol, start_date, end_date):
    try:
        # 尝试使用 futures_zh_daily_sina 函数获取数据
        data = ak.futures_zh_daily_sina(symbol=symbol)
        print(f"获取到 {symbol} 的原始数据：")
        print(data.head())
        
        if 'date' not in data.columns:
            print(f"警告：{symbol} 数据中没有 'date' 列。可用的列：{data.columns}")
            if '日期' in data.columns:
                data = data.rename(columns={'日期': 'date'})
            else:
                return None
        
        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        if data.empty:
            print(f"警告：{symbol} 在指定日期范围内没有数据")
            return None
        
        data = data.set_index('date')
        return data['close'] if 'close' in data.columns else data['收盘价']
    except Exception as e:
        print(f"获取 {symbol} 数据时发生错误: {e}")
        return None

# 设置日期范围
end_date = datetime.now()
start_date = end_date - timedelta(days=60)  # 获取最近60天的数据
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

print(f"获取从 {start_date_str} 到 {end_date_str} 的数据")

# 定义要获取的期货品种
symbols = ['IH0', 'IF0', 'IC0', 'IM0', 'TS0', 'TF0', 'T0', 'TK0']

# 获取数据
data_dict = {}
for symbol in symbols:
    data = get_futures_data(symbol, start_date, end_date)
    if data is not None:
        data_dict[symbol] = data

# 将所有数据合并到一个DataFrame中
df = pd.DataFrame(data_dict)
print("\n合并后的数据：")
print(df.head())
print(f"数据形状：{df.shape}")

if df.empty:
    print("错误：没有获取到任何有效数据。请检查数据源和网络连接。")
    exit()

# 计算收益率
returns = df.pct_change().dropna()
print("\n收益率数据：")
print(returns.head())
print(f"收益率数据形状：{returns.shape}")

# 计算相关性矩阵
correlation_matrix = returns.corr()
print("\n相关性矩阵：")
print(correlation_matrix)

# 检查相关性矩阵是否有效
if correlation_matrix.isnull().all().all():
    print("错误：相关性矩阵全为 NaN。请检查原始数据。")
    exit()

# 创建热力图
llm_factory.configure_matplotlib_for_chinese()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('期货收益相关性热力图')
plt.xlabel('期货品种')
plt.ylabel('期货品种')

# 保存热力图
os.makedirs('./output', exist_ok=True)
image_filename = f"{uuid.uuid4()}.png"
image_path = os.path.join('./output', image_filename)
plt.savefig(image_path)
plt.close()

# 找出最高和最低相关性
corr_unstack = correlation_matrix.unstack()
corr_unstack = corr_unstack[corr_unstack != 1.0]  # 移除对角线上的值
highest_corr = corr_unstack.max()
highest_corr_pair = corr_unstack.idxmax()
lowest_corr = corr_unstack.min()
lowest_corr_pair = corr_unstack.idxmin()

# 生成分析结果
analysis_result = f"""
# 期货收益相关性分析结果

## 1. 相关性矩阵

```
{correlation_matrix.to_string()}
```

## 2. 分析总结

- 最高相关性：{highest_corr_pair[0]} 和 {highest_corr_pair[1]} 之间的相关系数为 {highest_corr:.4f}
- 最低相关性：{lowest_corr_pair[0]} 和 {lowest_corr_pair[1]} 之间的相关系数为 {lowest_corr:.4f}
- 平均相关性：所有品种之间的平均相关系数为 {corr_unstack.mean():.4f}

## 3. 分析描述

基于 {', '.join(symbols)} 期货的日收益率数据，我们进行了相关性分析。主要发现如下：

- 最高相关性出现在 {highest_corr_pair[0]} 和 {highest_corr_pair[1]} 之间，相关系数为 {highest_corr:.4f}。
- 最低相关性出现在 {lowest_corr_pair[0]} 和 {lowest_corr_pair[1]} 之间，相关系数为 {lowest_corr:.4f}。
- 所有品种之间的平均相关系数为 {corr_unstack.mean():.4f}。

## 4. 热力图

![期货收益相关性热力图]({image_path})

热力图展示了不同期货品种之间相关性的强度。颜色越接近红色表示正相关性越强，越接近蓝色表示负相关性越强。

这些结果可用于构建多元化投资组合或设计套利策略。但请注意，相关性可能随时间变化，建议定期更新分析。
"""

print(analysis_result)

# 保存分析结果
with open("./markdowns/corr_fin_fut.md", "w", encoding="utf-8") as f:
    f.write(analysis_result)