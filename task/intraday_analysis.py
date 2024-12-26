import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import akshare as ak
from core.llms.llm_factory import LLMFactory

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StockAnalyzer:
    def __init__(self, symbol='601127'):
        """
        初始化股票分析器
        :param symbol: 股票代码，默认为601127
        """
        self.symbol = symbol
        self.output_dir = './output'
        os.makedirs(self.output_dir, exist_ok=True)

    def stock_zh_a_spot_em(self):
        """
        东方财富网-沪 A 股-实时行情数据
        :return:  A 股上市公司的实时行情数据
        :rtype: pandas.DataFrame
        """
        return ak.stock_zh_a_spot_em()

    def get_stock_spot(self, symbol: str) -> dict:
        """
        获取个股实时行情数据
        :param symbol: 股票代码
        :type symbol: str
        :return: 个股实时行情数据，包含以下字段:
            - code: 代码
            - name: 名称
            - price: 最新价
            - change_percent: 涨跌幅(%)
            - change_amount: 涨跌额
            - volume: 成交量(手)
            - amount: 成交额(元)
            - amplitude: 振幅(%)
            - high: 最高价
            - low: 最低价
            - open: 今开
            - pre_close: 昨收
            - volume_ratio: 量比
            - turnover: 换手率(%)
            - pe: 市盈率(动态)
            - pb: 市净率
            - total_value: 总市值(元)
            - circulating_value: 流通市值(元)
            - speed: 涨速
            - change_5min: 5分钟涨跌(%)
            - change_60day: 60日涨跌幅(%)
            - change_year: 年初至今涨跌幅(%)
        :rtype: dict
        """
        # 获取实时行情数据
        df = self.stock_zh_a_spot_em()
        
        # 查找对应股票代码的数据
        result = df[df['代码'] == symbol]
        
        if result.empty:
            return {}
            
        # 提取第一行数据
        data = result.iloc[0]
        
        # 构建返回字典
        return {
            'code': data['代码'],
            'name': data['名称'],
            'price': float(data['最新价']),
            'change_percent': float(data['涨跌幅']),
            'change_amount': float(data['涨跌额']),
            'volume': float(data['成交量']),
            'amount': float(data['成交额']),
            'amplitude': float(data['振幅']),
            'high': float(data['最高']),
            'low': float(data['最低']),
            'open': float(data['今开']),
            'pre_close': float(data['昨收']),
            'volume_ratio': float(data['量比']),
            'turnover': float(data['换手率']),
            'pe': float(data['市盈率-动态']),
            'pb': float(data['市净率']),
            'total_value': float(data['总市值']),
            'circulating_value': float(data['流通市值']),
            'speed': float(data['涨速']),
            'change_5min': float(data['5分钟涨跌']),
            'change_60day': float(data['60日涨跌幅']),
            'change_year': float(data['年初至今涨跌幅'])
        }

    def load_data(self):
        """
        从akshare加载股票分时数据
        """
        self.df = ak.stock_intraday_em(symbol=self.symbol)
        return self.process_data()
    
    def process_data(self):
        """
        处理原始数据，添加时间间隔标记
        """
        # 获取当前日期，用于构建完整的datetime
        today = datetime.now().date()
        
        # 将时间字符串转换为完整的datetime
        self.df['完整时间'] = pd.to_datetime(
            self.df['时间'].apply(lambda x: f"{today} {x}"),
            format='%Y-%m-%d %H:%M:%S'
        )
        
        # 保留原始时间列为time类型
        self.df['时间'] = self.df['完整时间'].dt.time
        
        # 添加时间间隔标记
        self.df['5分钟间隔'] = self.df['完整时间'].dt.floor('5min')
        self.df['15分钟间隔'] = self.df['完整时间'].dt.floor('15min')
        self.df['30分钟间隔'] = self.df['完整时间'].dt.floor('30min')
        
        return self.df
    
    def create_heatmap(self):
        """
        创建交易热力图，5分钟间隔和1小时段
        """
        # 按小时和5分钟间隔分组计算
        self.df['hour'] = self.df['完整时间'].dt.hour
        self.df['5min_interval'] = self.df['完整时间'].dt.minute // 5

        # 创建透视表
        pivot_data = pd.pivot_table(
            self.df,
            values=['成交价', '手数'],
            index='hour',
            columns='5min_interval',
            aggfunc={'成交价': 'mean', '手数': 'sum'}
        )

        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 价格热力图
        sns.heatmap(
            pivot_data['成交价'],
            ax=ax1,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            xticklabels=[f'{i*5:02d}' for i in range(12)],  # 5分钟间隔的标签
            yticklabels=[f'{i:02d}:00' for i in range(9, 16)]  # 交易时间9:00-15:00
        )
        ax1.set_title(f'{self.symbol} 价格热力图')
        ax1.set_xlabel('5分钟间隔')
        ax1.set_ylabel('小时')

        # 成交量热力图
        sns.heatmap(
            pivot_data['手数'],
            ax=ax2,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            xticklabels=[f'{i*5:02d}' for i in range(12)],
            yticklabels=[f'{i:02d}:00' for i in range(9, 16)]
        )
        ax2.set_title(f'{self.symbol} 成交量热力图')
        ax2.set_xlabel('5分钟间隔')
        ax2.set_ylabel('小时')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/trading_heatmap.png')
        plt.close()
    
    def calculate_statistics(self):
        """
        计算各项统计数据
        """
        # 整体统计
        self.overall_stats = {
            'total_volume': self.df['手数'].sum(),
            'price_change': self.df['成交价'].iloc[-1] - self.df['成交价'].iloc[0],
            'price_change_percentage': (self.df['成交价'].iloc[-1] / self.df['成交价'].iloc[0] - 1) * 100,
            'trade_types': self.df['买卖盘性质'].value_counts().to_dict(),
            'avg_price': self.df['成交价'].mean(),
            'max_price': self.df['成交价'].max(),
            'min_price': self.df['成交价'].min(),
        }
        
        # 5分钟统计
        self.stats_5min = self.df.groupby('5分钟间隔').agg({
            '成交价': ['mean', 'min', 'max'],
            '手数': 'sum',
            '买卖盘性质': lambda x: x.value_counts().to_dict()
        })
        
        # 15分钟统计
        self.stats_15min = self.df.groupby('15分钟间隔').agg({
            '成交价': ['mean', 'min', 'max'],
            '手数': 'sum',
            '买卖盘性质': lambda x: x.value_counts().to_dict()
        })
        
        return self.overall_stats, self.stats_5min, self.stats_15min
    
    def identify_key_moments(self):
        """
        识别关键交易时刻
        """
        # 计算价格变化
        self.df['价格变化'] = self.df['成交价'].diff()
        
        # 找出显著的价格变化
        price_std = self.df['价格变化'].std()
        self.significant_price_changes = self.df[abs(self.df['价格变化']) > price_std * 2]
        
        # 找出大成交量
        volume_std = self.df['手数'].std()
        self.large_volume_trades = self.df[self.df['手数'] > self.df['手数'].mean() + volume_std * 2]
        
        return {
            'significant_price_changes': self.significant_price_changes,
            'large_volume_trades': self.large_volume_trades
        }
    
    def generate_llm_analysis(self):
        """
        结合实时行情数据生成LLM分析报告
        """
        # 获取实时行情数据
        self.spot_data = self.get_stock_spot(self.symbol)
        
        prompt = f"""分析以下股票{self.symbol}的交易数据并生成详细报告：

实时市场数据：
- 股票名称：{self.spot_data.get('name')}
- 最新价：{self.spot_data.get('price')}
- 涨跌幅：{self.spot_data.get('change_percent')}%
- 换手率：{self.spot_data.get('turnover')}%
- 量比：{self.spot_data.get('volume_ratio')}
- 市盈率(动态)：{self.spot_data.get('pe')}
- 市净率：{self.spot_data.get('pb')}
- 总市值：{self.spot_data.get('total_value') / 100000000:.2f}亿
- 流通市值：{self.spot_data.get('circulating_value') / 100000000:.2f}亿
- 60日涨跌幅：{self.spot_data.get('change_60day')}%
- 年初至今涨跌幅：{self.spot_data.get('change_year')}%

分时交易统计：
- 总成交量：{self.overall_stats['total_volume']}手
- 价格变动：{self.overall_stats['price_change']:.2f} ({self.overall_stats['price_change_percentage']:.2f}%)
- 最高价：{self.overall_stats['max_price']:.2f}
- 最低价：{self.overall_stats['min_price']:.2f}
- 平均价：{self.overall_stats['avg_price']:.2f}
- 交易类型分布：
  - 中性盘：{self.overall_stats['trade_types'].get('中性盘', 0)}次
  - 买盘：{self.overall_stats['trade_types'].get('买盘', 0)}次
  - 卖盘：{self.overall_stats['trade_types'].get('卖盘', 0)}次

请从以下几个方面进行分析：
1. 市场估值与定位分析（基于PE、PB等指标）
2. 全天交易趋势与市场情绪分析
3. 买卖盘力量对比分析
4. 短期和中长期走势研判（结合60日、年初至今涨跌幅）
5. 成交量与换手率分析
6. 对后市的潜在影响及投资建议

请用专业的角度分析这些数据，生成一份详细的分析报告。"""

        _llm_factory = LLMFactory()
        llm_client = _llm_factory.get_instance()
        return llm_client.one_chat(prompt)

    def generate_markdown_report(self):
        """
        生成markdown格式的完整报告
        """
        llm_analysis = self.generate_llm_analysis()
        
        report = f"""# {self.symbol} {self.spot_data.get('name')}股票交易分析报告

## 市场概况
| 指标 | 数值 |
|------|------|
| 最新价 | {self.spot_data.get('price')} |
| 涨跌幅 | {self.spot_data.get('change_percent')}% |
| 换手率 | {self.spot_data.get('turnover')}% |
| 量比 | {self.spot_data.get('volume_ratio')} |
| 市盈率(动态) | {self.spot_data.get('pe')} |
| 市净率 | {self.spot_data.get('pb')} |
| 总市值 | {self.spot_data.get('total_value') / 100000000:.2f}亿 |
| 流通市值 | {self.spot_data.get('circulating_value') / 100000000:.2f}亿 |
| 60日涨跌幅 | {self.spot_data.get('change_60day')}% |
| 年初至今涨跌幅 | {self.spot_data.get('change_year')}% |

## 交易统计

### 全天交易概况
| 指标 | 数值 |
|------|------|
| 总成交量 | {self.overall_stats['total_volume']:,.0f}手 |
| 价格变动 | {self.overall_stats['price_change']:.2f} ({self.overall_stats['price_change_percentage']:.2f}%) |
| 最高价 | {self.overall_stats['max_price']:.2f} |
| 最低价 | {self.overall_stats['min_price']:.2f} |
| 平均价 | {self.overall_stats['avg_price']:.2f} |

### 交易类型分布
| 交易类型 | 次数 |
|----------|------|
| 中性盘 | {self.overall_stats['trade_types'].get('中性盘', 0)} |
| 买盘 | {self.overall_stats['trade_types'].get('买盘', 0)} |
| 卖盘 | {self.overall_stats['trade_types'].get('卖盘', 0)} |

## 重要交易时刻

### 显著价格变动
{self.significant_price_changes[['时间', '成交价', '价格变化', '手数', '买卖盘性质']].head(10).to_markdown()}

### 大额成交
{self.large_volume_trades[['时间', '成交价', '手数', '买卖盘性质']].head(10).to_markdown()}

## 交易热力图
![交易热力图](./output/trading_heatmap.png)

## AI分析报告
{llm_analysis}
"""
        # 保存报告
        with open(f'{self.output_dir}/trading_analysis.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def runner(symbol='601127'):
    """
    主函数
    """
    analyzer = StockAnalyzer(symbol)
    
    # 加载并处理数据
    analyzer.load_data()
    
    # 生成热力图
    analyzer.create_heatmap()
    
    # 计算统计数据
    analyzer.calculate_statistics()
    
    # 识别关键时刻
    analyzer.identify_key_moments()
    
    # 生成报告
    report = analyzer.generate_markdown_report()
    
    print(f"分析完成，报告已保存到 {analyzer.output_dir}/trading_analysis.md")
    return report

if __name__ == "__main__":
    runner()