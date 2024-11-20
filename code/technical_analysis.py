from core.utils.code_tools import code_tools
import pandas as pd
import numpy as np

stock_data_provider = code_tools["stock_data_provider"]
llm_client = code_tools["llm_client"]

# 获取中远海控的代码
stock_code = stock_data_provider.search_stock_code('中远海控')

# 获取2024年至今的日线数据
start_date = '20240101'
end_date = stock_data_provider.get_latest_trading_date()
df = stock_data_provider.get_historical_daily_data(stock_code, start_date, end_date)

# 重命名列名
df = df.rename(columns={
    '开盘': 'open',
    '收盘': 'close', 
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume'
})

# 计算MA指标
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA60'] = df['close'].rolling(window=60).mean()

# 计算MACD
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Hist'] = df['MACD'] - df['Signal']

# 计算KDJ
low_min = df['low'].rolling(window=9).min()
high_max = df['high'].rolling(window=9).max()
df['K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
df['D'] = df['K'].rolling(window=3).mean()
df['J'] = 3 * df['K'] - 2 * df['D']

# 计算RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

# 计算布林带
df['Middle'] = df['close'].rolling(window=20).mean()
std = df['close'].rolling(window=20).std()
df['Upper'] = df['Middle'] + (std * 2)
df['Lower'] = df['Middle'] - (std * 2)

# 计算成交量均线
df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
df['Volume_MA10'] = df['volume'].rolling(window=10).mean()

# 获取最新数据
latest_data = df.tail(1).iloc[0]

prompt = f"""请作为专业的技术分析专家，对中远海控(股票代码:{stock_code})的技术指标进行分析，并以JSON格式输出分析结果。请包含以下内容：

当前技术指标数据:
1. 收盘价: {latest_data['close']:.2f}
2. MA指标: MA5={latest_data['MA5']:.2f}, MA10={latest_data['MA10']:.2f}, MA20={latest_data['MA20']:.2f}
3. MACD指标: MACD={latest_data['MACD']:.2f}, Signal={latest_data['Signal']:.2f}, Hist={latest_data['Hist']:.2f}
4. KDJ指标: K={latest_data['K']:.2f}, D={latest_data['D']:.2f}, J={latest_data['J']:.2f}
5. RSI指标: {latest_data['RSI']:.2f}
6. 布林带: Upper={latest_data['Upper']:.2f}, Middle={latest_data['Middle']:.2f}, Lower={latest_data['Lower']:.2f}

请按照以下JSON格式输出分析结果:
{{
    "technical_analysis": {{
        "trend_analysis": "对趋势的综合分析，限制200字",
        "key_indicators": {{
            "moving_averages": "对均线系统的分析，限制100字",
            "macd": "对MACD指标的分析，限制100字", 
            "kdj": "对KDJ指标的分析，限制100字",
            "rsi": "对RSI指标的分析，限制100字",
            "bollinger": "对布林带指标的分析，限制100字"
        }},
        "risk_assessment": "风险评估，限制150字",
        "trading_suggestion": "交易建议，限制150字",
        "confidence_score": "分析可信度评分(0-100)"
    }}
}}"""

# 使用LLM进行分析
final_result = llm_client.one_chat(prompt)

# 保存结果
code_tools.add("output_result", final_result)
