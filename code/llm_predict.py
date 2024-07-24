import akshare as ak
import datetime
import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from matplotlib import font_manager
from core.llms.llm_factory import LLMFactory

llm_factory = LLMFactory()
llm_client = llm_factory.get_instance()
# 获取中远海控2024年的日线股票数据
# 中远海控的股票代码是 "601919.SH"
symbol = "sh601919"
# 使用stock_zh_a_daily函数获取数据
# 获取今天的日期
today = datetime.date.today()

# 将日期转换成yyyyMMdd格式
end_date = today.strftime("%Y%m%d")

# 使用stock_zh_a_daily函数获取数据
cosco_daily_data_2024 = ak.stock_zh_a_daily(symbol=symbol, start_date="20240101", end_date=end_date)

# 准备输入数据
input_data = cosco_daily_data_2024[['date', 'close']].set_index('date')

# 定义数据处理函数
def data_processor(data):
    return data.asfreq('D').fillna(method='ffill').iloc[-30:]  # 使用最近30天的数据

# 定义响应处理函数
def response_processor(response, num_of_predict):
    # 使用正则表达式提取数字
    numbers = re.findall(r'\d+\.\d+', response)
    predictions = [float(num) for num in numbers[-num_of_predict:]]
    return pd.Series(predictions, name='close', index=pd.date_range(start=input_data.index[-1] + pd.Timedelta(days=1), periods=num_of_predict))

# 使用llm_client进行预测
predictions = llm_client.predict(input_data, num_of_predict=5, data_processor=data_processor, response_processor=response_processor)



# 确保output文件夹存在
os.makedirs('output', exist_ok=True)

# 设置中文字体
def configure_matplotlib_for_chinese():
    import platform
    system = platform.system()
    if system == 'Windows':
        font_name = 'SimHei'
    elif system == 'Darwin':
        font_name = 'STHeiti'
    else:  # For Linux
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
        else:
            raise FileNotFoundError(f"Font file not found: {font_path}")
    
    # Set the font properties
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# 运行配置函数
configure_matplotlib_for_chinese()

# 访问之前步骤的数据
historical_data = cosco_daily_data_2024
prediction_data = predictions

# 合并历史数据和预测数据
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data.set_index('date', inplace=True)
combined_data = pd.concat([historical_data['close'], prediction_data['close']])

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(historical_data.index, historical_data['close'], label='历史数据')
plt.plot(prediction_data.index, prediction_data['close'], color='red', label='预测数据')
plt.title('中远海控股价历史数据和预测')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)

# 保存图表
file_name = f"output/{uuid.uuid4()}.png"
plt.savefig(file_name)
plt.close()

# 计算一些统计数据
last_historical_price = historical_data['close'].iloc[-1]
first_prediction_price = prediction_data['close'].iloc[0]
prediction_change = (prediction_data['close'].iloc[-1] - first_prediction_price) / first_prediction_price * 100

# 准备返回值
results = []
results.append(f"![中远海控股价历史数据和预测图表]({file_name})")
results.append("主要发现：")
results.append(f"1. 最后一个历史数据点的收盘价为 {last_historical_price:.2f}")
results.append(f"2. 预测的第一天收盘价为 {first_prediction_price:.2f}")
results.append(f"3. 预测期间的价格变化率为 {prediction_change:.2f}%")
results.append(f"4. 预测显示未来5天的股价呈现{('上升' if prediction_change > 0 else '下降')}趋势")

# 使用LLM API进行分析
llm_client = llm_factory.get_instance()
prompt = f"根据以下数据分析中远海控未来5天的股价趋势：最后历史价格 {last_historical_price:.2f}，预测5天后价格 {prediction_data['close'].iloc[-1]:.2f}，预测期间变化率 {prediction_change:.2f}%。请给出简短的市场分析和投资建议。"
response = llm_client.one_chat(prompt)

results.append("5. LLM分析结果：")
results.append(response)

# 将结果保存到analysis_result变量
analysis_result = "\n".join(results)


# 输出最终结果
prompt=f"""
用户想要获取中远海控2024年的日线股票数据，并使用LLM进行数据分析，给出分析和预测。

请根据以下数据，使用LLM进行数据分析，并给出市场分析和投资建议。

数据：
{historical_data.tail(30).to_markdown()}

LLM分析结果：
{analysis_result}

请以markdown格式输出最终的结果:
"""
response = llm_client.one_chat(prompt)
# 创建或更新output.md文件
with open("./markdowns/601919_predict.md", "w", encoding="utf-8") as f:
    f.write(response)