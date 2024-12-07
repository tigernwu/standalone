import pandas as pd
from core.utils.code_tools import code_tools
from core.utils.log import logger
from core.utils.json_from_text import extract_json_from_text
import ta  # 导入技术指标库

# 初始化工具
def initialize_tools():
    from dealer.stock_data_provider import StockDataProvider
    from core.llms.llm_factory import LLMFactory
    llm_client = LLMFactory().get_instance("SimpleDeepSeekClient")
    stock_data_provider = StockDataProvider(llm_client)
    return stock_data_provider, llm_client

stock_data_provider, llm_client = initialize_tools()

# 输入数据
symbol = '上证指数'
start_date_str = '20241020'
end_date_str = '20241231'  # 假设一个结束日期

# 将日期字符串转换为 datetime 对象
start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
end_date = pd.to_datetime(end_date_str, format='%Y%m%d')

# 获取上证指数历史数据
def get_shanghai_index_data(symbol, start_date, end_date):
    try:
        index_code = stock_data_provider.search_index_code(symbol)
        # 调用API获取历史数据
        data_dict = stock_data_provider.get_index_data([index_code], start_date_str, end_date_str)
        data = data_dict[index_code]
        
        if data.empty:
            logger.warning("获取的数据为空。")
            return
        
        # 确保日期字段是 datetime 类型
        data['日期'] = pd.to_datetime(data['日期'])
        
        # 筛选 start_date 及之后的数据
        filtered_data = data[data['日期'] >= start_date]
        
        # 将结果存储到 code_tools
        code_tools.add_var('上证指数历史数据', filtered_data)
        
        logger.info("数据获取成功，已存储到 code_tools。")
    except ValueError as ve:
        logger.error(f"值错误: {ve}")
    except TypeError as te:
        logger.error(f"类型错误: {te}")
    except KeyError as ke:
        logger.error(f"键错误: {ke}")
    except Exception as e:
        logger.error(f"未知错误: {e}")

# 计算技术指标
def calculate_technical_indicators():
    try:
        # 从 code_tools 中获取上证指数历史数据
        data = code_tools.get("上证指数历史数据")
        
        # 确保数据按日期排序
        data = data.sort_values(by='日期')
        
        # 计算 RSI（相对强弱指数）
        data['RSI'] = ta.momentum.RSIIndicator(close=data['收盘'], window=14).rsi()
        
        # 计算 CCI（顺势指标）
        data['CCI'] = ta.trend.CCIIndicator(high=data['最高'], low=data['最低'], close=data['收盘'], window=20).cci()
        
        # 计算布林带
        bollinger = ta.volatility.BollingerBands(close=data['收盘'], window=20, window_dev=2)
        data['bollinger_hband'] = bollinger.bollinger_hband()
        data['bollinger_lband'] = bollinger.bollinger_lband()
        
        # 计算支撑位和阻力位（这里我们用简单的移动平均线来表示）
        data['SMA20'] = ta.trend.SMAIndicator(close=data['收盘'], window=20).sma_indicator()
        data['SMA50'] = ta.trend.SMAIndicator(close=data['收盘'], window=50).sma_indicator()
        
        # 将计算好的数据存储到 code_tools
        code_tools.add_var('技术指标数据', data)
        
        logger.info("技术指标计算成功，已存储到 code_tools。")
    except Exception as e:
        logger.error(f"计算技术指标时出现错误: {e}")

# 计算波动率
def calculate_volatility():
    try:
        # 从技术指标数据中获取数据
        data = code_tools.get("技术指标数据")
        
        # 提取每日收盘价
        close_prices = data['收盘']
        
        # 计算每日收益率
        daily_returns = close_prices.pct_change().dropna()
        
        # 计算收益率的标准差作为波动率
        volatility = daily_returns.rolling(window=20).std()
        
        # 对齐日期
        volatility = volatility.dropna()
        volatility_data = pd.DataFrame({
            '日期': data.loc[volatility.index, '日期'],
            '波动率': volatility.values
        })
        
        # 将波动率数据存储到 code_tools
        code_tools.add_var('波动率', volatility_data)
        
        logger.info("波动率计算成功，已存储到 code_tools。")
    except Exception as e:
        logger.error(f"计算波动率时出现错误: {e}")

# 计算波动率变化
def calculate_volatility_change():
    try:
        # 从 code_tools 中获取波动率数据
        volatility_data = code_tools.get("波动率")
        
        # 确保波动率数据是按日期排序的，并且没有缺失值
        volatility_data = volatility_data.sort_values(by='日期').dropna()
        
        # 提取波动率列
        volatility_series = volatility_data['波动率']
        
        # 计算每日波动率的变化量
        volatility_change = volatility_series.diff()
        
        # 创建波动率变化的数据框
        volatility_change_data = pd.DataFrame({
            '日期': volatility_data['日期'],
            '波动率变化': volatility_change
        }).dropna()
        
        # 将计算结果存储到 code_tools 中
        code_tools.add_var('波动率变化', volatility_change_data)
        
        logger.info("波动率变化计算成功，已存储到 code_tools。")
    except Exception as e:
        logger.error(f"计算波动率变化时出现错误: {e}")

# 计算交易量变化
def calculate_volume_change():
    try:
        # 从技术指标数据中获取每日交易量数据
        data = code_tools.get("技术指标数据")
        volume_data = data[['日期', '成交量']]
        
        # 确保交易量数据是按日期排序的，且没有缺失值
        volume_data = volume_data.sort_values(by='日期').dropna()
        
        # 计算每日交易量的变化量
        volume_series = volume_data['成交量']
        volume_change = volume_series.diff()
        
        # 创建交易量变化的数据框
        volume_change_data = pd.DataFrame({
            '日期': volume_data['日期'],
            '交易量变化': volume_change
        }).dropna()
        
        # 将计算结果存储到 code_tools
        code_tools.add_var('交易量变化', volume_change_data)
        
        logger.info("交易量变化计算成功，已存储到 code_tools。")
    except Exception as e:
        logger.error(f"计算交易量变化时出现错误: {e}")

# 准备输入特征
def prepare_features():
    try:
        # 从 code_tools 中获取技术指标数据
        data = code_tools.get("技术指标数据")
        
        # 获取波动率和波动率变化数据
        volatility = code_tools.get("波动率")
        volatility_change = code_tools.get("波动率变化")
        
        # 获取交易量变化数据
        volume_change = code_tools.get("交易量变化")
        
        # 合并所有特征
        features = data.merge(volatility, on='日期', how='inner')
        features = features.merge(volatility_change, on='日期', how='inner')
        features = features.merge(volume_change, on='日期', how='inner')
        
        # 删除任何缺失值的行
        features = features.dropna()
        
        # 将准备好的特征数据存储到 code_tools
        code_tools.add_var('输入特征', features)
        
        logger.info("输入特征准备成功，已存储到 code_tools。")
    except Exception as e:
        logger.error(f"准备输入特征时出现错误: {e}")

# 使用 LLM 预测下一个交易日的点位
def predict_next_trading_day():
    try:
        # 从 code_tools 中获取输入特征
        features = code_tools.get("输入特征")
        
        # 选择最近几天的数据，例如最近5天
        recent_days = 5
        recent_data = features.tail(recent_days)
        
        # 准备最近几天的特征数据
        recent_features = recent_data.to_dict(orient='records')
        
        # 构建提示，包含预测指导
        prompt = f"""请根据以下最近 {recent_days} 天的特征数据预测下一个交易日的上证指数收盘点位。
请参考以下指导：
- 考虑波动率和波动率变化的趋势对指数的影响。
- 关注交易量和交易量变化是否预示着市场的活跃度变化。
- 分析 RSI、CCI 等技术指标的数值和趋势。
- 观察布林带上轨和下轨，以及收盘价与布林带的关系。
- 结合短期和长期移动平均线（如 SMA20 和 SMA50）的交叉情况。
请以 JSON 格式返回结果，例如：{{"预测点位": 1234.56}}
特征数据如下：
{recent_features}"""

        # 调用 LLM 进行预测
        response = llm_client.one_chat(prompt)
        
        # 提取预测结果
        if response:
            prediction_json = extract_json_from_text(response)
            prediction = prediction_json.get('预测点位')
            if prediction:
                # 将预测结果存储到 code_tools
                code_tools.add_var('下一个交易日的预测点位', prediction)
                logger.info(f"下一个交易日的预测点位: {prediction}")
            else:
                logger.warning("预测结果中没有找到'预测点位'字段。")
        else:
            logger.warning("LLM响应为空。")
    except Exception as e:
        logger.error(f"预测下一个交易日点位时出现错误: {e}")

def runner(index_name='上证指数', start_date_str1='20240820', end_date_str1='20241231'):
    global symbol, start_date_str, end_date_str, start_date, end_date
    symbol = index_name
    start_date_str = start_date_str1
    end_date_str = end_date_str1
    start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
    end_date = pd.to_datetime(end_date_str, format='%Y%m%d')
    
    # 调用函数获取数据
    get_shanghai_index_data(symbol, start_date, end_date)
    
    # 计算技术指标
    calculate_technical_indicators()
    
    # 计算波动率
    calculate_volatility()
    
    # 计算波动率变化
    calculate_volatility_change()
    
    # 计算交易量变化
    calculate_volume_change()
    
    # 准备输入特征
    prepare_features()
    
    # 预测下一个交易日的点位
    predict_next_trading_day()

# 运行代码
if __name__ == "__main__":
    runner()
