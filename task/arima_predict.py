from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from core.llms.llm_factory import LLMFactory
from dealer.stock_data_provider import StockDataProvider

llm_client = LLMFactory().get_instance()
stock_data_provider = StockDataProvider(llm_client)

def search_index_code(index_name):
    try:
        index_code = stock_data_provider.search_index_code(index_name)
        return index_code
    except Exception as e:
        print(f"Error in search_index_code: {e}")
        return None

def get_index_data(index_code, start_date, end_date):
    try:
        data = stock_data_provider.get_index_data([index_code], start_date, end_date)
        index_data = data[index_code]
        # Standardize column names
        index_data = index_data.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        # Set date index
        index_data['date'] = pd.to_datetime(index_data['date'])
        index_data.set_index('date', inplace=True)
        return index_data
    except Exception as e:
        print(f"Error in get_index_data: {e}")
        return None

def check_stationarity(series):
    """
    检查时间序列的平稳性，返回ADF检验结果和需要的差分阶数
    """
    max_diff = 2  # 最大差分次数
    d = 0
    adf_result = adfuller(series)
    
    # 如果p值大于0.05，说明序列不平稳，需要进行差分
    while adf_result[1] > 0.05 and d < max_diff:
        d += 1
        diff_series = series.diff(d).dropna()
        adf_result = adfuller(diff_series)
    
    return adf_result, d

def prepare_data(df):
    """准备数据，只保留收盘价序列"""
    return df['close']

def train_arima(series):
    """
    训练ARIMA模型，自动选择最佳参数
    """
    # 检查平稳性并确定d参数
    _, d = check_stationarity(series)
    
    # 设置参数范围
    p_range = range(0, 3)
    q_range = range(0, 3)
    
    best_aic = float('inf')
    best_params = None
    best_model = None
    
    # 网格搜索最佳参数
    for p in p_range:
        for q in q_range:
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, d, q)
                    best_model = results
            except:
                continue
    
    if best_model is None:
        raise Exception("无法找到合适的ARIMA模型参数")
    
    print(f"\nBest ARIMA parameters (p,d,q): {best_params}")
    print(f"AIC: {best_aic:.2f}")
    
    # 评估模型
    predictions = best_model.get_prediction(start=len(series)-30)
    predictions = predictions.predicted_mean
    
    rmse = np.sqrt(mean_squared_error(series[-30:], predictions))
    print(f"Last 30 days RMSE: {rmse:.2f}")
    
    return best_model

def predict_next_day(model, series):
    """预测下一个交易日的收盘价"""
    forecast = model.get_forecast(steps=1)
    mean_forecast = forecast.predicted_mean.values[0]
    conf_int = forecast.conf_int()
    
    print(f"\nNext trading day prediction:")
    print(f"Point forecast: {mean_forecast:.2f}")
    print(f"95% Confidence Interval: [{conf_int.iloc[0,0]:.2f}, {conf_int.iloc[0,1]:.2f}]")
    
    return mean_forecast, conf_int

def runner(symbol:str = "上证指数"):
    # 获取指数数据
    index_code = search_index_code(symbol)
    if index_code is None:
        return
    
    # 获取近期数据
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=200)
    
    df = get_index_data(index_code, start_date, end_date)
    if df is None:
        return
    
    # 准备数据
    series = prepare_data(df)
    
    # 训练模型
    model = train_arima(series)
    
    # 预测下一个交易日
    predict_next_day(model, series)

if __name__ == "__main__":
    runner()