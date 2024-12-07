from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

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
        # 统一列名
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
        # 设置日期索引
        index_data['date'] = pd.to_datetime(index_data['date'])
        index_data.set_index('date', inplace=True)
        return index_data
    except Exception as e:
        print(f"Error in get_index_data: {e}")
        return None



def prepare_features(df):
    """准备特征数据"""
    df = df.copy()
    
    # 计算技术指标
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 价格动量
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    # 成交量变化
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 波动性指标
    df['price_range'] = df['high'] - df['low']
    df['price_range_ma5'] = df['price_range'].rolling(window=5).mean()
    
    # 添加滞后特征
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    
    # 删除包含NaN的行
    df.dropna(inplace=True)
    
    return df

def create_training_data(df):
    """创建训练数据集"""
    # 特征列
    feature_columns = [
        'open', 'high', 'low', 'volume', 'amount', 'amplitude', 
        'pct_change', 'turnover', 'MA5', 'MA10', 'MA20', 
        'momentum', 'volume_ratio', 'price_range', 'price_range_ma5'
    ]
    
    # 添加滞后特征
    for i in range(1, 6):
        feature_columns.extend([f'close_lag_{i}', f'volume_lag_{i}'])
    
    X = df[feature_columns]
    y = df['close']
    
    return X, y

def train_model(X, y):
    """训练随机森林模型"""
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # 初始化并训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"训练集RMSE: {train_rmse:.2f}")
    print(f"测试集RMSE: {test_rmse:.2f}")
    print(f"测试集R2分数: {test_r2:.4f}")
    
    return model, X_test, y_test

def predict_next_day(model, df):
    """预测下一个交易日的收盘价"""
    # 准备最后一行数据作为预测输入
    last_row = df.iloc[[-1]]
    X_pred = create_training_data(last_row)[0]
    
    # 预测
    prediction = model.predict(X_pred)[0]
    
    return prediction

def runner(symbol:str = "上证指数"):
    # 获取上证指数数据
    index_code = search_index_code(symbol)
    if index_code is None:
        return
    
    # 获取近一年数据
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    
    df = get_index_data(index_code, start_date, end_date)
    if df is None:
        return
    
    # 准备数据
    df_processed = prepare_features(df)
    X, y = create_training_data(df_processed)
    
    # 训练模型
    model, X_test, y_test = train_model(X, y)
    
    # 预测下一个交易日
    next_day_prediction = predict_next_day(model, df_processed)
    print(f"\n下一个交易日预测收盘价: {next_day_prediction:.2f}")
    
    # 输出重要特征
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排名前10:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    runner()