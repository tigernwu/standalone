from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta

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

def calculate_technical_indicators(df):
    """Calculate technical indicators using TA"""
    df = df.copy()
    
    # Trend Indicators
    # SMA and EMA
    for period in [5, 10, 20, 30, 60]:
        df[f'SMA_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        df[f'EMA_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()
    
    # Momentum Indicators
    # RSI for multiple periods
    for period in [6, 12, 24]:
        df[f'RSI_{period}'] = ta.momentum.rsi(df['close'], window=period)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['STOCH_k'] = stoch.stoch()
    df['STOCH_d'] = stoch.stoch_signal()
    
    # ROC for multiple periods
    for period in [12, 24, 48]:
        df[f'ROC_{period}'] = ta.momentum.roc(df['close'], window=period)
    
    # TSI
    df['TSI'] = ta.momentum.tsi(df['close'])
    
    # Volatility Indicators
    # Bollinger Bands
    for period in [20, 30]:
        bollinger = ta.volatility.BollingerBands(df['close'], window=period)
        df[f'BB_high_{period}'] = bollinger.bollinger_hband()
        df[f'BB_mid_{period}'] = bollinger.bollinger_mavg()
        df[f'BB_low_{period}'] = bollinger.bollinger_lband()
        df[f'BB_width_{period}'] = bollinger.bollinger_wband()
    
    # ATR for multiple periods
    for period in [14, 28]:
        df[f'ATR_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    
    # Ulcer Index
    df['UI'] = ta.volatility.ulcer_index(df['close'])
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['FI'] = ta.volume.force_index(df['close'], df['volume'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    
    # Custom features
    df['daily_return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']/df['close'].shift(1))
    
    # Volatility features
    for window in [5, 10, 20, 30]:
        df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
        df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
        df[f'price_momentum_{window}'] = df['close'].pct_change(window)
    
    return df

def prepare_features(df):
    """Prepare feature data with advanced feature engineering"""
    df = df.copy()
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Add lagged features
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
        df[f'return_lag_{i}'] = df['daily_return'].shift(i)
    
    # Add rolling mean features
    for window in [3, 5, 10]:
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
    
    # Price difference features
    df['price_diff'] = df['close'] - df['open']
    df['high_low_diff'] = df['high'] - df['low']
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    return df

def create_training_data(df):
    """Create training dataset with feature selection"""
    # Exclude date and target variable from features
    exclude_columns = ['close']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns]
    y = df['close']
    
    return X, y

def train_model(X, y):
    """Train GBDT model with staged predictions and advanced evaluation"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, shuffle=False
    )
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.01, shuffle=False
    )
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Initialize GBDT model with optimized parameters
    model = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=4,
        subsample=0.8,
        max_features=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-4
    )
    
    # Train model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    train_pred_scaled = model.predict(X_train_scaled)
    val_pred_scaled = model.predict(X_val_scaled)
    test_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform predictions
    train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    print(feature_importance.head(15))
    
    return model, scaler_X, scaler_y, X_test, y_test, metrics

def predict_next_day(model, df, scaler_X, scaler_y):
    """Predict next trading day's closing price with confidence interval"""
    last_row = df.iloc[[-1]]
    X_pred = create_training_data(last_row)[0]
    
    X_pred_scaled = scaler_X.transform(X_pred)
    
    # Make prediction
    prediction_scaled = model.predict(X_pred_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    
    # Calculate prediction interval (using std of historical predictions)
    std = model.predict(X_pred_scaled).std()
    confidence_interval = 1.96 * std  # 95% confidence interval
    
    return prediction, confidence_interval

def runner(symbol:str = "上证指数"):
    # Get index data
    index_code = search_index_code(symbol)
    if index_code is None:
        return
    
    # Get data for the last 200 days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=120)
    
    df = get_index_data(index_code, start_date, end_date)
    if df is None:
        return
    
    # Prepare data
    df_processed = prepare_features(df)
    X, y = create_training_data(df_processed)
    
    # Train and evaluate model
    model, scaler_X, scaler_y, X_test, y_test, metrics = train_model(X, y)
    
    # Predict next trading day
    prediction, confidence_interval = predict_next_day(model, df_processed, scaler_X, scaler_y)
    
    print(f"\nNext trading day prediction:")
    print(f"Predicted closing price: {prediction:.2f}")
    print(f"95% Confidence Interval: [{prediction-confidence_interval:.2f}, {prediction+confidence_interval:.2f}]")
    
    # Calculate prediction change
    last_close = df['close'].iloc[-1]
    predicted_change = ((prediction - last_close) / last_close) * 100
    print(f"Predicted change: {predicted_change:.2f}%")

if __name__ == "__main__":
    runner()