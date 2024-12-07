from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import ta

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
    # SMA
    df['SMA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # EMA
    df['EMA5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['EMA10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    
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
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'])
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['STOCH_k'] = stoch.stoch()
    df['STOCH_d'] = stoch.stoch_signal()
    
    # ROC
    df['ROC'] = ta.momentum.roc(df['close'])
    
    # TSI (True Strength Index)
    df['TSI'] = ta.momentum.tsi(df['close'])
    
    # Volatility Indicators
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    df['BB_low'] = bollinger.bollinger_lband()
    df['BB_width'] = bollinger.bollinger_wband()
    
    # Average True Range
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Ulcer Index
    df['UI'] = ta.volatility.ulcer_index(df['close'])
    
    # Volume Indicators
    # On-Balance Volume
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Force Index
    df['FI'] = ta.volume.force_index(df['close'], df['volume'])
    
    # Volume Weighted Average Price
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    
    # Money Flow Index
    df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    
    # Custom Volatility Ratios
    df['daily_return'] = df['close'].pct_change()
    df['volatility_5d'] = df['daily_return'].rolling(window=5).std() * np.sqrt(252)
    df['volatility_10d'] = df['daily_return'].rolling(window=10).std() * np.sqrt(252)
    
    return df

def prepare_features(df):
    """Prepare feature data"""
    df = df.copy()
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Add lagged features
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
        df[f'return_lag_{i}'] = df['daily_return'].shift(i)
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    return df

def create_training_data(df):
    """Create training dataset"""
    # Feature columns
    feature_columns = [
        'open', 'high', 'low', 'volume', 'amplitude', 'pct_change', 'turnover',
        'SMA5', 'SMA10', 'SMA20', 'EMA5', 'EMA10', 'EMA20',
        'MACD', 'MACD_signal', 'MACD_diff',
        'ADX', 'ADX_pos', 'ADX_neg',
        'RSI', 'STOCH_k', 'STOCH_d', 'ROC', 'TSI',
        'BB_high', 'BB_mid', 'BB_low', 'BB_width',
        'ATR', 'UI', 'OBV', 'FI', 'VWAP', 'MFI',
        'volatility_5d', 'volatility_10d'
    ]
    
    # Add lagged features
    for i in range(1, 6):
        feature_columns.extend([f'close_lag_{i}', f'volume_lag_{i}', f'return_lag_{i}'])
    
    X = df[feature_columns]
    y = df['close']
    
    return X, y

def train_model(X, y):
    """Train LightGBM model with advanced features"""
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Further split training data to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape y to 2D array for scaling
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Initialize LightGBM model with optimized parameters
    model = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=20,  # LightGBM specific
        max_depth=-1,   # Let LightGBM optimize the depth
        min_child_samples=5,
        min_child_weight=0.01,
        subsample=0.8,
        colsample_bytree=0.94,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1, # L2 regularization
        n_jobs=-1,      # Use all CPU cores
        importance_type='split',
        random_state=22,
        verbose=0
    )
    
    # Train model with early stopping
    model.fit(
        X_train_scaled, y_train_scaled,
        eval_set=[(X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled)],
        eval_metric='rmse',
        callbacks=[
            # LightGBM specific callbacks for early stopping
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100) 
        ]
    )
    
    # Make predictions and inverse transform
    train_pred_scaled = model.predict(X_train_scaled)
    test_pred_scaled = model.predict(X_test_scaled)
    
    train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
    
    # Evaluate model
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Best iteration found: {model.best_iteration_}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return model, scaler_X, scaler_y, X_test, y_test

def predict_next_day(model, df, scaler_X, scaler_y):
    """Predict next trading day's closing price"""
    # Prepare last row data as prediction input
    last_row = df.iloc[[-1]]
    X_pred = create_training_data(last_row)[0]
    
    # Scale the input
    X_pred_scaled = scaler_X.transform(X_pred)
    
    # Predict and inverse transform
    prediction_scaled = model.predict(X_pred_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    
    return prediction

def runner(symbol:str = "上证指数"):
    # Get index data
    index_code = search_index_code(symbol)
    if index_code is None:
        return
    
    # Get last year's data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=150)
    
    df = get_index_data(index_code, start_date, end_date)
    if df is None:
        return
    
    # Prepare data
    df_processed = prepare_features(df)
    X, y = create_training_data(df_processed)
    
    # Train model
    model, scaler_X, scaler_y, X_test, y_test = train_model(X, y)
    
    # Predict next trading day
    next_day_prediction = predict_next_day(model, df_processed, scaler_X, scaler_y)
    print(f"\nNext trading day predicted closing price: {next_day_prediction:.2f}")

if __name__ == "__main__":
    runner()