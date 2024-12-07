from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

def calculate_volatility(df, windows=[5, 10, 20]):
    """Calculate historical volatility for different windows"""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    for window in windows:
        # Annualized volatility
        df[f'volatility_{window}d'] = log_returns.rolling(window=window).std() * np.sqrt(252)
    
    return df

def calculate_volume_ratio(df):
    """Calculate volume ratio indicators"""
    # 5-day average volume
    avg_5d_volume = df['volume'].rolling(window=5).mean()
    
    # Current volume to 5-day average volume ratio
    df['volume_ratio_5d'] = df['volume'] / avg_5d_volume
    
    # Calculate intraday volume concentration
    df['volume_concentration'] = df['volume'] / df['amount']
    
    # Relative volume (compared to previous day)
    df['relative_volume'] = df['volume'] / df['volume'].shift(1)
    
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    # Calculate middle band (20-day SMA)
    df['BB_middle'] = df['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    rolling_std = df['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    df['BB_upper'] = df['BB_middle'] + (rolling_std * num_std)
    df['BB_lower'] = df['BB_middle'] - (rolling_std * num_std)
    
    # Calculate bandwidth and %B
    df['BB_bandwidth'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_percent'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    return df

def prepare_features(df):
    """Prepare feature data"""
    df = df.copy()
    
    # Calculate technical indicators
    # Moving averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Add Bollinger Bands
    df = calculate_bollinger_bands(df)
    
    # Add volatility indicators
    df = calculate_volatility(df)
    
    # Add volume ratio indicators
    df = calculate_volume_ratio(df)
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    # Volume changes
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # Volatility indicators
    df['price_range'] = df['high'] - df['low']
    df['price_range_ma5'] = df['price_range'].rolling(window=5).mean()
    
    # Add lagged features
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    return df

def create_training_data(df):
    """Create training dataset"""
    # Feature columns
    feature_columns = [
        'open', 'high', 'low', 'volume', 'amount', 'amplitude', 
        'pct_change', 'turnover', 'MA5', 'MA10', 'MA20', 
        'momentum', 'volume_ratio', 'price_range', 'price_range_ma5',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_bandwidth', 'BB_percent',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'volume_ratio_5d', 'volume_concentration', 'relative_volume'
    ]
    
    # Add lagged features
    for i in range(1, 6):
        feature_columns.extend([f'close_lag_{i}', f'volume_lag_{i}'])
    
    X = df[feature_columns]
    y = df['close']
    
    return X, y

def train_model(X, y):
    """Train SVR model"""
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape y to 2D array for scaling
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Initialize and train model
    model = SVR(
        kernel='rbf',
        C=100,
        epsilon=0.1,
        gamma='scale'
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    
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
    start_date = end_date - pd.Timedelta(days=180)
    
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
    
    # Calculate feature correlations with target
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)
    
    print("\nTop 10 features by correlation with closing price:")
    print(correlations.head(10))

if __name__ == "__main__":
    runner()