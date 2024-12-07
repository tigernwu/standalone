from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from core.llms.llm_factory import LLMFactory
from dealer.stock_data_provider import StockDataProvider

llm_client = LLMFactory().get_instance()
stock_data_provider = StockDataProvider(llm_client)

def search_index_code(index_name):
    """Search for index code by name"""
    try:
        index_code = stock_data_provider.search_index_code(index_name)
        return index_code
    except Exception as e:
        print(f"Error in search_index_code: {e}")
        return None

def get_index_data(index_code, start_date, end_date):
    """Get historical index data"""
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
        # Convert date format
        index_data['date'] = pd.to_datetime(index_data['date'])
        return index_data
    except Exception as e:
        print(f"Error in get_index_data: {e}")
        return None

def prepare_prophet_data(df):
    """Prepare data for Prophet model"""
    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df[['date', 'close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Add additional features
    prophet_df['volume'] = df['volume']
    
    return prophet_df

def train_prophet_model(df, forecast_days=30):
    """Train Prophet model and make predictions"""
    # Initialize Prophet model with parameters
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility of the trend
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'  # For financial data, multiplicative seasonality often works better
    )
    
    # Add additional regressors if available
    if 'volume' in df.columns:
        model.add_regressor('volume')
    
    # Fit the model
    model.fit(df)
    
    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add regressor values for future dates
    if 'volume' in df.columns:
        # For simplicity, use the mean volume for future predictions
        future['volume'] = df['volume'].mean()
    
    # Make predictions
    forecast = model.predict(future)
    
    return model, forecast

def evaluate_model(df, forecast):
    """Evaluate model performance"""
    # Calculate metrics for the training period
    actual = df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    print(f"Model Performance Metrics:")
    print(f"MAPE: {mape:.2%}")
    print(f"RMSE: {rmse:.2f}")

def plot_results(df, forecast, title="Stock Index Forecast"):
    """Create interactive plot using plotly"""
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        name='Actual',
        line=dict(color='blue')
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Predicted',
        line=dict(color='red')
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )

    return fig

def runner(symbol:str = "上证指数", forecast_days:int = 30):
    """Main function to run the prediction pipeline"""
    # Get index data
    index_code = search_index_code(symbol)
    if index_code is None:
        return
    
    # Get historical data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=200)  
    
    df = get_index_data(index_code, start_date, end_date)
    if df is None:
        return
    
    # Prepare data for Prophet
    prophet_df = prepare_prophet_data(df)
    
    # Train model and get forecast
    model, forecast = train_prophet_model(prophet_df, forecast_days)
    
    # Evaluate model
    evaluate_model(prophet_df, forecast)
    
    # Create and display plot
    fig = plot_results(prophet_df, forecast, f"{symbol} Forecast")
    
    # Print next day prediction
    next_day = forecast[forecast['ds'] > df['date'].max()].iloc[0]
    print(f"\nNext trading day prediction:")
    print(f"Date: {next_day['ds'].strftime('%Y-%m-%d')}")
    print(f"Predicted Close: {next_day['yhat']:.2f}")
    print(f"Lower Bound: {next_day['yhat_lower']:.2f}")
    print(f"Upper Bound: {next_day['yhat_upper']:.2f}")
    
    return model, forecast, fig

if __name__ == "__main__":
    model, forecast, fig = runner()
    # To display the plot in a notebook environment:
    # fig.show()