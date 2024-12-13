from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from core.stock.baidu_news import BaiduFinanceAPI
from datetime import datetime, timedelta
import core.stock.tushare_provider as tp
from core.utils.shared_cache import cache,lg_limiter,bg_limiter,cd_limiter,sn_limiter
import akshare as ak
from datetime import datetime, timedelta
baidu_news_api = BaiduFinanceAPI()
from core.stock.stock_matcher import StockMatcher
matcher = StockMatcher()
from cachetools import cached,LFUCache
from core.utils.function_cache import FunctionCache
function_cache = FunctionCache()
import talib as ta
from core.stock.ts_code_matcher import TsCodeMatcher
from core.utils.log import logger

analyze_cache=FunctionCache("./data/analyze_single_stock_cache.db")

def get_minute_data(
                    symbol: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    period: str = "1",
                    adjust: str = "") -> pd.DataFrame:
    """
    Fetch minute-level K-line data for Chinese A-share stocks from East Money.
    
    Args:
        symbol (str): Stock symbol, e.g., "000001"
        start_date (str, optional): Start datetime in format "YYYY-MM-DD HH:MM:SS"
                                Default fetches earliest available data
        end_date (str, optional): End datetime in format "YYYY-MM-DD HH:MM:SS"
                                Default fetches latest available data
        period (str): Time period, options: {"1", "5", "15", "30", "60"}
                    Note: 1-min data only available for last 5 trading days
        adjust (str): Price adjustment, options: {"", "qfq", "hfq"}
                    "": No adjustment, "qfq": Forward adjusted, "hfq": Backward adjusted
                    Note: 1-min data cannot be adjusted
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - timestamp: datetime object
            - open: Opening price
            - close: Closing price
            - high: Highest price
            - low: Lowest price
            - volume: Trading volume (in lots)
            - amount: Trading amount
            - vwap: Volume weighted average price (for 1-min data)
            - change_pct: Price change percentage (for >1-min data)
            - change: Price change (for >1-min data)
            - amplitude: Price amplitude percentage (for >1-min data)
            - turnover: Turnover rate percentage (for >1-min data)
    
    Raises:
        ValueError: If invalid period or adjust parameters are provided
    """
    import akshare as ak
    # Set default datetime range if not provided
    start_date = start_date or "1979-09-01 09:32:00"
    end_date = end_date or "2222-01-01 09:32:00"
    
    # Validate inputs
    valid_periods = {"1", "5", "15", "30", "60"}
    valid_adjusts = {"", "qfq", "hfq"}
    
    if period not in valid_periods:
        raise ValueError(f"Period must be one of {valid_periods}")
    if adjust not in valid_adjusts:
        raise ValueError(f"Adjust must be one of {valid_adjusts}")
        
    # Fetch data
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        adjust=adjust
    )
    
    # Rename columns based on period
    if period == "1":
        column_mappings = {
            "时间": "timestamp",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "均价": "vwap"
        }
    else:
        column_mappings = {
            "时间": "timestamp",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "涨跌幅": "change_pct",
            "涨跌额": "change",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "换手率": "turnover"
        }
    
    # Rename columns and convert timestamp
    df = df.rename(columns=column_mappings)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def get_daily_data(symbol: str, start_date: str = None, end_date: str = None, 
                   period: str = 'daily', adjust: str = '') -> pd.DataFrame:
    """
    Retrieve daily stock data from East Money (东方财富) with customizable date range.
    
    Args:
        symbol (str): Stock symbol (e.g., '000001' for Ping An Bank)
        start_date (str, optional): Start date in 'YYYYMMDD' format. 
            Defaults to 365 days before end_date if None
        end_date (str, optional): End date in 'YYYYMMDD' format. 
            Defaults to current date if None
        period (str, optional): Data frequency. Defaults to 'daily'.
            Options: {'daily', 'weekly', 'monthly'}
        adjust (str, optional): Price adjustment method. Defaults to '' (no adjustment).
            Options: {'': no adjustment, 'qfq': forward adjustment, 'hfq': backward adjustment}
    
    Returns:
        pandas.DataFrame: Stock data with following columns:
            - date: Trading date
            - symbol: Stock symbol
            - open: Opening price
            - close: Closing price
            - high: Highest price
            - low: Lowest price
            - volume: Trading volume (in lots of 100 shares)
            - amount: Trading amount (in CNY)
            - amplitude: Price amplitude (%)
            - pct_change: Price change percentage (%)
            - price_change: Price change amount (CNY)
            - turnover: Turnover rate (%)
    """
    import akshare as ak
    # Handle default end_date
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # Handle default start_date
    if start_date is None:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime('%Y%m%d')
    
    # Get data from akshare
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust
    )
    
    # Rename columns to English
    column_mapping = {
        '日期': 'date',
        '股票代码': 'symbol',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_change',
        '涨跌额': 'price_change',
        '换手率': 'turnover'
    }
    
    df = df.rename(columns=column_mapping)
    
    return df

def get_last_trading_dates(days: int = 400) -> Tuple[str, str]:
    """
    返回起始日期，结束日期
    """
    dates = tp.get_last_trading_dates(days)
    return dates[-1],dates[0]

def get_stock_daily(symbol: str, start_date, end_date) -> pd.DataFrame:
    """获取股票日线行情数据。

    Args:
        symbol (str): 股票代码，例如 '601138.SH' 表示中国平安。
        start_date: 开始日期，格式为 'YYYYMMDD'，例如 '20240101'。
        end_date: 结束日期，格式为 'YYYYMMDD'，例如 '20241128'。

    Returns:
        pd.DataFrame: 股票日线行情数据，包含以下列：

            - ts_code: 股票代码
            - trade_date: 交易日期 (YYYYMMDD)
            - open: 开盘价 (单位: 元)
            - high: 最高价 (单位: 元)
            - low: 最低价 (单位: 元)
            - close: 收盘价 (单位: 元)
            - pre_close: 前收盘价 (单位: 元)
            - change: 涨跌额 (单位: 元)
            - pct_chg: 涨跌幅 (%)
            - vol: 成交量 (单位: 手)
            - amount: 成交额 (单位: 元)
            - code: 股票代码


    Example:
        >>> df = get_stock_daily('601138.SH', '20240101', '20241128')
        >>> print(df)
              ts_code  trade_date   open   high    low  close  pre_close  change  pct_chg         vol        amount     code
        0  601138.SH     20241128  22.32  22.32  21.78  21.85      22.32   -0.47  -2.1057   908157.54  1995650.738  601138
        1  601138.SH     20241127  21.92  22.42  21.72  22.32      22.09    0.23   1.0412   868574.73  1922345.866  601138
        2  601138.SH     20241126  21.89  22.42  21.71  22.09      22.11   -0.02  -0.0905   896672.97  1977884.775  601138
        ...        ...         ...    ...    ...    ...    ...        ...     ...      ...         ...          ...      ...

    """
    return tp.fetch_stock_daily_data(symbol, None, start_date, end_date)

def get_technical_factor(
    data: Union[str, pd.DataFrame], 
    windows: Dict[str, list] = None,
    start_date: str = None, 
    end_date: str = None,
    include_cdl: bool = False,
    days: Optional[int] = None,
    use_tushare: bool = False
) -> pd.DataFrame:
    """Calculate various technical indicators for stock data.
    
    Args:
        data: Either a DataFrame with OHLCV data or a stock symbol (e.g., '601138.SH')
        windows: Dictionary of lookback periods for different indicators
        start_date: Start date for data retrieval if symbol is provided (format: YYYYMMDD)
        end_date: End date for data retrieval if symbol is provided (format: YYYYMMDD)
        include_cdl: Whether to include candlestick pattern indicators
        days: Number of days to look back (alternative to start_date/end_date)
        use_tushare: Whether to use Tushare data source (True) or East Money data source (False)
    
    Returns:
        DataFrame containing all calculated technical indicators and pattern signals with NA values removed,
        including a 'date' column (renamed from 'trade_date')
    """
    # Handle input data
    if isinstance(data, str):
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if days:
            # 获取交易日历以计算start_date
            start_date, end_date = get_last_trading_dates(days)
        elif not start_date:
            # 如果既没有指定days也没有指定start_date，使用默认的365+120天
            start_date = (datetime.now() - timedelta(days=365+120)).strftime('%Y%m%d')
            
        df = get_daily_bar(data, start_date, end_date, is_tushare=use_tushare)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if days:
            # 如果提供了DataFrame并指定了days，只取最近的days行
            df = df.head(days)
    else:
        raise ValueError("data must be either a stock symbol string or a pandas DataFrame")

    # Validate required columns
    required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by date in ascending order
    df = df.sort_values('trade_date')
    df = df.reset_index(drop=True)
    
    # Initialize default windows if not provided
    default_windows = {
        'sma': [5, 10, 20, 60],
        'ema': [5, 10, 20, 60],
        'macd': [12, 26, 9],  # fast, slow, signal
        'rsi': [6, 12, 24],
        'stoch': [14, 3, 3],  # k, d, smooth
        'bbands': [20, 2],    # period, std
        'atr': [14],
        'adx': [14],
        'mfi': [14],
        'volatility': [10, 20, 30],
        'tsi': [25, 13],      # long, short
        'ultimate': [7, 14, 28]  # for Ultimate Oscillator
    }
    
    windows = windows or default_windows
    
    # Convert price and volume data to pandas Series for easier calculation
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']  # 注意这里不再使用vol
    
    # Initialize DataFrame with original OHLCV data and date
    result_df = df[['trade_date', 'open', 'high', 'low', 'close', 'volume']].copy()
    result_df = result_df.rename(columns={'trade_date': 'date'})
    
    # 在初始化result_df后，添加量比计算
    # 计算5日平均成交量（不包括当日）
    ma5_volume = volume.shift(1).rolling(window=5).mean()
    # 计算量比
    result_df['volume_ratio'] = volume / ma5_volume

    # Calculate Moving Averages
    for period in windows['sma']:
        result_df[f'sma_{period}'] = ta.SMA(close, timeperiod=period)
        
    for period in windows['ema']:
        result_df[f'ema_{period}'] = ta.EMA(close, timeperiod=period)
    
    # MACD
    fast, slow, signal = windows['macd']
    macd, macd_signal, macd_hist = ta.MACD(close, 
                                          fastperiod=fast,
                                          slowperiod=slow, 
                                          signalperiod=signal)
    result_df['macd'] = macd
    result_df['macd_signal'] = macd_signal
    result_df['macd_hist'] = macd_hist
    
    # RSI
    for period in windows['rsi']:
        result_df[f'rsi_{period}'] = ta.RSI(close, timeperiod=period)
    
    # Stochastic
    k_period, d_period, smooth = windows['stoch']
    slowk, slowd = ta.STOCH(high, low, close,
                           fastk_period=k_period,
                           slowk_period=d_period,
                           slowk_matype=0,
                           slowd_period=smooth,
                           slowd_matype=0)
    result_df['stoch_k'] = slowk
    result_df['stoch_d'] = slowd
    
    # Bollinger Bands
    period, std = windows['bbands']
    upperband, middleband, lowerband = ta.BBANDS(close,
                                                timeperiod=period,
                                                nbdevup=std,
                                                nbdevdn=std,
                                                matype=0)
    result_df['bb_upper'] = upperband
    result_df['bb_middle'] = middleband
    result_df['bb_lower'] = lowerband
    result_df['bb_width'] = (upperband - lowerband) / middleband
    
    # Average True Range
    for period in windows['atr']:
        result_df[f'atr_{period}'] = ta.ATR(high, low, close, timeperiod=period)
    
    # ADX - Average Directional Index
    for period in windows['adx']:
        result_df[f'adx_{period}'] = ta.ADX(high, low, close, timeperiod=period)
        result_df[f'plus_di_{period}'] = ta.PLUS_DI(high, low, close, timeperiod=period)
        result_df[f'minus_di_{period}'] = ta.MINUS_DI(high, low, close, timeperiod=period)
    
    # Money Flow Index
    for period in windows['mfi']:
        result_df[f'mfi_{period}'] = ta.MFI(high, low, close, volume, timeperiod=period)
    
    # On Balance Volume
    result_df['obv'] = ta.OBV(close, volume)
    
    # Force Index
    result_df['force_index'] = (close - close.shift(1)) * volume
    
    # Volatility Indicators
    for period in windows['volatility']:
        result_df[f'volatility_{period}'] = (close.rolling(window=period).std() / 
                                           close.rolling(window=period).mean() * 100)
    
    # True Strength Index
    long_period, short_period = windows['tsi']
    momentum = close - close.shift(1)
    abs_momentum = momentum.abs()
    
    # Double smoothed momentum
    smooth1 = pd.Series(ta.EMA(momentum, timeperiod=long_period))
    smooth2 = pd.Series(ta.EMA(smooth1, timeperiod=short_period))
    
    # Double smoothed absolute momentum
    abs_smooth1 = pd.Series(ta.EMA(abs_momentum, timeperiod=long_period))
    abs_smooth2 = pd.Series(ta.EMA(abs_smooth1, timeperiod=short_period))
    
    result_df['tsi'] = 100 * smooth2 / abs_smooth2
    
    # Ultimate Oscillator
    bp = close - pd.Series(np.minimum(low, close.shift(1)))
    tr = pd.Series(np.maximum(high, close.shift(1))) - pd.Series(np.minimum(low, close.shift(1)))
    
    period1, period2, period3 = windows['ultimate']
    avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
    avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
    avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
    
    result_df['ultimate_oscillator'] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    
    # Additional indicators
    result_df['roc'] = ta.ROC(close, timeperiod=10)
    result_df['cci'] = ta.CCI(high, low, close, timeperiod=14)
    result_df['willr'] = ta.WILLR(high, low, close, timeperiod=14)
    result_df['chaikin_ad'] = ta.AD(high, low, close, volume)
    result_df['cmf'] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    result_df['avg_price'] = ta.AVGPRICE(open_price, high, low, close)
    result_df['typical_price'] = (high + low + close) / 3
    result_df['weighted_close'] = ta.WCLPRICE(high, low, close)
    result_df['pvt'] = (close.pct_change() * volume).cumsum()
    
    # Pattern Recognition (only if include_cdl is True)
    if include_cdl:
        pattern_functions = {
            'engulfing': ta.CDLENGULFING,
            'morning_star': lambda o, h, l, c: ta.CDLMORNINGSTAR(o, h, l, c, penetration=0),
            'evening_star': lambda o, h, l, c: ta.CDLEVENINGSTAR(o, h, l, c, penetration=0),
            'hammer': ta.CDLHAMMER,
            'hanging_man': ta.CDLHANGINGMAN,
            'three_white_soldiers': ta.CDL3WHITESOLDIERS,
            'three_black_crows': ta.CDL3BLACKCROWS,
            'three_methods': ta.CDLRISEFALL3METHODS,
            'marubozu': ta.CDLMARUBOZU,
            'doji': ta.CDLDOJI,
            'harami': ta.CDLHARAMI
        }
        
        for pattern_name, pattern_func in pattern_functions.items():
            result_df[f'pattern_{pattern_name}'] = pattern_func(open_price, high, low, close)
    
    # Drop rows containing NA values and sort back to descending order
    result_df = result_df.dropna()
    result_df = result_df.sort_values('date', ascending=False).reset_index(drop=True)
    
    return result_df

def interpret_patterns(signals: Dict[str, np.ndarray], index: int = -1) -> Dict[str, str]:
    """
    Interpret the pattern signals for a given index (default is latest)
    
    Args:
        signals: Dictionary containing pattern signals
        index: Index to interpret (-1 for most recent)
        
    Returns:
        Dictionary with pattern interpretations
    """
    interpretations = {}
    pattern_keys = [k for k in signals.keys() if k.startswith('pattern_')]
    
    for key in pattern_keys:
        value = signals[key][index]
        pattern_name = key.replace('pattern_', '')
        
        if value != 0:
            direction = "Bullish" if value > 0 else "Bearish"
            interpretations[pattern_name] = f"{direction} signal"
            
    return interpretations

def get_index_daily_basic(symbol: str, start_date, end_date) -> pd.DataFrame:
    """获取指数日线行情数据。

    Args:
        symbol (str): 指数代码，例如 '000001' 表示上证指数。
        start_date: 开始日期，格式为 'YYYYMMDD'，例如 '20240101'。
        end_date: 结束日期，格式为 'YYYYMMDD'，例如 '20241128'。

    Returns:
        pd.DataFrame: 指数日线行情数据，包含以下列：

            - trade_date: 交易日期 (YYYYMMDD)
            - total_mv: 总市值 (单位: 元)
            - float_mv: 流通市值 (单位: 元)
            - total_share: 总股本 (单位: 股)
            - float_share: 流通股本 (单位: 股)
            - free_share: 自由流通股本 (单位: 股)
            - turnover_rate: 换手率 (%)
            - turnover_rate_f: 自由流通股换手率 (%)
            - pe: 市盈率 (静态)
            - pe_ttm: 市盈率 (TTM)
            - pb: 市净率
            - code: 指数代码


    Example:
        >>> df = get_index_daily_basic('000001', '20240101', '20241128')
        >>> print(df)
                trade_date     total_mv     float_mv  total_share  float_share  free_share  turnover_rate  turnover_rate_f    pe  pe_ttm   pb     code
        0      20241128  6.229460e+13  4.797827e+13  5.753688e+12  4.603319e+12  1.791159e+12            1.10              2.82  14.44  14.16  1.32  000001
        1      20241127  6.256993e+13  4.817586e+13  5.753716e+12  4.602865e+12  1.791159e+12            1.09              2.81  14.50  14.22  1.32  000001
        2      20241126  6.165579e+13  4.746168e+13  5.753836e+12  4.601223e+12  1.791220e+12            1.01              2.60  14.29  14.02  1.30  000001
        ...           ...           ...           ...           ...           ...           ...             ...              ...   ...    ...   ...      ...

    """
    return tp.fetch_index_daily_basic(None, symbol, start_date, end_date)

def get_ths_index_daily(symbol: str,start_date,end_date) ->  pd.DataFrame:
    return tp.get_ths_index_daily(symbol,start_date,end_date)

def get_index_daily(symbol: str, start_date, end_date) -> pd.DataFrame:
    """获取指数日线行情数据。

    Args:
        symbol (str): 指数代码，例如 '000001' 表示上证指数。
        start_date: 开始日期，格式为 'YYYYMMDD'，例如 '20240101'。
        end_date: 结束日期，格式为 'YYYYMMDD'，例如 '20241128'。

    Returns:
        pd.DataFrame: 指数日线行情数据，包含以下列：

            - trade_date: 交易日期 (YYYYMMDD)
            - close: 收盘价 (单位: 点)
            - open: 开盘价 (单位: 点)
            - high: 最高价 (单位: 点)
            - low: 最低价 (单位: 点)
            - pre_close: 前收盘价 (单位: 点)
            - change: 涨跌额 (单位: 点)
            - pct_chg: 涨跌幅 (%)
            - vol: 成交量 (单位: 手)
            - amount: 成交额 (单位: 元)
            - code: 指数代码


    Example:
        >>> df = get_index_daily('000001', '20240101', '20241128')
        >>> print(df)
               trade_date      close       open       high        low  pre_close    change  pct_chg          vol         amount     code
        0      20241128  3295.6983  3308.2797  3319.2559  3290.3641  3309.7789  -14.0806  -0.4254  511953047.0  558164131.7  000001
        1      20241127  3309.7789  3250.5910  3309.8805  3227.3553  3259.7572   50.0217   1.5345  511202303.0  571617596.0  000001
        2      20241126  3259.7572  3256.8576  3285.3281  3252.8661  3263.7597   -4.0025  -0.1226  472219304.0  509490596.3  000001
        ...           ...        ...        ...        ...        ...        ...       ...      ...          ...           ...      ...

    """
    return tp.get_index_daily(None, symbol, start_date, end_date)

@analyze_cache.cache(ttl=60*60*24*2)
def get_financial_analysis_summary(symbol: str) -> dict:
    return tp.get_financial_indicators(symbol)

@analyze_cache.cache(ttl=60*60*24*2)
def get_stock_profit_forecast(symbol: str) -> dict:
    return tp.fetch_stock_forecast_data(symbol)

def remove_prefix(code: str) -> str:
    """移除股票代码的前缀"""
    return code.lstrip('SH').lstrip('SZ').lstrip('BJ')

def get_kpl_rank(num: int = 100) -> list:
    """
    获取开盘啦涨停、跌停、炸板等榜单数据
    
    参数:
    num (int): 获取的数据条数，默认100条
    
    返回:
    list: 股票代码列表
    """
    # 获取榜单数据
    df = tp.fetch_kpl_list(limit=num)
    
    # 如果DataFrame为空，返回空列表
    if df.empty:
        return []
    
    # 如果code列存在，提取并转换为列表
    if 'code' in df.columns:
        # 去除重复值并转换为列表
        return df['code'].drop_duplicates().tolist()
    
    # 如果没有code列但有ts_code列，处理ts_code列
    elif 'ts_code' in df.columns:
        # 提取ts_code的第一部分（股票代码）
        codes = df['ts_code'].str.split('.').str[0].drop_duplicates().tolist()
        return codes
    
    # 如果既没有code列也没有ts_code列，返回空列表
    return []

def get_xueqiu_hot_follow(num: int = 100) -> dict:
    """获取雪球关注排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
    df = ak.stock_hot_follow_xq(symbol="最热门")
    result = {}
    for _, row in df.head(num).iterrows():
        code = remove_prefix(row['股票代码'])
        info = f"股票简称: {row['股票简称']}, 关注: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
        result[code] = info
    return result

def get_xueqiu_hot_tweet(num: int = 100) -> dict:
    """获取雪球讨论排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
    df = ak.stock_hot_tweet_xq(symbol="最热门")
    result = {}
    for _, row in df.head(num).iterrows():
        code = remove_prefix(row['股票代码'])
        info = f"股票简称: {row['股票简称']}, 讨论: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
        result[code] = info
    return result

def get_xueqiu_hot_deal(num: int = 100) -> dict:
    """获取雪球交易排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
    df = ak.stock_hot_deal_xq(symbol="最热门")
    result = {}
    for _, row in df.head(num).iterrows():
        code = remove_prefix(row['股票代码'])
        info = f"股票简称: {row['股票简称']}, 交易: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
        result[code] = info
    return result

def get_wencai_hot_rank(num: int = 100) -> dict:
    """获取问财热门股票排名,参数num: int = 100，返回值Dict[symbol,str]"""
    date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")  # 获取昨天的日期
    df = ak.stock_hot_rank_wc(date=date)
    result = {}
    for _, row in df.head(num).iterrows():
        code = row['股票代码']
        info = f"股票简称: {row['股票简称']}, 现价: {row['现价']:.2f}, 涨跌幅: {row['涨跌幅']:.2f}%, 热度: {row['个股热度']:.0f}, 排名: {row['个股热度排名']}"
        result[code] = info
    return result

def get_eastmoney_hot_rank(num: int = 100) -> dict:
    """获取东方财富人气榜-A股,参数num: int = 100，返回值Dict[symbol,str]"""
    df = ak.stock_hot_rank_em()
    result = {}
    for _, row in df.head(num).iterrows():
        code = remove_prefix(row['代码'])
        info = f"股票名称: {row['股票名称']}, 最新价: {row['最新价']:.2f}, 涨跌额: {row['涨跌额']:.2f}, 涨跌幅: {row['涨跌幅']:.2f}%"
        result[code] = info
    return result

def get_baidu_hotrank( hour=12, num=20) -> dict:
    """
    获取百度热门股票排行榜。参数  hour=7, num=20 返回 Dict[symbol,str]

    参数:
    hour (int): 获取最近多少小时的数据，默认为7小时。
    num (int): 需要获取的热门股票数量，默认为20。

    返回:
    dict: 键为股票代码，值为格式化的字符串，包含热门股票的详细信息，包括排名。
    """
    # 获取当前日期
    date = datetime.now().strftime("%Y%m%d")
    
    # 获取热门股票列表
    hotlist = baidu_news_api.fetch_hotrank(day=date, hour=hour, rn=num)
    
    # 格式化结果
    result = {}
    for rank, item in enumerate(hotlist, 1):  # enumerate从1开始，给每个项目一个排名
        stock_info = (
            f"排名: {rank}\n"  # 添加排名信息
            f"股票名称: {item['name']}\n"
            f"当前价格: {item['price']}\n"
            f"涨跌幅: {item['change']}\n"
            f"所属板块: {item['sector']}\n"
            f"排名变化: {item['rank_change']}\n"
            f"地区: {item['region']}\n"
            f"热度: {item['heat']}\n"
            f"----------------------"
        )
        result[item['code']] = stock_info
    
    return result

def get_ths_rank(num: int = 100) -> list:
    """
    获取同花顺App热榜股票代码列表
    
    参数:
    num (int): 获取的股票数量，默认100条
    
    返回:
    list: 股票代码列表
    """
    # 获取同花顺热榜数据，只获取A股热股
    df = tp.fetch_ths_hot_data(
        market='热股',
        limit=num,
        is_new='Y'  # 获取最新数据
    )
    
    # 如果DataFrame为空，返回空列表
    if df.empty:
        return []
    
    # 如果存在code列，直接提取
    if 'code' in df.columns:
        # 去除重复值并转换为列表
        return df['code'].drop_duplicates().tolist()
    
    # 如果没有code列但有ts_code列，处理ts_code列
    elif 'ts_code' in df.columns:
        # 提取ts_code的第一部分（股票代码）
        codes = df['ts_code'].str.split('.').str[0].drop_duplicates().tolist()
        return codes
    
    # 如果既没有code列也没有ts_code列，返回空列表
    return []

@function_cache.cache(ttl=60*60*2)
def get_combined_hot_stocks(num: int = 100) -> List[str]:
    """
    获取综合的热门股票列表，包括雪球讨论、雪球交易、问财热门、东方财富人气榜、百度热榜、开盘啦榜单和同花顺热榜。

    参数:
    num (int): 从每个来源获取的股票数量，默认为100。

    返回:
    List[str]: 交叉合并后的去重股票代码列表。
    """
    symbol_dict = matcher

    wencai = []

    # 获取各个来源的热门股票
    xueqiu_tweet = list(get_xueqiu_hot_tweet(num).keys())
    xueqiu_deal = list(get_xueqiu_hot_deal(num).keys())
    eastmoney = list(get_eastmoney_hot_rank(num).keys())
    baidu = list(get_baidu_hotrank(num=num).keys())
    kpl = get_kpl_rank(num)
    ths = get_ths_rank(num)  # 添加同花顺热榜
    
    try:
        wencai = list(get_wencai_hot_rank(num).keys())
    except Exception as e:
        pass
    
    # 将所有列表合并到一个列表中
    all_lists = [xueqiu_tweet, xueqiu_deal, eastmoney, baidu, kpl, ths]  # 添加ths列表
    if len(wencai) > 0:
        all_lists.append(wencai)

    # 创建一个集合来存储已经添加的股票代码，用于去重
    seen = set()
    result = []

    # 交叉合并列表
    max_length = max(len(lst) for lst in all_lists)
    for i in range(max_length):
        for lst in all_lists:
            if i < len(lst):
                stock = lst[i]
                if isinstance(stock, str) and stock not in seen and stock in symbol_dict:
                    seen.add(stock)
                    result.append(stock)

    return result

def get_institutional_research_records(symbol: str, trade_date: str, start_date: str, end_date: str, limit: int=10) -> list:
    """
    获取机构调研记录并转换为中文列名格式
    
    Args:
        symbol (str): 股票代码
        trade_date (str): 调研日期 YYYYMMDD
        start_date (str): 始日期 YYYYMMDD
        end_date (str): 结束日期 YYYYMMDD
        limit (int): 返回记录数量限制
        
    Returns:
        list: 处理后的调研记录列表
    """
    # 获取原始数据
    df = tp.get_institutional_research_records(
        symbol=symbol,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=0,
        limit=limit
    )
    
    # 定义列名映射
    column_mapping = {
        'name': '股票名称',
        'surv_date': '调研日期',
        'fund_visitors': '调研机构',
        'rece_place': '接待地点',
        'rece_mode': '接待方式',
        'rece_org': '接待人员',
        'org_type': '机构类型',
        'comp_rece': '公司接待人员',
        'content': '调研内容',
        'code': '股票代码'
    }
    
    # 重命名列
    df.rename(columns=column_mapping, inplace=True)
    
    # 确保code列存在，如果不存在则从ts_code转换
    if '股票代码' not in df.columns and 'ts_code' in df.columns:
        df['股票代码'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df['股票代码'].tolist()

def get_broker_monthly_picks(month: str = None, offset: int = 0, limit: int = 1000) -> list:
    return tp.get_broker_monthly_picks(month, offset, limit)

def get_broker_reports(symbol:str) -> list:
    return tp.fetch_broker_reports(symbol)

def get_broker_monthly_picks(month: str = None) -> list:
    return tp.get_broker_monthly_picks(month)

def get_latest_stock_chip_performance(symbol: str = None) -> dict:
    return tp.get_latest_stock_chip_performance(symbol)

def get_latest_stock_chip_distribution(symbol: str = None) -> list:
    return tp.get_latest_stock_chip_distribution(symbol)

@analyze_cache.cache(ttl=60*60*2)
def get_latest_cyq_data(symbol: str = None) -> list:
    return get_latest_stock_chip_performance(symbol)

@analyze_cache.cache(ttl=60*60*24*2)
def get_investment_ratings(symbol: str = None) -> list:
    return tp.fetch_broker_reports(symbol)

@analyze_cache.cache(ttl=60*60*2)
def get_stock_big_deal(symbol: str = None) -> list:
    return tp.fetch_block_trades(symbol)

def get_sector_moneyflow(symbol: str = None) -> list:
    return tp.get_sector_moneyflow(symbol)

def fetch_ths_industry_moneyflow(symbol: str = None) -> list:
    return tp.fetch_ths_industry_moneyflow(symbol)

@analyze_cache.cache(ttl=60*60*2)
def get_recent_stock_moneyflow(symbol: str = None, limit: int = 3) -> list:
    return tp.get_recent_stock_moneyflow(symbol, limit)

def get_stock_sector(symbol: str = None) -> list:
    return tp.get_sector_moneyflow(symbol)

def get_concept_constituents() ->str:
    df = tp.get_concept_constituents()
    # 获取name列的唯一值并转为列表
    unique_names = df['name'].unique().tolist()
    # 用逗号连接所有名称
    result = ','.join(unique_names)
    return result

@analyze_cache.cache(ttl=60*60*2)
def get_full_realtime_data()->dict:
    df = tp.fetch_realtime_stock_data()
    column_mapping = {
        'NAME':'name',
        'PRICE': '当前价格',
        'PCT_CHANGE': '涨跌幅',
        'CHANGE': '涨跌额',
        'VOLUME': '成交量',
        'AMOUNT': '成交金额',
        'SWING': '振幅',
        'LOW': '今日最低价',
        'HIGH': '今日最高价',
        'OPEN': '今日开盘价',
        'CLOSE': '今日收盘价',
        'VOL_RATIO': '量比',
        'TURNOVER_RATE': '换手率',
        'PE': '市盈率',
        'PB': '市净率',
        'TOTAL_MV': '总市值',
        'FLOAT_MV': '流通市值',
        'RISE': '涨速',
        '5MIN': '5分钟涨幅',
        '60DAY': '60天涨幅',
        '1YEAR': '1年涨幅'
    }

    # 使用 rename() 方法替换列名
    df = df.rename(columns=column_mapping)

    return df.set_index('code').to_dict(orient='index')

@analyze_cache.cache(ttl=60*60*2)
def get_speculative_capital_trading() -> list:
    dates = tp.get_last_trading_dates()
    start_date = dates[-1]
    end_date = dates[0]
    df: pd.DataFrame = tp.fetch_daily_hot_money_details(start_date=start_date, end_date=end_date)
    # 删除 hm_orgs 和 trade_date 列
    df = df.drop(columns=['hm_orgs', 'trade_date','ts_code'])

    # 按照 code 合并重复数据
    df_merged = df.groupby('code').agg({
        'hm_name': lambda x: ','.join(x.unique()),  
        'buy_amount': 'sum',
        'sell_amount': 'sum',
        'net_amount': 'sum', 
        'ts_name': 'first'   
    }).reset_index()
    # 重命名列
    df_merged.rename(columns={
        'ts_name': 'name'
    }, inplace=True)

    return df_merged.set_index('code').to_dict(orient='index')

def get_technical_analysis_factor(symbol,days:int =1):
    df = tp.fetch_stock_technical_factors(symbol,limit=days)   
    df.drop(columns=['ts_code'],inplace=True)
    return df.to_dict(orient='records')

def get_technical_factor_for_minutes_data(
    data: Union[str, pd.DataFrame],
    period: str = "1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    windows: Dict[str, list] = None
) -> pd.DataFrame:
    """Calculate technical indicators for minute-level stock data.
    
    Args:
        data: Either a DataFrame with OHLCV data or a stock symbol (e.g., '000001')
        period: Time period, options: {"1", "5", "15", "30", "60"}
                Note: 1-min data only available for last 5 trading days
        start_date: Start datetime in format "YYYY-MM-DD HH:MM:SS" (ignored if period="1")
        end_date: End datetime in format "YYYY-MM-DD HH:MM:SS" (ignored if period="1")
        windows: Dictionary of lookback periods for different indicators
    
    Returns:
        DataFrame containing calculated technical indicators with NA values removed
    """
    # Handle input data
    if isinstance(data, str):
        df = get_minute_data(
            symbol=data,
            start_date=None if period == "1" else start_date,
            end_date=None if period == "1" else end_date,
            period=period
        )
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be either a stock symbol string or a pandas DataFrame")

    # Validate required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by timestamp in ascending order
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    
    # Initialize default windows if not provided
    default_windows = {
        'sma': [5, 10, 20, 60],
        'ema': [5, 10, 20, 60],
        'macd': [12, 26, 9],  # fast, slow, signal
        'rsi': [6, 12, 24],
        'stoch': [14, 3, 3],  # k, d, smooth
        'bbands': [20, 2],    # period, std
        'atr': [14],
        'adx': [14],
        'mfi': [14],
        'volatility': [10, 20, 30],
        'tsi': [25, 13],      # long, short
        'ultimate': [7, 14, 28]  # for Ultimate Oscillator
    }
    
    windows = windows or default_windows
    
    # Convert price and volume data to pandas Series
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Initialize result DataFrame
    result_df = pd.DataFrame()
    result_df['timestamp'] = df['timestamp']
    result_df['open'] = open_price
    result_df['high'] = high
    result_df['low'] = low
    result_df['close'] = close
    result_df['volume'] = volume
    
    # Calculate volume ratio (only for period="1")
    if period == "1" and 'vwap' in df.columns:
        result_df['vwap'] = df['vwap']
    elif period != "1" and 'change_pct' in df.columns:
        result_df['change_pct'] = df['change_pct']
        result_df['change'] = df['change']
        result_df['amplitude'] = df['amplitude']
        result_df['turnover'] = df['turnover']
    
    # Calculate Moving Averages
    for period in windows['sma']:
        result_df[f'sma_{period}'] = ta.SMA(close, timeperiod=period)
        
    for period in windows['ema']:
        result_df[f'ema_{period}'] = ta.EMA(close, timeperiod=period)
    
    # MACD
    fast, slow, signal = windows['macd']
    macd, macd_signal, macd_hist = ta.MACD(close, 
                                          fastperiod=fast,
                                          slowperiod=slow, 
                                          signalperiod=signal)
    result_df['macd'] = macd
    result_df['macd_signal'] = macd_signal
    result_df['macd_hist'] = macd_hist
    
    # RSI
    for period in windows['rsi']:
        result_df[f'rsi_{period}'] = ta.RSI(close, timeperiod=period)
    
    # Stochastic
    k_period, d_period, smooth = windows['stoch']
    slowk, slowd = ta.STOCH(high, low, close,
                           fastk_period=k_period,
                           slowk_period=d_period,
                           slowk_matype=0,
                           slowd_period=smooth,
                           slowd_matype=0)
    result_df['stoch_k'] = slowk
    result_df['stoch_d'] = slowd
    
    # Bollinger Bands
    period, std = windows['bbands']
    upperband, middleband, lowerband = ta.BBANDS(close,
                                                timeperiod=period,
                                                nbdevup=std,
                                                nbdevdn=std,
                                                matype=0)
    result_df['bb_upper'] = upperband
    result_df['bb_middle'] = middleband
    result_df['bb_lower'] = lowerband
    result_df['bb_width'] = (upperband - lowerband) / middleband
    
    # Additional indicators
    result_df['roc'] = ta.ROC(close, timeperiod=10)
    result_df['cci'] = ta.CCI(high, low, close, timeperiod=14)
    result_df['willr'] = ta.WILLR(high, low, close, timeperiod=14)
    result_df['obv'] = ta.OBV(close, volume)
    
    # Drop rows containing NA values and sort by timestamp in descending order
    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    return result_df

def get_cdl_factor(
    data: Union[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None
) -> pd.DataFrame:
    """Calculate candlestick pattern indicators for daily stock data.
    
    Args:
        data: Either a DataFrame with OHLCV data or a stock symbol (e.g., '000001')
        start_date: Start date in format "YYYYMMDD" (default: 365 days ago)
        end_date: End date in format "YYYYMMDD" (default: today)
        days: Number of days to look back (alternative to start_date/end_date)
    
    Returns:
        DataFrame containing candlestick pattern signals with NA values removed.
        Pattern values: 
        - 100: Strong bullish signal
        - 50: Weak bullish signal
        - 0: No pattern
        - -50: Weak bearish signal
        - -100: Strong bearish signal
    """
    # Handle input data
    if isinstance(data, str):
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if days:
            # 获取交易日历以计算start_date
            start_date, end_date = get_last_trading_dates(days)
        elif not start_date:
            # 如果既没有指定days也没有指定start_date，使用默认的365天
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
        df = get_stock_daily(data, start_date, end_date)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if days:
            # 如果提供了DataFrame并指定了days，只取最近的days行
            df = df.head(days)
    else:
        raise ValueError("data must be either a stock symbol string or a pandas DataFrame")

    # Validate required columns
    required_columns = ['trade_date', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by date in ascending order for calculation
    df = df.sort_values('trade_date')
    df = df.reset_index(drop=True)
    
    # Extract OHLC data
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Initialize result DataFrame with timestamp
    result_df = pd.DataFrame()
    result_df['timestamp'] = pd.to_datetime(df['trade_date'])  # Convert trade_date to timestamp
    
    # Dictionary of all available candlestick pattern functions in talib
    pattern_functions = {
        'CDL2CROWS': ta.CDL2CROWS,
        'CDL3BLACKCROWS': ta.CDL3BLACKCROWS,
        'CDL3INSIDE': ta.CDL3INSIDE,
        'CDL3LINESTRIKE': ta.CDL3LINESTRIKE,
        'CDL3OUTSIDE': ta.CDL3OUTSIDE,
        'CDL3STARSINSOUTH': ta.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': ta.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': lambda o, h, l, c: ta.CDLABANDONEDBABY(o, h, l, c, penetration=0),
        'CDLADVANCEBLOCK': ta.CDLADVANCEBLOCK,
        'CDLBELTHOLD': ta.CDLBELTHOLD,
        'CDLBREAKAWAY': ta.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': ta.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': ta.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': ta.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': lambda o, h, l, c: ta.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0),
        'CDLDOJI': ta.CDLDOJI,
        'CDLDOJISTAR': ta.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': ta.CDLDRAGONFLYDOJI,
        'CDLENGULFING': ta.CDLENGULFING,
        'CDLEVENINGDOJISTAR': lambda o, h, l, c: ta.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0),
        'CDLEVENINGSTAR': lambda o, h, l, c: ta.CDLEVENINGSTAR(o, h, l, c, penetration=0),
        'CDLGAPSIDESIDEWHITE': ta.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': ta.CDLGRAVESTONEDOJI,
        'CDLHAMMER': ta.CDLHAMMER,
        'CDLHANGINGMAN': ta.CDLHANGINGMAN,
        'CDLHARAMI': ta.CDLHARAMI,
        'CDLHARAMICROSS': ta.CDLHARAMICROSS,
        'CDLHIGHWAVE': ta.CDLHIGHWAVE,
        'CDLHIKKAKE': ta.CDLHIKKAKE,
        'CDLHIKKAKEMOD': ta.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': ta.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': ta.CDLIDENTICAL3CROWS,
        'CDLINNECK': ta.CDLINNECK,
        'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER,
        'CDLKICKING': ta.CDLKICKING,
        'CDLKICKINGBYLENGTH': ta.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': ta.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': ta.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': ta.CDLLONGLINE,
        'CDLMARUBOZU': ta.CDLMARUBOZU,
        'CDLMATCHINGLOW': ta.CDLMATCHINGLOW,
        'CDLMATHOLD': lambda o, h, l, c: ta.CDLMATHOLD(o, h, l, c, penetration=0),
        'CDLMORNINGDOJISTAR': lambda o, h, l, c: ta.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0),
        'CDLMORNINGSTAR': lambda o, h, l, c: ta.CDLMORNINGSTAR(o, h, l, c, penetration=0),
        'CDLONNECK': ta.CDLONNECK,
        'CDLPIERCING': ta.CDLPIERCING,
        'CDLRICKSHAWMAN': ta.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': ta.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': ta.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR,
        'CDLSHORTLINE': ta.CDLSHORTLINE,
        'CDLSPINNINGTOP': ta.CDLSPINNINGTOP,
        'CDLSTALLEDPATTERN': ta.CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': ta.CDLSTICKSANDWICH,
        'CDLTAKURI': ta.CDLTAKURI,
        'CDLTASUKIGAP': ta.CDLTASUKIGAP,
        'CDLTHRUSTING': ta.CDLTHRUSTING,
        'CDLTRISTAR': ta.CDLTRISTAR,
        'CDLUNIQUE3RIVER': ta.CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': ta.CDLUPSIDEGAP2CROWS,
        'CDLXSIDEGAP3METHODS': ta.CDLXSIDEGAP3METHODS
    }
    
    # Calculate all pattern signals
    for pattern_name, pattern_func in pattern_functions.items():
        try:
            result_df[pattern_name] = pattern_func(open_price, high, low, close)
        except Exception as e:
            print(f"Warning: Failed to calculate {pattern_name}: {str(e)}")
            continue
    
    # Add basic price data
    result_df['open'] = open_price
    result_df['high'] = high
    result_df['low'] = low
    result_df['close'] = close
    
    # Add additional columns if available
    if 'vol' in df.columns:
        result_df['volume'] = df['vol']
    if 'amount' in df.columns:
        result_df['amount'] = df['amount']
    if 'pct_chg' in df.columns:
        result_df['change_pct'] = df['pct_chg']
    if 'change' in df.columns:
        result_df['change'] = df['change']
    
    # Drop rows containing NA values and sort by timestamp in descending order
    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    # Add a summary column counting total bullish and bearish signals
    pattern_columns = [col for col in result_df.columns if col.startswith('CDL')]
    if pattern_columns:
        result_df['bullish_patterns'] = (result_df[pattern_columns] > 0).sum(axis=1)
        result_df['bearish_patterns'] = (result_df[pattern_columns] < 0).sum(axis=1)
        result_df['pattern_score'] = result_df[pattern_columns].sum(axis=1)
    
    return result_df

def get_cdl_factor_for_minute_data(
    data: Union[str, pd.DataFrame],
    period: str = "1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Calculate candlestick pattern indicators for minute-level stock data.
    
    Args:
        data: Either a DataFrame with OHLCV data or a stock symbol (e.g., '000001')
        period: Time period, options: {"1", "5", "15", "30", "60"}
                Note: 1-min data only available for last 5 trading days
        start_date: Start datetime in format "YYYY-MM-DD HH:MM:SS" (ignored if period="1")
        end_date: End datetime in format "YYYY-MM-DD HH:MM:SS" (ignored if period="1")
    
    Returns:
        DataFrame containing candlestick pattern signals with NA values removed.
        Pattern values: 
        - 100: Strong bullish signal
        - 50: Weak bullish signal
        - 0: No pattern
        - -50: Weak bearish signal
        - -100: Strong bearish signal
    """
    # Handle input data
    if isinstance(data, str):
        df = get_minute_data(
            symbol=data,
            start_date=None if period == "1" else start_date,
            end_date=None if period == "1" else end_date,
            period=period
        )
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be either a stock symbol string or a pandas DataFrame")

    # Validate required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by timestamp in ascending order
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    
    # Extract OHLC data
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Initialize result DataFrame with timestamp
    result_df = pd.DataFrame()
    result_df['timestamp'] = df['timestamp']
    
    # Dictionary of all available candlestick pattern functions in talib
    pattern_functions = {
        'CDL2CROWS': ta.CDL2CROWS,
        'CDL3BLACKCROWS': ta.CDL3BLACKCROWS,
        'CDL3INSIDE': ta.CDL3INSIDE,
        'CDL3LINESTRIKE': ta.CDL3LINESTRIKE,
        'CDL3OUTSIDE': ta.CDL3OUTSIDE,
        'CDL3STARSINSOUTH': ta.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': ta.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': lambda o, h, l, c: ta.CDLABANDONEDBABY(o, h, l, c, penetration=0),
        'CDLADVANCEBLOCK': ta.CDLADVANCEBLOCK,
        'CDLBELTHOLD': ta.CDLBELTHOLD,
        'CDLBREAKAWAY': ta.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': ta.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': ta.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': ta.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': lambda o, h, l, c: ta.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0),
        'CDLDOJI': ta.CDLDOJI,
        'CDLDOJISTAR': ta.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': ta.CDLDRAGONFLYDOJI,
        'CDLENGULFING': ta.CDLENGULFING,
        'CDLEVENINGDOJISTAR': lambda o, h, l, c: ta.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0),
        'CDLEVENINGSTAR': lambda o, h, l, c: ta.CDLEVENINGSTAR(o, h, l, c, penetration=0),
        'CDLGAPSIDESIDEWHITE': ta.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': ta.CDLGRAVESTONEDOJI,
        'CDLHAMMER': ta.CDLHAMMER,
        'CDLHANGINGMAN': ta.CDLHANGINGMAN,
        'CDLHARAMI': ta.CDLHARAMI,
        'CDLHARAMICROSS': ta.CDLHARAMICROSS,
        'CDLHIGHWAVE': ta.CDLHIGHWAVE,
        'CDLHIKKAKE': ta.CDLHIKKAKE,
        'CDLHIKKAKEMOD': ta.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': ta.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': ta.CDLIDENTICAL3CROWS,
        'CDLINNECK': ta.CDLINNECK,
        'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER,
        'CDLKICKING': ta.CDLKICKING,
        'CDLKICKINGBYLENGTH': ta.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': ta.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': ta.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': ta.CDLLONGLINE,
        'CDLMARUBOZU': ta.CDLMARUBOZU,
        'CDLMATCHINGLOW': ta.CDLMATCHINGLOW,
        'CDLMATHOLD': lambda o, h, l, c: ta.CDLMATHOLD(o, h, l, c, penetration=0),
        'CDLMORNINGDOJISTAR': lambda o, h, l, c: ta.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0),
        'CDLMORNINGSTAR': lambda o, h, l, c: ta.CDLMORNINGSTAR(o, h, l, c, penetration=0),
        'CDLONNECK': ta.CDLONNECK,
        'CDLPIERCING': ta.CDLPIERCING,
        'CDLRICKSHAWMAN': ta.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': ta.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': ta.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR,
        'CDLSHORTLINE': ta.CDLSHORTLINE,
        'CDLSPINNINGTOP': ta.CDLSPINNINGTOP,
        'CDLSTALLEDPATTERN': ta.CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': ta.CDLSTICKSANDWICH,
        'CDLTAKURI': ta.CDLTAKURI,
        'CDLTASUKIGAP': ta.CDLTASUKIGAP,
        'CDLTHRUSTING': ta.CDLTHRUSTING,
        'CDLTRISTAR': ta.CDLTRISTAR,
        'CDLUNIQUE3RIVER': ta.CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': ta.CDLUPSIDEGAP2CROWS,
        'CDLXSIDEGAP3METHODS': ta.CDLXSIDEGAP3METHODS
    }
    
    # Calculate all pattern signals
    for pattern_name, pattern_func in pattern_functions.items():
        try:
            result_df[pattern_name] = pattern_func(open_price, high, low, close)
        except Exception as e:
            print(f"Warning: Failed to calculate {pattern_name}: {str(e)}")
            continue
    
    # Add basic price data
    result_df['open'] = open_price
    result_df['high'] = high
    result_df['low'] = low
    result_df['close'] = close
    
    # Add period-specific columns
    if period == "1" and 'vwap' in df.columns:
        result_df['vwap'] = df['vwap']
    elif period != "1" and 'change_pct' in df.columns:
        result_df['change_pct'] = df['change_pct']
        result_df['change'] = df['change']
        result_df['amplitude'] = df['amplitude']
        result_df['turnover'] = df['turnover']
    
    # Drop rows containing NA values and sort by timestamp in descending order
    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    # Add a summary column counting total bullish and bearish signals
    pattern_columns = [col for col in result_df.columns if col.startswith('CDL')]
    if pattern_columns:
        result_df['bullish_patterns'] = (result_df[pattern_columns] > 0).sum(axis=1)
        result_df['bearish_patterns'] = (result_df[pattern_columns] < 0).sum(axis=1)
        result_df['pattern_score'] = result_df[pattern_columns].sum(axis=1)
    
    return result_df

def translate_cdl_columns_to_chinese(df: pd.DataFrame) -> pd.DataFrame:
    """将K线形态分析结果的列名转换为中文。
    
    Args:
        df: get_cdl_factor_for_minute_data 函数返回的DataFrame
        
    Returns:
        转换后的DataFrame，包含中文列名
   Engulfing（吞没形态）


    由两根蜡烛组成，第二根蜡烛完全"吞没"第一根蜡烛的实体
    看涨吞没：在下跌趋势中，第二根为阳线且完全吞没前一根阴线
    看跌吞没：在上涨趋势中，第二根为阴线且完全吞没前一根阳线
    信号强度：较强，特别是在趋势反转点


    Morning Star（晨星形态）


    由三根蜡烛组成的底部反转形态
    第一根是大阴线，第二根是小实体（阴阳皆可），第三根是大阳线
    第二根与第一、三根之间通常有价格跳空
    预示着下跌趋势可能结束，适合寻找做多机会
    信号强度：很强


    Evening Star（夜星形态）


    与晨星相反的顶部反转形态
    第一根是大阳线，第二根是小实体，第三根是大阴线
    预示着上涨趋势可能结束，适合寻找做空机会
    信号强度：很强


    Hammer（锤子线）


    单根蜡烛形态，有长下影线，小实体位于上部
    在下跌趋势末端出现，预示可能反转向上
    下影线长度通常是实体的2-3倍
    信号强度：中等，需要后续确认


    Hanging Man（上吊线）


    形状与锤子线相同，但出现在上涨趋势顶部
    预示可能反转向下
    同样需要长下影线和位于上部的小实体
    信号强度：中等，需要后续确认


    Three White Soldiers（三白兵）


    连续三根上涨的阳线，每根都收在前一根的高点之上
    是强势上涨的信号
    实体最好大致相等，开盘价在前一天实体中部
    信号强度：强


    Three Black Crows（三乌鸦）


    与三白兵相反，是连续三根下跌的阴线
    预示强势下跌
    每根都开在前一根中部，收在新低点
    信号强度：强


    Three Methods（五法：上升三法或下降三法）


    五根蜡烛组成的持续形态
    上升三法：一根大阳线后是三根小阴线（不破第一根低点），最后一根大阳线创新高
    下降三法：一根大阴线后是三根小阳线（不破第一根高点），最后一根大阴线创新低
    信号强度：中等


    Marubozu（光头光脚阳/阴线）


    没有上下影线的实体
    阳线代表强势，阴线代表弱势
    开盘价等于最低价（阳线）或最高价（阴线）
    收盘价等于最高价（阳线）或最低价（阴线）
    信号强度：单独出现时中等


    Doji（十字线）


    开盘价和收盘价相同或极其接近
    表示市场犹豫不决，可能预示趋势反转
    在上升趋势顶部更有意义
    信号强度：弱到中等，需要配合其他指标


    Harami（孕线形态）


    两根蜡烛组成，第二根完全包含在第一根实体之内
    看涨孕线：大阴线后跟小阳线
    看跌孕线：大阳线后跟小阴线
    预示当前趋势可能减弱
    信号强度：中等
    """
    # 基础列名映射
    basic_columns = {
        'timestamp': '时间戳',
        'open': '开盘价',
        'high': '最高价',
        'low': '最低价',
        'close': '收盘价',
        'vwap': '成交量加权平均价',
        'change_pct': '涨跌幅',
        'change': '涨跌额',
        'amplitude': '振幅',
        'turnover': '换手率',
        'bullish_patterns': '看涨形态数量',
        'bearish_patterns': '看跌形态数量',
        'pattern_score': '形态得分'
    }
    
    # K线形态中文名称映射
    pattern_columns = {
        'CDL2CROWS': '两只乌鸦',
        'CDL3BLACKCROWS': '三只乌鸦',
        'CDL3INSIDE': '三内部上涨和下跌',
        'CDL3LINESTRIKE': '三线打击',
        'CDL3OUTSIDE': '三外部上涨和下跌',
        'CDL3STARSINSOUTH': '南方三星',
        'CDL3WHITESOLDIERS': '三白兵',
        'CDLABANDONEDBABY': '弃婴',
        'CDLADVANCEBLOCK': '大敌当前',
        'CDLBELTHOLD': '捉腰带线',
        'CDLBREAKAWAY': '脱离',
        'CDLCLOSINGMARUBOZU': '收盘缺影线',
        'CDLCONCEALBABYSWALL': '藏婴吞没',
        'CDLCOUNTERATTACK': '反击线',
        'CDLDARKCLOUDCOVER': '乌云压顶',
        'CDLDOJI': '十字',
        'CDLDOJISTAR': '十字星',
        'CDLDRAGONFLYDOJI': '蜻蜓十字',
        'CDLENGULFING': '吞没模式',
        'CDLEVENINGDOJISTAR': '十字暮星',
        'CDLEVENINGSTAR': '暮星',
        'CDLGAPSIDESIDEWHITE': '向上/下跳空并列阳线',
        'CDLGRAVESTONEDOJI': '墓碑十字',
        'CDLHAMMER': '锤头',
        'CDLHANGINGMAN': '上吊线',
        'CDLHARAMI': '母子线',
        'CDLHARAMICROSS': '十字孕线',
        'CDLHIGHWAVE': '风高浪大线',
        'CDLHIKKAKE': '陷阱',
        'CDLHIKKAKEMOD': '修正陷阱',
        'CDLHOMINGPIGEON': '家鸽',
        'CDLIDENTICAL3CROWS': '三胞胎乌鸦',
        'CDLINNECK': '颈内线',
        'CDLINVERTEDHAMMER': '倒锤头',
        'CDLKICKING': '反冲形态',
        'CDLKICKINGBYLENGTH': '由较长缺影线决定的反冲形态',
        'CDLLADDERBOTTOM': '梯底',
        'CDLLONGLEGGEDDOJI': '长脚十字',
        'CDLLONGLINE': '长蜡烛',
        'CDLMARUBOZU': '光头光脚/缺影线',
        'CDLMATCHINGLOW': '相同低价',
        'CDLMATHOLD': '铺垫',
        'CDLMORNINGDOJISTAR': '十字晨星',
        'CDLMORNINGSTAR': '晨星',
        'CDLONNECK': '颈上线',
        'CDLPIERCING': '刺透形态',
        'CDLRICKSHAWMAN': '黄包车夫',
        'CDLRISEFALL3METHODS': '三法上涨和下跌',
        'CDLSEPARATINGLINES': '分离线',
        'CDLSHOOTINGSTAR': '射击之星',
        'CDLSHORTLINE': '短蜡烛',
        'CDLSPINNINGTOP': '纺锤',
        'CDLSTALLEDPATTERN': '停顿形态',
        'CDLSTICKSANDWICH': '条形三明治',
        'CDLTAKURI': '探水竿',
        'CDLTASUKIGAP': '跳空并列阴阳线',
        'CDLTHRUSTING': '插入',
        'CDLTRISTAR': '三星',
        'CDLUNIQUE3RIVER': '奇特三河床',
        'CDLUPSIDEGAP2CROWS': '向上跳空的两只乌鸦',
        'CDLXSIDEGAP3METHODS': '上升/下降跳空三法'
    }
    
    # 合并所有列名映射
    all_columns = {**basic_columns, **pattern_columns}
    
    # 创建新的DataFrame，仅包含存在的列
    result_df = pd.DataFrame()
    for eng_col, ch_col in all_columns.items():
        if eng_col in df.columns:
            result_df[ch_col] = df[eng_col]
    
    return result_df

def get_daily_bar(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_tushare: bool = True
) -> pd.DataFrame:
    """获取股票日线数据，支持通过股票代码或名称查询。
    
    Args:
        symbol (str): 股票代码或名称，例如：'000001'、'平安银行'、'000001.SZ'
        start_date (str, optional): 开始日期，格式为'YYYYMMDD'，默认为end_date往前365天
        end_date (str, optional): 结束日期，格式为'YYYYMMDD'，默认为当天
        is_tushare (bool): 是否使用Tushare数据源，默认True
            - True: 使用Tushare数据源(get_stock_daily)
            - False: 使用东方财富数据源(get_daily_data)
    
    Returns:
        pd.DataFrame: 统一格式的日线数据，包含以下列：
            - trade_date: 交易日期 (YYYYMMDD)
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量(手)
            - amount: 成交额(元)
            - change: 涨跌额
            - pct_chg: 涨跌幅(%)
            
        数据按照trade_date降序排列，即最新日期在最前面
    
    Examples:
        >>> # 使用股票代码查询
        >>> df = get_daily_bar('000001')
        >>> # 使用股票名称查询
        >>> df = get_daily_bar('平安银行')
        >>> # 指定日期范围
        >>> df = get_daily_bar('000001', '20230101', '20231231')
        >>> # 使用东方财富数据源
        >>> df = get_daily_bar('000001', is_tushare=False)
    """
    # 处理默认日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)
        start_date = start_dt.strftime('%Y%m%d')
    
    if is_tushare:
        # 使用Tushare数据源，get_stock_daily内部会处理代码转换
        df = get_stock_daily(symbol, start_date, end_date)
        
        # 选择和重命名需要的列
        df = df[['trade_date', 'open', 'high', 'low', 'close', 
                'vol', 'amount', 'change', 'pct_chg']]
        df = df.rename(columns={'vol': 'volume'})
        
    else:
        # 使用东方财富数据源，需要手动处理代码转换
        matcher = TsCodeMatcher()
        # 获取不带后缀的代码
        pure_symbol = matcher[symbol].split('.')[0]
        
        df = get_daily_data(pure_symbol, start_date, end_date)
        
        # 统一列名格式
        df = df.rename(columns={
            'date': 'trade_date',
            'pct_change': 'pct_chg',
            'price_change': 'change'
        })
        
        # 选择需要的列
        df = df[['trade_date', 'open', 'high', 'low', 'close', 
                'volume', 'amount', 'change', 'pct_chg']]
        
        # 确保trade_date格式统一为YYYYMMDD
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
    
    # 统一数据类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'amount', 'change', 'pct_chg']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # 按日期降序排序
    df = df.sort_values('trade_date', ascending=False).reset_index(drop=True)
    
    return df


@function_cache.cache(ttl=60)
def get_stock_fund_flow( indicator: Literal[ "即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时") -> Dict[str, Dict]:
        """
        获取个股资金流量表，参数 indicator: Literal[ "即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时" 返回值 Dict[str, Dict]
        """
        # 获取资金流数据
        stock_fund_flow_individual_df = ak.stock_fund_flow_individual(symbol=indicator)
        
        # 将DataFrame转换为字典
        result = {}
        for _, row in stock_fund_flow_individual_df.iterrows():
            stock_code = row['股票代码']
            row_dict = row.to_dict()
            result[stock_code] = row_dict
        
        return result

@function_cache.cache(ttl=60)
def get_top_flow_concept( indicator: Literal["即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时", topn: int=20) -> List[Dict]:
    """
    获取概念资金流量表，带有顺序重试机制
    
    Args:
        indicator: 时间周期，默认"即时"
        topn: 返回前多少条数据，默认20
        
    Returns:
        List[Dict]: 概念资金流数据列表
    """
    # 定义重试顺序
    retry_sequence = ["即时", "3日排行", "5日排行", "10日排行", "20日排行"]
    
    # 如果指定的indicator不是"即时"，将重试序列重新排序，从指定的indicator开始
    if indicator != "即时":
        try:
            start_index = retry_sequence.index(indicator)
            retry_sequence = retry_sequence[start_index:] + retry_sequence[:start_index]
        except ValueError:
            logger.warning(f"无效的indicator值: {indicator}，使用默认顺序")
    
    last_exception = None
    
    # 按顺序尝试每个周期
    for period in retry_sequence:
        try:
            df = ak.stock_fund_flow_concept(symbol=period)
            if not df.empty:
                result = df.head(topn).to_dict(orient="records")
                if result:  # 确保结果不为空
                    if period != indicator:
                        logger.info(f"使用备选周期 {period} 获取数据成功")
                    return result
        except Exception as e:
            last_exception = e
            logger.warning(f"获取 {period} 周期数据失败: {str(e)}")
            continue
    
    # 如果所有尝试都失败了
    error_msg = f"所有周期都获取失败，最后一次错误: {str(last_exception)}" if last_exception else "所有周期都获取失败"
    logger.error(error_msg)
    return []  # 返回空列表而不是抛出异常，保证服务的稳定性

@function_cache.cache(ttl=60)
def get_stock_fund_flow_concept(indicator: Literal["即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时") -> Dict[str, Dict]:
    """
    获取概念资金流量表，参数 indicator: Literal[ "即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时" 返回值 Dict[str, Dict]
    """
    stock_fund_flow_concept_df = ak.stock_fund_flow_concept(symbol=indicator)
    result = {}
    for _, row in stock_fund_flow_concept_df.iterrows():
        stock_code = row['行业']
        row_dict = row.to_dict()
        result[stock_code] = row_dict
    
    return result

@function_cache.cache(ttl=60)
def get_industry_fund_flow(indicator: Literal["即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时") -> Dict[str, Dict]:
    """
    获取行业资金流量表，参数 indicator: Literal[ "即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时" 返回值 Dict[str, Dict]
    """
    # 获取行业资金流数据
    stock_fund_flow_industry_df = ak.stock_fund_flow_industry(symbol=indicator)
    if stock_fund_flow_industry_df is None or stock_fund_flow_industry_df.empty or len(stock_fund_flow_industry_df.columns) == 0:
        stock_fund_flow_industry_df = ak.stock_sector_fund_flow_rank(indicator="3日排行")
    
    # 将DataFrame转换为字典
    result = {}
    for _, row in stock_fund_flow_industry_df.iterrows():
        industry = row['行业']
        row_dict = row.to_dict()
        result[industry] = row_dict
    
    return result