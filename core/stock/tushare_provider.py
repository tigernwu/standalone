import tushare as ts
import os
import pandas as pd
from core.utils.config_setting import Config
from core.utils.ts_data import tsdata
from core.stock.ts_code_matcher import TsCodeMatcher
from typing import List, Optional
import datetime
from functools  import lru_cache
from cachetools import cached,LFUCache

lfu_cache = LFUCache(maxsize=128)

config = Config()
TUSHARE_TOKEN = config.get("ts_key")
ts.set_token(TUSHARE_TOKEN)

@lru_cache(maxsize=4)
def trade_calendar_last_trade_day(from_date:Optional[datetime.datetime]=None,allow_in_trading=False,in_trading_limit_time=15)->datetime.datetime:
    """
    获取上一个交易日
    """
    pro = ts.pro_api()
    if from_date is None:
        from_date = datetime.datetime.now()

    from_date_str = from_date.strftime("%Y%m%d")
    df = pro.trade_cal(exchange="SSE",start_date=from_date_str,end_date=from_date_str,limit=1)
    is_open = df.iloc[0]['is_open']
    pretrade_date=df.iloc[0]['pretrade_date']
    if is_open==1 and allow_in_trading:
        from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
        return from_date
    elif is_open==1   and  from_date.hour >= in_trading_limit_time:
        return from_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        return datetime.datetime.strptime(pretrade_date,"%Y%m%d")

def get_last_trading_dates( days:int=3)->list:
    """
    获取最近几个交易日
    """
    pro = ts.pro_api()
    now = datetime.datetime.now().strftime("%Y%m%d")
    df = pro.trade_cal(exchange="SSE",limit=days,end_date = now ,is_open="1")
    return df["cal_date"].to_list()


def fetch_realtime_stock_data( src='dc'):
    """
    获取股票实时交易数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    src (str): 数据源，可选值为 'sina' 或 'dc'，默认 'dc'。
    offset (int): 数据偏移量，默认 0。
    limit (int): 单次获取数据的最大数量，默认 1000。

    返回:
    pandas.DataFrame: 包含实时交易数据的DataFrame。
    """

    pro = ts.pro_api()

    df = ts.realtime_list(src=src)

    if 'TS_CODE' in df.columns:
        df['code'] = df['TS_CODE'].str.split('.').str[0]
        df.drop(columns=['TS_CODE'], inplace=True)

    return df

@tsdata
def get_management_compensation(symbol, end_date=None, offset=0, limit=1000):
    """
    获取上市公司管理层薪酬和持股数据

    参数:
    symbol (str): 股票代码，支持单个或多个代码输入，格式为市场代码（如'000001.SZ'）
    end_date (str, optional): 报告期，格式为YYYYMMDD
    offset (int, optional): 数据偏移量，默认值为0
    limit (int, optional): 单次获取数据的最大数量，默认值为1000

    返回:
    DataFrame: 包含管理层薪酬和持股数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.stk_rewards(ts_code=ts_code, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_hs_constituents(hs_type: str, is_new: str = '1', offset: int = 0, limit: int = 5000) -> pd.DataFrame:
    """
    获取沪股通或深股通的成分数据。

    参数:
    hs_type (str): 类型，'SH'表示沪股通，'SZ'表示深股通。
    is_new (str, optional): 是否最新，'1'表示最新，'0'表示非最新，默认为'1'。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为5000。

    返回:
    pd.DataFrame: 包含沪股通或深股通成分数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.hs_const(hs_type=hs_type, is_new=is_new, offset=offset, limit=limit)
    
    # 去掉ts_code中的后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+', '', regex=True)
    
    return df

@tsdata
def fetch_realtime_ticks(symbol: str, src: str = 'sina', offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取股票的实时分笔成交数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    src (str): 数据源，可选 'sina' 或 'dc'，默认 'sina'。
    offset (int): 数据偏移量，默认 0。
    limit (int): 每次获取的数据量，默认 1000。

    返回:
    pd.DataFrame: 包含实时分笔成交数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]
    
    df = ts.realtime_tick(ts_code=ts_code, src=src)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_new_share_data(start_date=None, end_date=None, symbol=None, offset=0, limit=2000):
    """
    获取新股上市列表数据

    参数:
    start_date (str): 上网发行开始日期，格式为YYYYMMDD
    end_date (str): 上网发行结束日期，格式为YYYYMMDD
    symbol (str): 股票代码，可以是A股、港股、美股、指数、期货的代码
    offset (int): 数据偏移量，默认从0开始
    limit (int): 单次获取数据的最大条数，默认2000条

    返回:
    DataFrame: 包含新股上市列表数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    # 如果提供了symbol，将其转换为ts_code
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.new_share(start_date=start_date, end_date=end_date, ts_code=ts_code, offset=offset, limit=limit)
    
    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_real_time_quotes(symbol: str, src: str = 'dc', offset: int = 0, limit: int = 50) -> pd.DataFrame:
    """
    获取实时股票行情数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    src (str): 数据源，可选值为 'sina' 或 'dc'，默认 'dc'。
    offset (int): 数据偏移量，默认 0。
    limit (int): 数据条数限制，默认 50（单次取数据上限）。

    返回:
    pd.DataFrame: 包含实时股票行情数据的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    # 调用tushare的realtime_quote函数获取数据
    df = ts.realtime_quote(ts_code=ts_code, src=src)

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_stock_company_info(symbol=None, exchange=None, offset=0, limit=4500):
    """
    获取上市公司基础信息。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    exchange (str, optional): 交易所代码，SSE上交所，SZSE深交所，BSE北交所。
    offset (int, optional): 数据偏移量，默认0。
    limit (int, optional): 单次提取数据量，默认4500。

    返回:
    pd.DataFrame: 包含上市公司基础信息的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.stock_company(ts_code=ts_code, exchange=exchange, fields='ts_code,com_name,com_id,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,introduction,website,email,office,employees,main_business,business_scope',offset=offset,limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_company_management(symbol=None, ann_date=None, start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取上市公司管理层信息

    参数:
    symbol (str, optional): 股票代码，支持单个或多个股票输入，默认为None
    ann_date (str, optional): 公告日期（YYYYMMDD格式），默认为None
    start_date (str, optional): 公告开始日期，默认为None
    end_date (str, optional): 公告结束日期，默认为None
    offset (int, optional): 数据偏移量，默认为0
    limit (int, optional): 单次获取数据的最大数量，默认为1000

    返回:
    pd.DataFrame: 包含上市公司管理层信息的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None
    
    pro = ts.pro_api()
    df = pro.stk_managers(ts_code=ts_code, ann_date=ann_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

def fetch_stock_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取股票的日行情数据，支持前后复权数据。

    参数:
    symbol (str): 股票代码（支持多个股票同时提取，逗号分隔）
    trade_date (str): 交易日期（YYYYMMDD）
    start_date (str): 开始日期(YYYYMMDD)
    end_date (str): 结束日期(YYYYMMDD)
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大数量，默认5000

    返回:
    pd.DataFrame: 包含股票日行情数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()

    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    pro = ts.pro_api()
    df = pro.daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_historical_name_changes(
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取历史名称变更记录的函数封装。

    参数:
    symbol (str, optional): 股票代码，可以是A股、港股、美股、指数、期货的代码。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大数量，默认值为1000。

    返回:
    pd.DataFrame: 包含历史名称变更记录的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.namechange(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        fields='ts_code,name,start_date,end_date,change_reason'
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_stock_backup_data(trade_date: str = None, symbol: str = None, offset: int = 0, limit: int = 7000) -> pd.DataFrame:
    """
    获取股票备用基础数据，数据从2016年开始。

    参数:
    trade_date (str, 可选): 交易日期，格式为YYYYMMDD。
    symbol (str, 可选): 股票代码，支持A股、港股、美股、指数、期货。
    offset (int, 可选): 数据偏移量，默认值为0。
    limit (int, 可选): 单次获取数据的最大条数，默认值为7000。

    返回:
    pd.DataFrame: 包含股票备用基础数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.bak_basic(trade_date=trade_date, ts_code=ts_code, fields='trade_date,ts_code,name,industry,area,pe,float_share,total_share,total_assets,liquid_assets,fixed_assets,reserved,reserved_pershare,eps,bvps,pb,list_date,undp,per_undp,rev_yoy,profit_yoy,gpr,npr,holder_num',offset=offset,limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

def get_futures_trade_calendar(exchange: str = None, start_date: str = None, end_date: str = None, is_open: int = None, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取各大期货交易所的交易日历数据。

    参数:
    exchange (str, 可选): 交易所代码，如 'SHFE' 上期所, 'DCE' 大商所, 'CFFEX' 中金所, 'CZCE' 郑商所, 'INE' 上海国际能源交易所。
    start_date (str, 可选): 开始日期，格式为 'YYYYMMDD'。
    end_date (str, 可选): 结束日期，格式为 'YYYYMMDD'。
    is_open (int, 可选): 是否交易，0 表示休市，1 表示交易。
    offset (int, 可选): 数据偏移量，默认值为 0。
    limit (int, 可选): 单次返回的数据条数，默认值为 1000。

    返回:
    pd.DataFrame: 包含交易所交易日历数据的 DataFrame。
    """
    pro = ts.pro_api()
    df = pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date, is_open=is_open, offset=offset, limit=limit)
    
    # 如果输出结果包含 ts_code 列，去掉 .XX 后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_a_weekly_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 4500,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取A股周线行情数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期（每周最后一个交易日期，YYYYMMDD格式）。
    start_date (str): 开始日期。
    end_date (str): 结束日期。
    limit (int): 单次获取数据的最大行数，默认4500行。
    offset (int): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含周线行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.weekly(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,trade_date,open,high,low,close,vol,amount',
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_stock_daily_basic(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取股票每日重要的基本面指标，可用于选股分析、报表展示等。

    参数:
    symbol (str): 股票代码（A股、港股、美股、指数、期货），可选
    trade_date (str): 交易日期（YYYYMMDD），可选
    start_date (str): 开始日期(YYYYMMDD)，可选
    end_date (str): 结束日期(YYYYMMDD)，可选
    fields (str): 需要返回的字段，可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次返回数据量，默认5000

    返回:
    pd.DataFrame: 包含每日基本面指标的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.daily_basic(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_a_stock_monthly_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = 'ts_code,trade_date,open,high,low,close,vol,amount',
    offset: int = 0,
    limit: int = 4500
) -> pd.DataFrame:
    """
    获取A股月线数据

    :param symbol: 股票代码 (symbol, trade_date 两个参数任选一)
    :param trade_date: 交易日期 (每月最后一个交易日日期，YYYYMMDD格式)
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param fields: 返回字段，默认返回所有字段
    :param offset: 数据偏移量，默认从0开始
    :param limit: 单次最大返回数据量，默认4500行
    :return: DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.monthly(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_stock_adj_factors(symbol: str = None, trade_date: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取股票复权因子数据。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为1000。

    返回:
    pd.DataFrame: 包含复权因子数据的DataFrame。
    """
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = ''
    
    df = pro.adj_factor(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_stock_limit_prices(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 5800
) -> pd.DataFrame:
    """
    获取全市场（包含A/B股和基金）每日涨跌停价格，包括涨停价格，跌停价格等。
    每个交易日8点40左右更新当日股票涨跌停价格。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次提取记录数，默认值为5800，最大值为5800。

    返回:
    pd.DataFrame: 包含交易日期、股票代码、涨停价、跌停价的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.stk_limit(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_stock_weekly_monthly_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = "week",
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取股票的周/月线行情数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    freq (str): 频率，可选值为"week"或"month"。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大数量，默认值为6000。

    返回:
    pd.DataFrame: 包含股票周/月线行情数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    pro = ts.pro_api()

    # 调用底层API获取数据
    data = pro.stk_weekly_monthly(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        offset=offset,
        limit=limit
    )

    # 处理数据
    df = pd.DataFrame(data)
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_stock_premarket_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 8000
) -> pd.DataFrame:
    """
    获取每日开盘前股票的股本情况，包括总股本和流通股本，涨跌停价格等。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大条数，默认为8000。

    返回:
    pd.DataFrame: 包含交易日期、股票代码、总股本、流通股本、昨日收盘价、今日涨停价、今日跌停价的DataFrame。
    """
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    
    # 将symbol转换为ts_code
    ts_code = matcher[symbol] if symbol else None
    
    # 调用Tushare接口获取数据
    df = pro.stk_premarket(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )
    
    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_stock_basic_info(
    ts_code: str = None,
    name: str = None,
    market: str = None,
    list_status: str = 'L',
    exchange: str = None,
    is_hs: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取股票基础信息数据，包括股票代码、名称、上市日期、退市日期等。

    参数:
    ts_code (str): TS股票代码，可选。
    name (str): 股票名称，可选。
    market (str): 市场类别（主板/创业板/科创板/CDR/北交所），可选。
    list_status (str): 上市状态（L上市 D退市 P暂停上市），默认是L。
    exchange (str): 交易所（SSE上交所 SZSE深交所 BSE北交所），可选。
    is_hs (str): 是否沪深港通标的（N否 H沪股通 S深股通），可选。
    offset (int): 数据偏移量，默认是0。
    limit (int): 每次获取的数据量，默认是5000。

    返回:
    pd.DataFrame: 包含股票基础信息的DataFrame。
    """
    # 如果提供了symbol，转换为ts_code
    if ts_code:
        matcher = TsCodeMatcher()
        ts_code = matcher[ts_code]

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    data = pro.stock_basic(
        ts_code=ts_code,
        name=name,
        market=market,
        list_status=list_status,
        exchange=exchange,
        is_hs=is_hs,
        offset=offset,
        limit=limit
    )

    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in data.columns:
        data['code'] = data['ts_code'].str.split('.').str[0]

    return data

@tsdata
def fetch_ggt_daily_data(trade_date=None, start_date=None, end_date=None, limit=1000, offset=0):
    """
    获取港股通每日成交信息。

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD，支持单日和多日输入。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    limit (int, optional): 单次获取数据的最大数量，默认1000。
    offset (int, optional): 数据偏移量，默认0。

    返回:
    pandas.DataFrame: 包含港股通每日成交信息的DataFrame。
    """
    pro = ts.pro_api()
    
    # 获取数据
    df = pro.ggt_daily(trade_date=trade_date, start_date=start_date, end_date=end_date, limit=limit, offset=offset)
    
    # 如果结果包含ts_code列，去掉后缀.XX
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_hk_monthly_trading_stats(month=None, start_month=None, end_month=None, limit=1000, offset=0):
    """
    获取港股通每月成交信息。

    参数:
    month (str, optional): 月度（格式YYYYMM，支持多个输入，逗号分隔）
    start_month (str, optional): 开始月度（格式YYYYMM）
    end_month (str, optional): 结束月度（格式YYYYMM）
    limit (int, optional): 单次返回数据条数，默认1000
    offset (int, optional): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含港股通每月成交信息的DataFrame
    """
    pro = ts.pro_api()
    
    # 获取数据
    df = pro.ggt_monthly(trade_date=month, start_date=start_month, end_date=end_month, limit=limit, offset=offset)
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_stock_suspend_info(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    suspend_type: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取股票每日停复牌信息

    参数:
    symbol (str, optional): 股票代码(可输入多值)
    trade_date (str, optional): 交易日日期
    start_date (str, optional): 停复牌查询开始日期
    end_date (str, optional): 停复牌查询结束日期
    suspend_type (str, optional): 停复牌类型：S-停牌,R-复牌
    offset (int, optional): 数据偏移量，默认0
    limit (int, optional): 单次返回数据量，默认5000

    返回:
    pd.DataFrame: 包含停复牌信息的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.suspend_d(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        suspend_type=suspend_type,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_hsgt_top10_data(
    trade_date: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    market_type: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取沪股通、深股通每日前十大成交详细数据。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    market_type (str): 市场类型，1表示沪市，3表示深市。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 每次获取的数据量，默认1000条。

    返回:
    pd.DataFrame: 包含沪股通、深股通每日前十大成交详细数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    data = pro.hsgt_top10(
        trade_date=trade_date,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in data.columns:
        data['code'] = data['ts_code'].str.split('.').str[0]
    
    return data

@tsdata
def get_financial_audit_data(symbol, ann_date=None, start_date=None, end_date=None, period=None, offset=0, limit=1000):
    """
    获取上市公司定期财务审计意见数据

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货
    ann_date (str, optional): 公告日期
    start_date (str, optional): 公告开始日期
    end_date (str, optional): 公告结束日期
    period (str, optional): 报告期(每个季度最后一天的日期,比如20171231表示年报)
    offset (int, optional): 数据偏移量，默认0
    limit (int, optional): 单次获取数据量，默认1000

    返回:
    pd.DataFrame: 包含审计意见数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.fina_audit(ts_code=ts_code, ann_date=ann_date, start_date=start_date, end_date=end_date, period=period, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def get_financial_indicators(symbol, ann_date=None, start_date=None, end_date=None, period=None):
    """
    获取上市公司最新一条财务指标数据,并格式化为中文键的字典。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str, optional): 公告日期。
    start_date (str, optional): 报告期开始日期。
    end_date (str, optional): 报告期结束日期。
    period (str, optional): 报告期（每个季度最后一天的日期，比如20171231表示年报）。

    返回:
    dict: 包含最新财务指标的中文键值对字典。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    # 字段中文映射
    cn_mapping = {
        'ts_code': '股票代码',
        'ann_date': '公告日期',
        'end_date': '报告期',
        'eps': '每股收益',
        'dt_eps': '稀释每股收益',
        'total_revenue_ps': '每股营业总收入',
        'revenue_ps': '每股营业收入',
        'capital_rese_ps': '每股资本公积',
        'surplus_rese_ps': '每股盈余公积',
        'undist_profit_ps': '每股未分配利润',
        'extra_item': '非经常性损益',
        'profit_dedt': '扣除非经常性损益后的净利润',
        'gross_margin': '毛利',
        'current_ratio': '流动比率',
        'quick_ratio': '速动比率',
        'cash_ratio': '保守速动比率',
        'ar_turn': '应收账款周转率',
        'ca_turn': '流动资产周转率',
        'fa_turn': '固定资产周转率',
        'assets_turn': '总资产周转率',
        'op_income': '经营活动净收益',
        'ebit': '息税前利润',
        'ebitda': '息税折旧摊销前利润',
        'netdebt': '净债务',
        'working_capital': '营运资金',
        'networking_capital': '营运流动资本',
        'invest_capital': '全部投入资本',
        'retained_earnings': '留存收益',
        'bps': '每股净资产',
        'ocfps': '每股经营活动现金流量净额',
        'retainedps': '每股留存收益',
        'cfps': '每股现金流量净额',
        'netprofit_margin': '销售净利率',
        'grossprofit_margin': '销售毛利率',
        'roe': '净资产收益率',
        'roa': '总资产报酬率',
        'npta': '总资产净利润',
        'roic': '投入资本回报率',
        'roe_yearly': '年化净资产收益率',
        'debt_to_assets': '资产负债率',
        'op_yoy': '营业利润同比增长率',
        'tr_yoy': '营业总收入同比增长率',
        'netprofit_yoy': '净利润同比增长率'
    }

    pro = ts.pro_api()
    df = pro.fina_indicator(ts_code=ts_code, ann_date=ann_date, start_date=start_date, end_date=end_date, period=period)
    
    if df.empty:
        return {}
    
    # 获取最新一条数据
    latest_data = df.iloc[0].to_dict()
    
    # 转换为中文键的字典
    result = {}
    for key, value in latest_data.items():
        if key in cn_mapping:
            # 处理数值格式
            if isinstance(value, float):
                if abs(value) > 1:
                    value = round(value, 2)
                else:
                    value = round(value, 4)
            result[cn_mapping[key]] = value
            
    return result

@tsdata
def fetch_stock_performance(symbol, ann_date=None, start_date=None, end_date=None, period=None, offset=0, limit=1000):
    """
    获取单只股票的历史业绩快报数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str, optional): 公告日期，格式为YYYYMMDD。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。
    period (str, optional): 报告期，格式为YYYYMMDD，表示每个季度最后一天的日期。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大数量，默认值为1000。

    返回:
    pandas.DataFrame: 包含业绩快报数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.express(ts_code=ts_code, ann_date=ann_date, start_date=start_date, end_date=end_date, period=period, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_ggt_daily_trades(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    market_type: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取港股通每日成交数据，其中包括沪市、深市详细数据，每天18~20点之间完成当日更新。

    参数:
    symbol (str): 股票代码（可选），支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期（可选），格式为YYYYMMDD。
    start_date (str): 开始日期（可选），格式为YYYYMMDD。
    end_date (str): 结束日期（可选），格式为YYYYMMDD。
    market_type (str): 市场类型（可选），2：港股通（沪），4：港股通（深）。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大数量，默认值为1000。

    返回:
    pd.DataFrame: 包含港股通每日成交数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.ggt_top10(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

def fetch_stock_forecast_data(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    type: str = None,
    offset: int = 0,
    limit: int = 1000
) -> list:
    """
    获取股票业绩预告数据。

    参数:
    symbol (str): 股票代码 (可选)
    ann_date (str): 公告日期 (可选)
    start_date (str): 公告开始日期 (可选)
    end_date (str): 公告结束日期 (可选)
    period (str): 报告期 (可选)
    type (str): 预告类型 (可选)
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认1000

    返回:
    list: 包含业绩预告数据的字典列表，所有记录具有相同的公告日期
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.forecast(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        period=period,
        type=type,
        offset=offset,
        limit=limit
    )
    
    if df.empty:
        return []
    
    # 获取第一条记录的公告日期
    first_ann_date = df['ann_date'].iloc[0]
    
    # 筛选相同公告日期的数据
    df = df[df['ann_date'] == first_ann_date]
    
    # 定义字段映射关系
    field_mapping = {
        'ts_code': '股票代码',
        'ann_date': '公告日期',
        'end_date': '报告期',
        'type': '预告类型',
        'p_change_min': '预告净利润变动幅度下限',
        'p_change_max': '预告净利润变动幅度上限',
        'net_profit_min': '预告净利润下限',
        'net_profit_max': '预告净利润上限',
        'last_parent_net': '上年同期归属母公司净利润',
        'first_ann_date': '首次公告日',
        'summary': '业绩预告摘要',
        'change_reason': '业绩变动原因'
    }
    
    # 转换为字典列表
    result = []
    for _, row in df.iterrows():
        item = {}
        for eng_key, cn_key in field_mapping.items():
            if eng_key in row.index:
                # 处理股票代码，去掉后缀
                if eng_key == 'ts_code':
                    item[cn_key] = row[eng_key].split('.')[0]
                else:
                    item[cn_key] = row[eng_key]
        result.append(item)
    
    return result

@tsdata
def fetch_backup_market_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 7000,
    fields: str = None
) -> pd.DataFrame:
    """
    获取备用行情数据，包括特定的行情指标。数据从2017年中左右开始，早期有几天数据缺失，近期正常。
    
    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 开始行数，默认值为0。
    limit (int): 最大行数，默认值为7000，根据文档单次最大7000行数据。
    fields (str): 需要返回的字段，多个字段用逗号分隔。
    
    返回:
    pd.DataFrame: 包含备用行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.bak_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        fields=fields
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_cashflow_data(symbol, start_date=None, end_date=None, period=None, report_type=None, comp_type=None, is_calc=None, offset=0, limit=1000):
    """
    获取上市公司现金流量表数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。
    period (str, optional): 报告期，格式为YYYYMMDD。
    report_type (str, optional): 报告类型，见文档详细说明。
    comp_type (str, optional): 公司类型，1一般工商业 2银行 3保险 4证券。
    is_calc (int, optional): 是否计算报表。
    offset (int, optional): 数据偏移量，默认0。
    limit (int, optional): 单次获取数据条数，默认1000。

    返回:
    pd.DataFrame: 包含现金流量表数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date, period=period, report_type=report_type, comp_type=comp_type, is_calc=is_calc, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_institutional_trades(trade_date: str, symbol: str = None, offset: int = 0, limit: int = 10000) -> pd.DataFrame:
    """
    获取龙虎榜机构成交明细数据

    :param trade_date: 交易日期，格式为YYYYMMDD
    :param symbol: 股票代码，可选，默认为None。支持A股、港股、美股、指数、期货。
    :param offset: 数据偏移量，默认为0
    :param limit: 数据条数限制，默认为10000
    :return: 包含机构成交明细的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.top_inst(trade_date=trade_date, ts_code=ts_code, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_top_list_data(trade_date: str, symbol: str = None, offset: int = 0, limit: int = 10000) -> pd.DataFrame:
    """
    获取龙虎榜每日交易明细数据。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为10000。

    返回:
    pd.DataFrame: 包含龙虎榜交易明细的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.top_list(trade_date=trade_date, ts_code=ts_code, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_dividend_data(
    symbol: str = None,
    ann_date: str = None,
    record_date: str = None,
    ex_date: str = None,
    imp_ann_date: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取分红送股数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str): 公告日，格式为YYYYMMDD。
    record_date (str): 股权登记日期，格式为YYYYMMDD。
    ex_date (str): 除权除息日，格式为YYYYMMDD。
    imp_ann_date (str): 实施公告日，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大数量，默认值为5000。

    返回:
    pd.DataFrame: 包含分红送股数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.dividend(
        ts_code=ts_code,
        ann_date=ann_date,
        record_date=record_date,
        ex_date=ex_date,
        imp_ann_date=imp_ann_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def fetch_top10_float_holders(symbol: str, period: str = None, ann_date: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 100):
    """
    获取上市公司前十大流通股东数据

    :param symbol: 股票代码，支持A股、港股、美股、指数、期货
    :param period: 报告期（YYYYMMDD格式，一般为每个季度最后一天）
    :param ann_date: 公告日期
    :param start_date: 报告期开始日期
    :param end_date: 报告期结束日期
    :param offset: 数据偏移量，默认0
    :param limit: 单次获取数据量，默认100，最大100
    :return: DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]
    
    pro = ts.pro_api()
    df = pro.top10_floatholders(ts_code=ts_code, period=period, ann_date=ann_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_stock_main_business_composition(
    symbol: str,
    period: str = None,
    type: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 100,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取上市公司主营业务构成，分地区和产品两种方式。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    period (str, optional): 报告期(每个季度最后一天的日期,比如20171231表示年报)。
    type (str, optional): 类型：P按产品 D按地区 I按行业（请输入大写字母P或者D）。
    start_date (str, optional): 报告期开始日期。
    end_date (str, optional): 报告期结束日期。
    limit (int, optional): 单次提取数据的最大行数，默认100。
    offset (int, optional): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含主营业务构成的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.fina_mainbz(
        ts_code=ts_code,
        period=period,
        type=type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def fetch_financial_disclosure_dates(
    symbol: str = None,
    end_date: str = None,
    pre_date: str = None,
    actual_date: str = None,
    offset: int = 0,
    limit: int = 3000
) -> pd.DataFrame:
    """
    获取财报披露计划日期

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    end_date (str): 财报周期（每个季度最后一天的日期，比如20181231表示2018年年报，20180630表示中报）。
    pre_date (str): 计划披露日期。
    actual_date (str): 实际披露日期。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据量，默认3000。

    返回:
    pd.DataFrame: 包含财报披露计划日期的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.disclosure_date(
        ts_code=ts_code,
        end_date=end_date,
        pre_date=pre_date,
        actual_date=actual_date,
        offset=offset,
        limit=limit
    )

    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_top_10_shareholders(symbol: str, period: str = None, ann_date: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取上市公司前十大股东数据，包括持有数量和比例等信息。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    period (str, optional): 报告期（YYYYMMDD格式，一般为每个季度最后一天）。
    ann_date (str, optional): 公告日期。
    start_date (str, optional): 报告期开始日期。
    end_date (str, optional): 报告期结束日期。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大数量，默认值为1000。

    返回:
    pd.DataFrame: 包含前十大股东数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.top10_holders(ts_code=ts_code, period=period, ann_date=ann_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_company_balance_sheet(
    symbol: str,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    report_type: str = None,
    comp_type: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取上市公司资产负债表数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str, optional): 公告日期，格式为YYYYMMDD。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。
    period (str, optional): 报告期，格式为YYYYMMDD。
    report_type (str, optional): 报告类型，见下方详细说明。
    comp_type (str, optional): 公司类型，1一般工商业 2银行 3保险 4证券。
    offset (int, optional): 数据偏移量，默认0。
    limit (int, optional): 单次获取数据量，默认1000。

    返回:
    pd.DataFrame: 包含资产负债表数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.balancesheet(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        period=period,
        report_type=report_type,
        comp_type=comp_type,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_financial_income_data(
    symbol: str,
    ann_date: str = None,
    f_ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    report_type: str = None,
    comp_type: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取上市公司财务利润表数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str, optional): 公告日期（YYYYMMDD格式）。
    f_ann_date (str, optional): 实际公告日期。
    start_date (str, optional): 公告开始日期。
    end_date (str, optional): 公告结束日期。
    period (str, optional): 报告期（每个季度最后一天的日期，比如20171231表示年报，20170630半年报，20170930三季报）。
    report_type (str, optional): 报告类型，参考文档最下方说明。
    comp_type (str, optional): 公司类型（1一般工商业2银行3保险4证券）。
    offset (int, optional): 数据偏移量，默认0。
    limit (int, optional): 单次获取数据量，默认1000。

    返回:
    pd.DataFrame: 包含财务利润表数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.income(
        ts_code=ts_code,
        ann_date=ann_date,
        f_ann_date=f_ann_date,
        start_date=start_date,
        end_date=end_date,
        period=period,
        report_type=report_type,
        comp_type=comp_type,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def fetch_stock_pledge_details(symbol: str, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
    """
    获取股票质押明细数据

    :param symbol: 股票代码，支持A股、港股、美股、指数、期货
    :param limit: 单次获取数据的最大数量，默认1000
    :param offset: 数据偏移量，默认0
    :return: 包含股票质押明细数据的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]
    
    pro = ts.pro_api()
    df = pro.pledge_detail(ts_code=ts_code, limit=limit, offset=offset)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df


@tsdata
def get_stock_pledge_stats(symbol=None, end_date=None, offset=0, limit=1000):
    """
    获取股票质押统计数据

    :param symbol: str, 股票代码（A股、港股、美股、指数、期货），可选
    :param end_date: str, 截止日期，格式为YYYYMMDD，可选
    :param offset: int, 数据偏移量，默认0
    :param limit: int, 单次获取数据的最大数量，默认1000
    :return: pd.DataFrame, 包含股票质押统计数据的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.pledge_stat(ts_code=ts_code, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df


@tsdata
def fetch_concept_categories(src: str = 'ts', offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取概念股分类数据。

    参数:
    src (str): 数据来源，默认为'ts'。
    offset (int): 数据偏移量，默认为0。
    limit (int): 每次获取的数据量，默认为1000。

    返回:
    pd.DataFrame: 包含概念分类数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.concept(src=src,offset=offset,limit=limit)
    
    # 如果输出结果包含ts_code列，去掉000000.XX的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_concept_details(id=None, symbol=None, fields=None, offset=0, limit=1000):
    """
    获取概念股分类明细数据

    :param id: 概念分类ID（id来自概念股分类接口）
    :param symbol: 股票代码（A股，港股，美股，指数，期货）
    :param fields: 返回字段（可选）
    :param offset: 数据偏移量（默认0）
    :param limit: 数据条数限制（默认1000）
    :return: DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.concept_detail(id=id, ts_code=ts_code, fields=fields, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_stock_account_data(start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取股票账户开户数据旧版格式数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大数量，默认1000。

    返回:
    DataFrame: 包含股票账户开户数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.stk_account_old(start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_stock_account_data(start_date=None, end_date=None, date=None, offset=0, limit=1000):
    """
    获取股票账户开户数据，统计周期为一周。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    date (str): 具体日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大数量，默认1000。

    返回:
    pandas.DataFrame: 包含股票账户开户数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.stk_account(start_date=start_date, end_date=end_date, date=date, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_stock_repurchase_data(
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    symbol: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取上市公司回购股票数据。

    参数:
    ann_date (str, optional): 公告日期，格式为YYYYMMDD。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次返回数据条数，默认值为2000。

    返回:
    pd.DataFrame: 包含回购股票数据的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.repurchase(
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        ts_code=ts_code,
        offset=offset,
        limit=limit
    )

    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_unlocked_shares(symbol=None, ann_date=None, float_date=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取限售股解禁数据

    参数:
    symbol (str): 股票代码（可选）
    ann_date (str): 公告日期（格式：YYYYMMDD，可选）
    float_date (str): 解禁日期（格式：YYYYMMDD，可选）
    start_date (str): 解禁开始日期（格式：YYYYMMDD，可选）
    end_date (str): 解禁结束日期（格式：YYYYMMDD，可选）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认5000（单次最大5000条）

    返回:
    DataFrame: 包含解禁数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.share_float(ts_code=ts_code, ann_date=ann_date, float_date=float_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_stock_holder_numbers(symbol=None, ann_date=None, enddate=None, start_date=None, end_date=None, offset=0, limit=3000):
    """
    获取上市公司股东户数数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    ann_date (str): 公告日期。
    enddate (str): 截止日期。
    start_date (str): 公告开始日期。
    end_date (str): 公告结束日期。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据的最大数量，默认3000。

    返回:
    pd.DataFrame: 包含股东户数数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.stk_holdernumber(ts_code=ts_code, ann_date=ann_date, enddate=enddate, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

def fetch_block_trades(symbol: str = None) -> list:
    """
    获取最近两个交易日的大宗交易数据
    只返回最近5个交易日内的数据，如果不在范围内则返回空列表

    参数:
    symbol (str, optional): 股票代码，默认为None

    返回:
    list: 包含大宗交易数据的列表，每个元素为字典，包含中文键值
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    import pandas as pd
    from datetime import datetime, timedelta

    # 获取当前日期
    today = datetime.now().strftime('%Y%m%d')
    # 设置一个较早的起始日期，确保能获取到足够的交易日
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    # 获取最近5个交易日
    pro = ts.pro_api()
    trade_calendar = get_futures_trade_calendar(
        exchange="SSE",
        start_date=start_date,
        end_date=today,
        is_open=1,
        limit=5
    )
    
    if trade_calendar.empty:
        return []
        
    # 获取最近5个交易日的日期列表
    recent_trade_dates = set(trade_calendar['cal_date'].tolist())
    
    # 字段映射表
    field_mapping = {
        'ts_code': '股票代码',
        'trade_date': '交易日期',
        'price': '成交价',
        'vol': '成交量',
        'amount': '成交金额',
        'buyer': '买方营业部',
        'seller': '卖方营业部'
    }
    
    # 如果提供了股票代码，转换为ts_code格式
    if symbol:
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol]
    else:
        ts_code = None
    
    # 获取大宗交易数据
    df = pro.block_trade(ts_code=ts_code, limit=100)
    
    if df.empty:
        return []
    
    # 获取最近两个交易日
    latest_dates = sorted(df['trade_date'].unique(), reverse=True)[:2]
    
    # 检查这两个日期是否在最近5个交易日内
    if not all(date in recent_trade_dates for date in latest_dates):
        return []
    
    # 筛选最近两个交易日的数据
    df = df[df['trade_date'].isin(latest_dates)]
    
    # 重命名列
    df = df.rename(columns=field_mapping)
    
    # 处理股票代码，去掉后缀
    if '股票代码' in df.columns:
        df['股票代码'] = df['股票代码'].str.replace(r'\.\w+', '', regex=True)
    
    # 格式化数值（保留两位小数）
    for col in ['成交量', '成交金额', '成交价']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # 转换为字典列表
    result = df.to_dict('records')
    
    return result

def fetch_block_trades10(symbol: str = None) -> list:
    """
    获取最近两个交易日的大宗交易数据
    只返回最近10天内的数据，如果不在范围内则返回空列表

    参数:
    symbol (str, optional): 股票代码，默认为None

    返回:
    list: 包含大宗交易数据的列表，每个元素为字典，包含中文键值
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 获取10天前的日期
    ten_days_ago = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    
    # 字段映射表
    field_mapping = {
        'ts_code': '股票代码',
        'trade_date': '交易日期',
        'price': '成交价',
        'vol': '成交量',
        'amount': '成交金额',
        'buyer': '买方营业部',
        'seller': '卖方营业部'
    }
    
    # 如果提供了股票代码，转换为ts_code格式
    if symbol:
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol]
    else:
        ts_code = None
    
    # 获取大宗交易数据
    pro = ts.pro_api()
    df = pro.block_trade(ts_code=ts_code, limit=100)
    
    if df.empty:
        return []
    
    # 获取最近两个交易日并检查是否在10天内
    latest_dates = sorted(df['trade_date'].unique(), reverse=True)[:2]
    if not latest_dates or min(latest_dates) < ten_days_ago:
        return []
    
    # 筛选最近两个交易日的数据
    df = df[df['trade_date'].isin(latest_dates)]
    
    # 重命名列
    df = df.rename(columns=field_mapping)
    
    # 处理股票代码，去掉后缀
    if '股票代码' in df.columns:
        df['股票代码'] = df['股票代码'].str.replace(r'\.\w+', '', regex=True)
    
    # 格式化数值（保留两位小数）
    for col in ['成交量', '成交金额', '成交价']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # 转换为字典列表
    result = df.to_dict('records')
    
    return result


def fetch_broker_reports(symbol=None, report_date=None, start_date=None, end_date=None, offset=0, limit=100) -> list:
    """
    获取券商（卖方）研报的盈利预测数据
    
    Args:
        symbol (str, optional): 股票代码
        report_date (str, optional): 报告日期 YYYYMMDD
        start_date (str, optional): 开始日期 YYYYMMDD
        end_date (str, optional): 结束日期 YYYYMMDD
        offset (int): 数据偏移量，默认0
        limit (int): 单次获取数据量，默认100
    
    Returns:
        list: 符合条件的股票代码列表
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    from datetime import datetime, timedelta
    import pandas as pd
    
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    # 定义列名映射
    column_mapping = {
        'ts_code': '股票代码',
        'name': '股票名称',
        'report_date': '报告日期',
        'report_title': '报告标题',
        'report_type': '报告类型',
        'classify': '报告分类',
        'org_name': '机构名称',
        'author_name': '作者',
        'quarter': '预测季度',
        'op_rt': '预测营收(万元)',
        'op_pr': '预测营业利润(万元)',
        'tp': '预测利润总额(万元)',
        'np': '预测净利润(万元)',
        'eps': '预测每股收益(元)',
        'pe': '预测市盈率',
        'rd': '预测股息率',
        'roe': '预测净资产收益率',
        'ev_ebitda': '预测EV/EBITDA',
        'rating': '卖方评级',
        'max_price': '预测最高目标价',
        'min_price': '预测最低目标价',
        'imp_dg': '机构关注度'
    }
    
    def get_date_range(months):
        """获取指定月数范围的起止日期"""
        end = datetime.now()
        start = end - timedelta(days=30*months)
        return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    
    def fetch_and_process_data(start_date, end_date):
        """获取并处理指定日期范围的数据"""
        pro = ts.pro_api()
        try:
            df = pro.report_rc(
                ts_code=ts_code,
                report_date=report_date,
                start_date=start_date,
                end_date=end_date,
                offset=offset,
                limit=limit
            )
            
            if df.empty:
                return pd.DataFrame()
                
            # 重命名列
            df.rename(columns=column_mapping, inplace=True)
            
            # 处理股票代码，去除后缀
            if 'ts_code' in df.columns:
                df['股票代码'] = df['ts_code'].str.split('.').str[0]
                df.drop(columns=['ts_code'], inplace=True)
            elif '股票代码' in df.columns:
                df['股票代码'] = df['股票代码'].str.split('.').str[0]
                
            return df
            
        except Exception as e:
            print(f"获取数据出错: {e}")
            return pd.DataFrame()
    
    # 如果指定了日期范围，直接使用指定的日期
    if start_date and end_date:
        df = fetch_and_process_data(start_date, end_date)
        if not df.empty:
            return df['股票代码'].unique().tolist()
    
    # 否则逐步扩大时间范围查询
    for months in [1, 2, 3]:
        start, end = get_date_range(months)
        df = fetch_and_process_data(start, end)
        
        if not df.empty:
            return df['股票代码'].unique().tolist()
    
    return []

@tsdata
def get_broker_monthly_picks(month: str = None, offset: int = 0, limit: int = 1000) -> list:
    """
    获取券商月度金股数据
    
    Args:
        month (str, optional): 月度（YYYYMM），默认为None，自动获取当月或上月数据
        offset (int): 数据偏移量，默认值为0
        limit (int): 单次获取数据的最大行数，默认值为1000
    
    Returns:
        list: 股票代码列表
    """
    pro = ts.pro_api()
    
    # 处理月份参数
    if not month:
        current_date = datetime.now()
        current_month = current_date.strftime('%Y%m')
        
        # 尝试获取当月数据
        df = pro.broker_recommend(month=current_month, offset=offset, limit=limit)
        
        # 如果当月没有数据，获取上月数据
        if df.empty:
            # 计算上个月的日期
            if current_date.month == 1:
                last_month = datetime(current_date.year - 1, 12, 1)
            else:
                last_month = current_date.replace(day=1) - datetime.timedelta(days=1)
            
            month = last_month.strftime('%Y%m')
            df = pro.broker_recommend(month=month, offset=offset, limit=limit)
    else:
        df = pro.broker_recommend(month=month, offset=offset, limit=limit)
    
    # 定义列名映射
    column_mapping = {
        'ts_code': '股票代码',
        'name': '股票名称',
        'month': '月份',
        'broker': '券商名称',
        'analyst': '分析师',
        'researcher': '研究员',
        'report_date': '报告日期',
        'title': '标题',
        'reason': '推荐理由',
        'rating': '评级',
        'target_price': '目标价格',
        'pub_date': '发布日期'
    }
    
    # 重命名列
    df.rename(columns=column_mapping, inplace=True)
    
    # 处理股票代码
    if 'ts_code' in df.columns:
        df['股票代码'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    elif '股票代码' in df.columns:
        df['股票代码'] = df['股票代码'].str.split('.').str[0]
    
    return df['股票代码'].tolist()


def get_latest_stock_chip_performance(symbol: str = None) -> dict:
    """
    获取A股最新一条筹码平均成本和胜率数据，并返回中文字段的字典格式。

    参数:
    symbol (str): 股票代码，例如："000001"

    返回:
    dict: 包含最新一条筹码数据的字典，字段为中文。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    import tushare as ts
    
    # 字段映射关系
    field_mappings = {
        'ts_code': '股票代码',
        'trade_date': '交易日期',
        'his_low': '历史最低价',
        'his_high': '历史最高价',
        'cost_5pct': '5分位成本',
        'cost_15pct': '15分位成本',
        'cost_50pct': '50分位成本',
        'cost_85pct': '85分位成本',
        'cost_95pct': '95分位成本',
        'weight_avg': '加权平均成本',
        'winner_rate': '胜率'
    }
    
    try:
        # 获取股票代码
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol] if symbol else None
        
        if not ts_code:
            return {'error': '无效的股票代码'}
        
        # 调用API获取数据
        pro = ts.pro_api()
        df = pro.cyq_perf(
            ts_code=ts_code,
            limit=1  # 只获取最新一条数据
        )
        
        if df.empty:
            return {'error': '未找到数据'}
            
        # 处理数据
        # 1. 将第一行数据转为字典
        latest_data = df.iloc[0].to_dict()
        
        # 2. 处理股票代码（去除后缀）
        if 'ts_code' in latest_data:
            latest_data['ts_code'] = latest_data['ts_code'].split('.')[0]
            
        # 3. 重命名字段为中文
        result = {}
        for eng_key, value in latest_data.items():
            if eng_key in field_mappings:
                result[field_mappings[eng_key]] = value
                
        return result
        
    except Exception as e:
        return {'error': f'获取数据失败: {str(e)}'}


def get_latest_stock_chip_distribution(symbol: str = None) -> list:
    """
    获取A股最新一个交易日的全部筹码分布数据，并返回中文字段的列表格式。
    通过一次性获取更多数据，然后筛选最新日期的方式来提高效率。

    参数:
    symbol (str): 股票代码，例如："000001"

    返回:
    list: 包含最新一个交易日全部筹码分布数据的列表，每个元素为字典，字段为中文。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    import tushare as ts
    import pandas as pd
    
    # 字段映射关系
    field_mappings = {
        'ts_code': '股票代码',
        'trade_date': '交易日期',
        'price': '成本价格',
        'percent': '价格占比'
    }
    
    try:
        # 获取股票代码
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol] if symbol else None
        
        if not ts_code:
            return [{'error': '无效的股票代码'}]
        
        # 调用API一次性获取较多数据
        pro = ts.pro_api()
        df = pro.cyq_chips(
            ts_code=ts_code,
            limit=2000  # 获取最大数量以确保包含完整的最新日期数据
        )
        
        if df.empty:
            return [{'error': '未找到数据'}]
        
        # 获取最新日期
        latest_date = df['trade_date'].max()
        
        # 筛选最新日期的数据
        latest_df = df[df['trade_date'] == latest_date].copy()
        
        if latest_df.empty:
            return [{'error': '未找到该日期的分布数据'}]
        
        # 处理数据
        # 1. 去除股票代码后缀
        if 'ts_code' in latest_df.columns:
            latest_df['ts_code'] = latest_df['ts_code'].str.split('.').str[0]
        
        # 2. 重命名列为中文
        latest_df.rename(columns=field_mappings, inplace=True)
        
        # 3. 按成本价格排序
        latest_df = latest_df.sort_values(by='成本价格', ascending=True)
        
        # 4. 格式化数字
        latest_df['成本价格'] = latest_df['成本价格'].round(2)
        latest_df['价格占比'] = latest_df['价格占比'].round(4)
        
        # 5. 数据验证
        total_percent = latest_df['价格占比'].sum()
        if not (0.99 <= total_percent <= 1.01):  # 允许1%的误差
            print(f"警告: 价格占比总和 ({total_percent}) 不接近 100%")
            
        # 6. 添加数据统计
        data_count = len(latest_df)
        if data_count < 10:  # 假设正常数据应该至少有10个价格点
            print(f"警告: 数据点数量 ({data_count}) 可能不完整")
        
        # 7. 转换为列表格式
        result = latest_df.to_dict('records')
        
        return result
        
    except Exception as e:
        return [{'error': f'获取数据失败: {str(e)}'}]

def fetch_stock_technical_factors(symbol=None, trade_date=None, start_date=None, end_date=None, fields=None, offset=0, limit=10000):
    """
    获取股票每日技术面因子数据，用于跟踪股票当前走势情况。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日期（格式：yyyymmdd）。
    start_date (str, optional): 开始日期（格式：yyyymmdd）。
    end_date (str, optional): 结束日期（格式：yyyymmdd）。
    fields (str, optional): 需要返回的字段，多个字段用逗号分隔。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次返回数据条数，默认值为10000（单次最大值）。

    返回:
    pandas.DataFrame: 包含技术面因子数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.stk_factor(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, fields=fields, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_ccass_holdings(symbol=None, hk_code=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取中央结算系统持股汇总数据，覆盖全部历史数据。

    参数:
    symbol (str): 股票代码 (e.g. 605009.SH)，支持A股、港股、美股、指数、期货。
    hk_code (str): 港交所代码 （e.g. 95009）。
    trade_date (str): 交易日期 (YYYYMMDD格式)。
    start_date (str): 开始日期 (YYYYMMDD格式)。
    end_date (str): 结束日期 (YYYYMMDD格式)。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据条数，默认5000，最大5000。

    返回:
    DataFrame: 包含中央结算系统持股汇总数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.ccass_hold(ts_code=ts_code, hk_code=hk_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_ccass_holdings(symbol=None, hk_code=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=6000):
    """
    获取中央结算系统机构席位持股明细，数据覆盖全历史。
    
    参数:
    symbol (str): 股票代码 (e.g. 605009.SH)
    hk_code (str): 港交所代码 （e.g. 95009）
    trade_date (str): 交易日期(YYYYMMDD格式)
    start_date (str): 开始日期
    end_date (str): 结束日期
    offset (int): 数据偏移量，默认0
    limit (int): 单次返回数据条数，默认6000
    
    返回:
    DataFrame: 包含交易日期、股票代号、股票名称、参与者编号、机构名称、持股量、占已发行股份百分比等信息
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.ccass_hold_detail(ts_code=ts_code, hk_code=hk_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df


def fetch_stock_technical_factors(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    trade_date: str = None,
    offset: int = 0,
    limit: int = 10000
) -> pd.DataFrame:
    """
    获取股票每日技术面因子数据，用于跟踪股票当前走势情况。数据由Tushare社区自产，覆盖全历史。
    
    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    start_date (str, optional): 开始日期，格式为yyyymmdd。
    end_date (str, optional): 结束日期，格式为yyyymmdd。
    trade_date (str, optional): 交易日期，格式为yyyymmdd。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大数量，默认值为10000。
    
    返回:
    pd.DataFrame: 包含股票技术面因子数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher
    
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]
    
    params = {
        'ts_code': ts_code,
        'start_date': start_date,
        'end_date': end_date,
        'trade_date': trade_date,
        'offset': offset,
        'limit': limit
    }
    
    df = pro.stk_factor(**params)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_hk_hold_details(
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    exchange: str = None,
    symbol: str = None,
    limit: int = 3800,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取沪深港股通持股明细，数据来源港交所。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    exchange (str): 交易所类型，可选值为SH（沪股通）、SZ（深股通）、HK（港股通）。
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    limit (int): 单次提取记录数，默认3800条。
    offset (int): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含持股明细的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.hk_hold(
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        ts_code=ts_code,
        limit=limit,
        offset=offset
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def get_institutional_research_records(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=100):
    """
    获取上市公司机构调研记录数据

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货
    trade_date (str): 调研日期，格式为YYYYMMDD
    start_date (str): 调研开始日期，格式为YYYYMMDD
    end_date (str): 调研结束日期，格式为YYYYMMDD
    offset (int): 数据偏移量，默认值为0
    limit (int): 单次获取数据的最大条数，默认值为100

    返回:
    DataFrame: 包含机构调研记录的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.stk_surv(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit, fields='ts_code,name,surv_date,fund_visitors,rece_place,rece_mode,rece_org,org_type,comp_rece,content')
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_margin_data(trade_date: str = None, exchange_id: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取融资融券每日交易汇总数据。

    参数:
    trade_date (str, 可选): 交易日期，格式为YYYYMMDD。
    exchange_id (str, 可选): 交易所代码，可选值为'SSE'（上交所）、'SZSE'（深交所）、'BSE'（北交所）。
    start_date (str, 可选): 开始日期，格式为YYYYMMDD。
    end_date (str, 可选): 结束日期，格式为YYYYMMDD。
    offset (int, 可选): 数据偏移量，默认值为0。
    limit (int, 可选): 单次返回数据条数，默认值为1000。

    返回:
    pd.DataFrame: 包含融资融券每日交易汇总数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.margin(trade_date=trade_date, exchange_id=exchange_id, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_stock_holder_trades(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    trade_type: str = None,
    holder_type: str = None,
    offset: int = 0,
    limit: int = 3000
) -> pd.DataFrame:
    """
    获取上市公司增减持数据，了解重要股东近期及历史上的股份增减变化。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    ann_date (str, optional): 公告日期，格式为YYYYMMDD。默认为None。
    start_date (str, optional): 公告开始日期，格式为YYYYMMDD。默认为None。
    end_date (str, optional): 公告结束日期，格式为YYYYMMDD。默认为None。
    trade_type (str, optional): 交易类型，IN表示增持，DE表示减持。默认为None。
    holder_type (str, optional): 股东类型，C表示公司，P表示个人，G表示高管。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据的最大行数，默认为3000。

    返回:
    pd.DataFrame: 包含增减持数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    params = {
        'ts_code': ts_code,
        'ann_date': ann_date,
        'start_date': start_date,
        'end_date': end_date,
        'trade_type': trade_type,
        'holder_type': holder_type,
        'offset': offset,
        'limit': limit
    }
    pro = ts.pro_api()
    df = pro.stk_holdertrade(**params)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_margin_financing_summary(trade_date=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取转融通融资汇总数据

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD
    start_date (str, optional): 开始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    offset (int, optional): 数据偏移量，默认值为0
    limit (int, optional): 单次获取数据的最大行数，默认值为5000

    返回:
    DataFrame: 包含转融通融资汇总数据的DataFrame
    """
    pro = ts.pro_api()
    
    df = pro.slb_len(trade_date=trade_date, start_date=start_date, end_date=end_date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_slb_sec_data(trade_date=None, symbol=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取转融通转融券交易汇总数据

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    offset (int): 数据偏移量，默认0
    limit (int): 单次提取数据量，默认5000

    返回:
    DataFrame: 包含转融通转融券交易汇总数据的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.slb_sec(trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_margin_details(trade_date: str = None, symbol: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取沪深两市每日融资融券明细数据。

    参数:
    trade_date (str, 可选): 交易日期，格式为YYYYMMDD。
    symbol (str, 可选): 股票代码或简称，支持A股、港股、美股、指数、期货。
    start_date (str, 可选): 开始日期，格式为YYYYMMDD。
    end_date (str, 可选): 结束日期，格式为YYYYMMDD。
    offset (int, 可选): 数据偏移量，默认值为0。
    limit (int, 可选): 单次获取数据的最大数量，默认值为1000。

    返回:
    pd.DataFrame: 包含融资融券明细数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    # 将symbol转换为ts_code
    ts_code = matcher[symbol] if symbol else None
    
    # 调用API获取数据
    pro = ts.pro_api()
    df = pro.query('margin_detail', trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_margin_lending_details(trade_date=None, symbol=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取转融券交易明细数据

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    offset (int): 数据偏移量，默认为0
    limit (int): 单次获取数据的最大行数，默认为5000

    返回:
    DataFrame: 包含转融券交易明细的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.slb_sec_detail(trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def fetch_hsgt_moneyflow(trade_date=None, start_date=None, end_date=None, offset=0, limit=300):
    """
    获取沪股通、深股通、港股通每日资金流向数据。

    参数:
    trade_date (str, optional): 交易日期 (二选一)
    start_date (str, optional): 开始日期 (二选一)
    end_date (str, optional): 结束日期
    offset (int, optional): 数据偏移量，默认值为0
    limit (int, optional): 每次获取的数据条数，默认值为300

    返回:
    pd.DataFrame: 包含资金流向数据的DataFrame
    """
    pro = ts.pro_api()
    
    if trade_date:
        df = pro.query('moneyflow_hsgt', trade_date=trade_date, offset=offset, limit=limit)
    else:
        df = pro.query('moneyflow_hsgt', start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    return df

def get_recent_stock_moneyflow(symbol: str = None, limit: int = 5) -> list:
    """
    获取股票最近的资金流向数据，返回最近5条记录，便于分析大单小单成交情况。

    参数:
    symbol (str): 股票代码，例如："000001"
    limit (int): 返回的记录数量，默认5条

    返回:
    list: 包含最近5条资金流向数据的列表，每个元素为字典，字段为中文。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    import tushare as ts
    
    # 字段映射关系
    field_mappings = {
        'ts_code': '股票代码',
        'trade_date': '交易日期',
        'buy_sm_vol': '小单买入量',
        'buy_sm_amount': '小单买入金额（万元）',
        'sell_sm_vol': '小单卖出量',
        'sell_sm_amount': '小单卖出金额（万元）',
        'buy_md_vol': '中单买入量',
        'buy_md_amount': '中单买入金额（万元）',
        'sell_md_vol': '中单卖出量',
        'sell_md_amount': '中单卖出金额（万元）',
        'buy_lg_vol': '大单买入量',
        'buy_lg_amount': '大单买入金额（万元）',
        'sell_lg_vol': '大单卖出量',
        'sell_lg_amount': '大单卖出金额（万元）',
        'buy_elg_vol': '特大单买入量',
        'buy_elg_amount': '特大单买入金额（万元）',
        'sell_elg_vol': '特大单卖出量',
        'sell_elg_amount': '特大单卖出金额（万元）',
        'net_mf_vol': '净流入量',
        'net_mf_amount': '净流入额（万元）'
    }
    
    try:
        # 获取股票代码
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol] if symbol else None
        
        if not ts_code:
            return [{'错误': '无效的股票代码'}]
        
        # 调用API获取数据
        pro = ts.pro_api()
        df = pro.moneyflow(
            ts_code=ts_code,
            limit=limit
        )
        
        if df.empty:
            return [{'错误': '未找到数据'}]
        
        # 数据处理
        df = df.sort_values('trade_date', ascending=False)  # 按日期降序排序
        
        # 添加资金流向分析
        def analyze_flow(row):
            # 计算大单和特大单的净流入
            big_money_net = (row['buy_lg_amount'] + row['buy_elg_amount'] - 
                           row['sell_lg_amount'] - row['sell_elg_amount'])
            
            # 判断主力资金动向
            if big_money_net > 0:
                return '主力净流入' if big_money_net > row['net_mf_amount'] * 0.5 else '小幅净流入'
            else:
                return '主力净流出' if big_money_net < row['net_mf_amount'] * 0.5 else '小幅净流出'
        
        df['资金动向'] = df.apply(analyze_flow, axis=1)
        
        # 转换为列表格式
        result = []
        for _, row in df.iterrows():
            item = {}
            # 基础数据转换
            for eng_key, cn_key in field_mappings.items():
                if eng_key in row:
                    value = row[eng_key]
                    # 处理股票代码
                    if eng_key == 'ts_code':
                        value = value.split('.')[0]
                    # 处理金额，保留2位小数
                    elif 'amount' in eng_key:
                        value = round(float(value), 2)
                    item[cn_key] = value
            
            # 添加分析结果
            item['资金动向'] = row['资金动向']
            
            # 添加主力净额
            item['主力净额'] = round(float(row['buy_lg_amount'] + row['buy_elg_amount'] - 
                                    row['sell_lg_amount'] - row['sell_elg_amount']), 2)
            
            result.append(item)
            
        return result
        
    except Exception as e:
        return [{'错误': f'获取数据失败: {str(e)}'}]

@tsdata
def fetch_market_lending_summary(trade_date: str = None, symbol: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 5000) -> pd.DataFrame:
    """
    获取做市借券交易汇总数据。

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据的最大行数，默认为5000。

    返回:
    pd.DataFrame: 包含做市借券交易汇总数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    # 将symbol转换为ts_code
    ts_code = matcher[symbol] if symbol else None
    
    # 调用API获取数据
    pro = ts.pro_api()
    df = pro.slb_len_mm(trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_ths_moneyflow(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 6000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取同花顺个股资金流向数据，每日盘后更新。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    limit (int, optional): 单次获取数据的最大数量，默认6000。
    offset (int, optional): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含资金流向数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.moneyflow_ths(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

def get_limit_up_list_ths(trade_date:str=None,symbol:str=None,limit_type:str=None,exchange:str=None,start_date:str=None,end_date:str=None,limit:int=1000,offset:int=0):
    """
    获取涨停板数据

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货
    limit_type (str, optional): 涨停类型，包括“涨停”、“一字涨停”、“连板”、“一字连板”、“ST涨停”、“ST连板”
    exchange (str, optional): 交易所代码，可选值为SSE（上交所）、SZSE（深交所）、BSE（北交所）
    start_date (str, optional): 开始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    limit (int, optional): 单次获取数据的最大行数，默认值为2000
    offset (int, optional): 数据偏移量，默认值为0

    返回:
    pd.DataFrame: 包含涨停板数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.limit_list_ths(trade_date=trade_date, ts_code=ts_code, limit_type=limit_type, exchange=exchange, start_date=start_date, end_date=end_date, limit=limit, offset=offset)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def get_limit_up_list(trade_date:str=None,symbol:str=None,limit_type:str=None,exchange:str=None,start_date:str=None,end_date:str=None,limit:int=1000,offset:int=0):
    """
    获取涨停板数据

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货
    limit_type (str, optional): 涨停类型，包括“涨停”、“一字涨停”、“连板”、“一字连板”、“ST涨停”、“ST连板”
    exchange (str, optional): 交易所代码，可选值为SSE（上交所）、SZSE（深交所）、BSE（北交所）
    start_date (str, optional): 开始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    limit (int, optional): 单次获取数据的最大行数，默认值为2000
    offset (int, optional): 数据偏移量，默认值为0

    返回:
    pd.DataFrame: 包含涨停板数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.limit_list_d(trade_date=trade_date, ts_code=ts_code, limit_type=limit_type, exchange=exchange, start_date=start_date, end_date=end_date, limit=limit, offset=offset)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

def get_limit_up_ladder(trade_date:str=None,symbol:str=None,start_date:str=None,end_date:str=None,limit:int=2000,offset:int=0):
    """
    获取涨停板数据

    参数:
    trade_date (str, optional): 交易日期，格式为YYYYMMDD
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货
    start_date (str, optional): 开始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    limit (int, optional): 单次获取数据的最大行数，默认值为2000
    offset (int, optional): 数据偏移量，默认值为0

    返回:
    pd.DataFrame: 包含涨停板数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.limit_step(trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, limit=limit, offset=offset)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def get_hot_money_list(name: str = None, offset: int = 0, limit: int = 500) -> pd.DataFrame:
    """
    获取游资分类名录信息

    参数:
    name (str, optional): 游资名称
    offset (int, optional): 数据偏移量，默认值为0
    limit (int, optional): 单次获取数据的最大数量，默认值为500

    返回:
    pd.DataFrame: 包含游资分类名录信息的DataFrame
    """
    pro = ts.pro_api()
    df = pro.hm_list(name=name)
    
    # 如果输出结果包含ts_code列，去掉000000.XX的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_margin_securities(
    symbol: str = None,
    trade_date: str = None,
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取沪深京三大交易所融资融券标的（包括ETF），每天盘前更新。

    参数:
    symbol (str, optional): 标的代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日，格式为YYYYMMDD。
    exchange (str, optional): 交易所代码，SSE（上交所）、SZSE（深交所）、BSE（北交所）。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大行数，默认值为6000。

    返回:
    pd.DataFrame: 包含融资融券标的的DataFrame，包含以下列：
        - trade_date (str): 交易日期
        - ts_code (str): 标的代码
        - name (str): 标的名称
        - exchange (str): 交易所
        - code (str): 去除后缀的标的代码
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.margin_secs(
        ts_code=ts_code,
        trade_date=trade_date,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def fetch_ths_industry_moneyflow(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 5000
) -> list:
    """
    获取同花顺行业板块资金流向数据。

    参数:
    symbol (str, optional): 股票代码或板块代码，默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD，默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD，默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为5000。

    返回:
    list: 包含资金流向数据的字典列表。
    """
    # 字段映射关系
    field_mapping = {
        'trade_date': '交易日期',
        'ts_code': '板块代码',
        'industry': '板块名称',
        'lead_stock': '领涨股票名称',
        'close': '收盘指数',
        'pct_change': '指数涨跌幅',
        'company_num': '公司数量',
        'pct_change_stock': '领涨股涨跌幅',
        'close_price': '领涨股最新价',
        'net_buy_amount': '流入资金',
        'net_sell_amount': '流出资金',
        'net_amount': '净额'
    }

    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    pro = ts.pro_api()

    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    # 获取原始数据
    df = pro.moneyflow_ind_ths(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    # 处理空数据情况
    if df is None or df.empty:
        return []

    # 重命名列
    df = df.rename(columns=field_mapping)

    # 处理股票代码，移除后缀
    if '板块代码' in df.columns:
        df['板块代码'] = df['板块代码'].str.replace(r'\.\w+$', '', regex=True)

    # 转换为字典列表
    result = df.to_dict('records')

    # 确保数值类型的准确性
    for item in result:
        # 转换数值型字段
        numeric_fields = ['收盘指数', '指数涨跌幅', '公司数量', '领涨股涨跌幅', 
                         '领涨股最新价', '流入资金', '流出资金', '净额']
        for field in numeric_fields:
            if field in item and item[field] is not None:
                try:
                    # 保留小数位数
                    if field in ['指数涨跌幅', '领涨股涨跌幅']:
                        item[field] = round(float(item[field]), 2)
                    elif field in ['流入资金', '流出资金', '净额']:
                        item[field] = round(float(item[field]), 4)
                    elif field == '公司数量':
                        item[field] = int(item[field])
                    else:
                        item[field] = float(item[field])
                except (ValueError, TypeError):
                    item[field] = None

    return result

@tsdata
def fetch_stock_moneyflow(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取东方财富个股资金流向数据，每日盘后更新，数据开始于20230911。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大条数，默认6000条。

    返回:
    pd.DataFrame: 包含资金流向数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.moneyflow_dc(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

def get_concept_constituents(trade_date=None, symbol=None, offset=0, limit=200):
    """
    获取开盘啦概念题材的成分股数据。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str): 题材代码，支持A股、港股、美股、指数、期货等。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 每次获取的数据量，默认3000条。

    返回:
    DataFrame: 包含题材成分股数据的DataFrame。
    """
    # 将symbol转换为ts_code
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用接口获取数据
    pro = ts.pro_api()
    df = pro.kpl_concept_cons(trade_date=trade_date, ts_code=ts_code, offset=offset, limit=limit)

    # 去掉ts_code列中的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_eastmoney_moneyflow(trade_date: str = None, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 3000) -> pd.DataFrame:
    """
    获取东方财富大盘资金流向数据，每日盘后更新。

    参数:
    trade_date (str, 可选): 交易日期，格式为YYYYMMDD。
    start_date (str, 可选): 开始日期，格式为YYYYMMDD。
    end_date (str, 可选): 结束日期，格式为YYYYMMDD。
    offset (int, 可选): 数据偏移量，默认值为0。
    limit (int, 可选): 单次获取数据的最大条数，默认值为3000。

    返回:
    pd.DataFrame: 包含资金流向数据的DataFrame。
    """
    pro = ts.pro_api()
    # 调用东方财富API获取资金流向数据
    df = pro.moneyflow_mkt_dc(trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_kpl_concept_data(trade_date=None, symbol=None, name=None, offset=0, limit=5000):
    """
    获取开盘啦概念题材列表，每天盘后更新。

    参数:
    trade_date (str): 交易日期（YYYYMMDD格式），可选
    symbol (str): 题材代码（xxxxxx格式），可选
    name (str): 题材名称，可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大数量，默认5000

    返回:
    DataFrame: 包含交易日期、题材代码、题材名称、涨停数量、排名上升位数的数据框
    """
    pro = ts.pro_api()
    
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    # 调用接口获取数据
    df = pro.kpl_concept(trade_date=trade_date, ts_code=ts_code, name=name, offset=offset, limit=limit)
    
    # 去掉ts_code列中的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df


def get_sector_moneyflow(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 5000
) -> list:
    """
    获取东方财富板块资金流向数据。

    参数:
    symbol (str, optional): 股票代码或板块代码，默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD，默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD，默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大条数，默认为5000。

    返回:
    list: 包含板块资金流向数据的字典列表。
    """
    # 字段映射关系
    field_mapping = {
        'trade_date': '交易日期',
        'ts_code': '板块代码',
        'name': '板块名称',
        'pct_change': '板块涨跌幅',
        'close': '板块最新指数',
        'net_amount': '主力净流入',
        'net_amount_rate': '主力净流入占比',
        'buy_elg_amount': '超大单净流入',
        'buy_elg_amount_rate': '超大单净流入占比',
        'buy_lg_amount': '大单净流入',
        'buy_lg_amount_rate': '大单净流入占比',
        'buy_md_amount': '中单净流入',
        'buy_md_amount_rate': '中单净流入占比',
        'buy_sm_amount': '小单净流入',
        'buy_sm_amount_rate': '小单净流入占比',
        'buy_sm_amount_stock': '主力净流入最大股',
        'rank': '排名'
    }

    from core.stock.ts_code_matcher import TsCodeMatcher
    pro = ts.pro_api()

    # 如果提供了symbol，将其转换为ts_code
    if symbol:
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol]
    else:
        ts_code = None

    # 调用pro接口获取数据
    df = pro.moneyflow_ind_dc(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    # 处理空数据情况
    if df is None or df.empty:
        return []

    # 重命名列
    df = df.rename(columns=field_mapping)

    # 处理板块代码，移除后缀
    if '板块代码' in df.columns:
        df['板块代码'] = df['板块代码'].str.replace(r'\.\w+$', '', regex=True)

    # 转换为字典列表
    result = df.to_dict('records')

    # 处理数值类型
    for item in result:
        # 处理涨跌幅和占比字段（保留2位小数）
        percentage_fields = ['板块涨跌幅', '主力净流入占比', '超大单净流入占比', 
                           '大单净流入占比', '中单净流入占比', '小单净流入占比']
        for field in percentage_fields:
            if field in item and item[field] is not None:
                try:
                    item[field] = round(float(item[field]), 2)
                except (ValueError, TypeError):
                    item[field] = None

        # 处理金额字段（保留2位小数）
        amount_fields = ['主力净流入', '超大单净流入', '大单净流入', 
                        '中单净流入', '小单净流入']
        for field in amount_fields:
            if field in item and item[field] is not None:
                try:
                    item[field] = round(float(item[field]), 2)
                except (ValueError, TypeError):
                    item[field] = None

        # 处理指数值（保留2位小数）
        if '板块最新指数' in item and item['板块最新指数'] is not None:
            try:
                item['板块最新指数'] = round(float(item['板块最新指数']), 2)
            except (ValueError, TypeError):
                item['板块最新指数'] = None

        # 处理排名（转为整数）
        if '排名' in item and item['排名'] is not None:
            try:
                item['排名'] = int(item['排名'])
            except (ValueError, TypeError):
                item['排名'] = None

    return result

@tsdata
def get_index_info(
    symbol: str = None,
    name: str = None,
    market: str = 'SSE',
    publisher: str = None,
    category: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取指数基础信息

    :param symbol: 指数代码 (可选)
    :param name: 指数简称 (可选)
    :param market: 交易所或服务商 (默认SSE)
    :param publisher: 发布商 (可选)
    :param category: 指数类别 (可选)
    :param offset: 数据偏移量 (默认0)
    :param limit: 单次获取数据量 (默认1000)
    :return: DataFrame 包含指数基础信息
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()

    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.index_basic(
        ts_code=ts_code,
        name=name,
        market=market,
        publisher=publisher,
        category=category
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def fetch_daily_hot_money_details(
    trade_date: str = None,
    symbol: str = None,
    hm_name: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取每日游资交易明细数据。

    参数:
    trade_date (str): 交易日期(YYYYMMDD)，可选
    symbol (str): 股票代码，可选。支持A股、港股、美股、指数、期货。
    hm_name (str): 游资名称，可选
    start_date (str): 开始日期(YYYYMMDD)，可选
    end_date (str): 结束日期(YYYYMMDD)，可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次提取数据量，默认2000，最大2000

    返回:
    pd.DataFrame: 包含游资交易明细的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.hm_detail(
        trade_date=trade_date,
        ts_code=ts_code,
        hm_name=hm_name,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def fetch_kpl_list(
    symbol: str = None,
    trade_date: str = None,
    tag: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 8000,
    fields: str = 'ts_code,name,trade_date,tag,theme,status'
) -> pd.DataFrame:
    """
    获取开盘啦涨停、跌停、炸板等榜单数据。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。
    tag (str, optional): 板单类型（涨停/炸板/跌停/自然涨停/竞价)。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大条数，默认值为8000。
    fields (str, optional): 返回的字段，默认值为'ts_code,name,trade_date,tag,theme,status'。

    返回:
    pd.DataFrame: 包含指定字段的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.kpl_list(
        ts_code=ts_code,
        trade_date=trade_date,
        tag=tag,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


def fetch_ths_hot_data(
    trade_date: str = None,
    symbol: str = None,
    market: str = None,
    is_new: str = 'Y',
    offset: int = 0,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    获取同花顺App热榜数据，包括热股、概念板块、ETF、可转债、港美股等。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    market (str): 热榜类型，可选值包括：热股、ETF、可转债、行业板块、概念板块、期货、港股、热基、美股。
    is_new (str): 是否最新，默认Y，如果为N则为盘中和盘后阶段采集。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据的最大条数，默认2000。

    返回:
    pd.DataFrame: 包含热榜数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    df = pro.ths_hot(
        trade_date=trade_date,
        ts_code=ts_code,
        market=market,
        is_new=is_new,
        fields='trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,concept,rank_reason,hot,rank_time',
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_south_china_index_daily(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取南华指数每日行情数据。

    参数:
    symbol (str, optional): 指数代码或股票代码，默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD，默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD，默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为1000。

    返回:
    pandas.DataFrame: 包含南华指数每日行情数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.index_daily(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_a_share_limit_data(
    trade_date: str = None,
    symbol: str = None,
    limit_type: str = None,
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取沪深A股每日涨跌停、炸板数据情况。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    limit_type (str): 涨跌停类型（U涨停D跌停Z炸板）。
    exchange (str): 交易所（SH上交所SZ深交所BJ北交所）。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大条数，默认1000条。

    返回:
    pd.DataFrame: 包含涨跌停、炸板数据的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.limit_list_d(
        trade_date=trade_date,
        ts_code=ts_code,
        limit_type=limit_type,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        fields='trade_date,ts_code,industry,name,close,pct_chg,amount,limit_amount,float_mv,total_mv,turnover_ratio,fd_amount,first_time,last_time,open_times,up_stat,limit_times,limit'
    )

    # 去掉ts_code的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_sz_market_daily_summary(trade_date=None, symbol=None, start_date=None, end_date=None, offset=0, limit=2000):
    """
    获取深圳市场每日交易概况数据。

    参数:
    trade_date (str): 交易日期（YYYYMMDD格式）
    symbol (str): 板块代码或股票代码
    start_date (str): 开始日期（YYYYMMDD格式）
    end_date (str): 结束日期（YYYYMMDD格式）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认2000

    返回:
    pd.DataFrame: 包含深圳市场每日交易概况的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    df = pro.sz_daily_info(trade_date=trade_date, ts_code=ts_code, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_eastmoney_hot_data(
    trade_date: str = None,
    symbol: str = None,
    market: str = None,
    hot_type: str = None,
    is_new: str = 'Y',
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取东方财富App热榜数据，包括A股市场、ETF基金、港股市场、美股市场等。

    参数:
    trade_date (str): 交易日期，格式为YYYYMMDD。
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货等。
    market (str): 市场类型，可选值为'A股市场'、'ETF基金'、'港股市场'、'美股市场'。
    hot_type (str): 热点类型，可选值为'人气榜'、'飙升榜'。
    is_new (str): 是否最新，默认'Y'，如果为'N'则为盘中和盘后阶段采集。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据的最大条数，默认2000。

    返回:
    pd.DataFrame: 包含热榜数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    pro = ts.pro_api()
    
    ts_code = matcher[symbol] if symbol else None
    
    df = pro.dc_hot(
        trade_date=trade_date,
        ts_code=ts_code,
        market=market,
        hot_type=hot_type,
        is_new=is_new,
        fields='trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,rank_time'
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_ths_concept_members(symbol: str = None, code: str = None, offset: int = 0, limit: int = 200) -> pd.DataFrame:
    """
    获取同花顺概念板块成分列表。

    参数:
    symbol (str): 板块指数代码或股票代码，可选。
    code (str): 股票代码，可选。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 每次获取的数据量，默认值为200。

    返回:
    pd.DataFrame: 包含概念板块成分的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.ths_member(ts_code=ts_code, code=code, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_index_weekly_data(symbol=None, trade_date=None, start_date=None, end_date=None, fields=None, offset=0, limit=1000):
    """
    获取指数周线行情数据

    :param symbol: str, 可选, 指数代码 (如 '000001.SH')
    :param trade_date: str, 可选, 交易日期 (如 '20190329')
    :param start_date: str, 可选, 开始日期 (如 '20180101')
    :param end_date: str, 可选, 结束日期 (如 '20190329')
    :param fields: str, 可选, 返回字段 (如 'ts_code,trade_date,open,high,low,close,vol,amount')
    :param offset: int, 可选, 数据偏移量, 默认 0
    :param limit: int, 可选, 单次获取数据量, 默认 1000
    :return: pd.DataFrame, 包含指数周线行情数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None
    
    pro = ts.pro_api()
    df = pro.index_weekly(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, fields=fields, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_index_components_and_weights(
    index_code: str,
    trade_date: str,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取指定指数在特定交易日的成分和权重数据。

    参数:
    index_code (str): 指数代码，来源[指数基础信息接口](https://tushare.pro/document/2?doc_id=94)
    trade_date (str): 交易日期
    start_date (str, optional): 开始日期，默认为None
    end_date (str, optional): 结束日期，默认为None
    offset (int, optional): 数据偏移量，默认为0
    limit (int, optional): 单次获取数据的最大数量，默认为1000

    返回:
    pd.DataFrame: 包含指数成分和权重的DataFrame
    """
    pro = ts.pro_api()
    
    # 调用Tushare的index_weight接口获取数据
    df = pro.index_weight(
        index_code=index_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df


def fetch_index_daily_basic(
    trade_date: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 3000
) -> pd.DataFrame:
    """
    获取指数每日指标数据

    :param trade_date: 交易日期（格式：YYYYMMDD，比如20181018）
    :param symbol: 指数代码（A股，港股，美股，指数，期货）
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param fields: 需要返回的字段，多个字段用逗号分隔
    :param offset: 数据偏移量，默认0
    :param limit: 单次提取数据量，默认3000（单次数据上限）
    :return: DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.index_dailybasic(
        trade_date=trade_date,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

def get_index_daily(
    trade_date: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 3000
)->pd.DataFrame:
    """
    获取指数日线数据

    :param trade_date: 交易日期（格式：YYYYMMDD，比如20181018）
    :param symbol: 指数代码（A股，港股，美股，指数，期货）
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param fields: 需要返回的字段，多个字段用逗号分隔
    :param offset: 数据偏移量，默认0
    :param limit: 单次提取数据量，默认3000（单次数据上限）
    :return: DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.index_daily(
        trade_date=trade_date,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_index_monthly_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = 'ts_code,trade_date,open,high,low,close,vol,amount',
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取指数月线行情数据，每月更新一次。

    参数:
    symbol (str): 指数代码，可以是A股、港股、美股、指数、期货的代码。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    fields (str): 需要返回的字段，默认返回所有字段。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次返回数据的最大行数，默认1000行。

    返回:
    pd.DataFrame: 包含指数月线行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.index_monthly(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_ths_indices(
    symbol: str = None,
    exchange: str = None,
    type: str = None,
    offset: int = 0,
    limit: int = 5000
) -> pd.DataFrame:
    """
    获取同花顺板块指数数据。

    参数:
    symbol (str, optional): 指数代码，默认为None。
    exchange (str, optional): 市场类型，A-a股, HK-港股, US-美股，默认为None。
    type (str, optional): 指数类型，N-概念指数, I-行业指数, R-地域指数, S-同花顺特色指数, ST-同花顺风格指数, TH-同花顺主题指数, BB-同花顺宽基指数，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为5000。

    返回:
    pd.DataFrame: 包含同花顺板块指数数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()

    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.ths_index(ts_code=ts_code, exchange=exchange, type=type)

    # 去掉ts_code列中的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_sw_industry_classification(index_code=None, level=None, parent_code=None, src='SW2021', offset=0, limit=1000):
    """
    获取申万行业分类信息。

    参数:
    index_code (str): 指数代码
    level (str): 行业分级（L1/L2/L3）
    parent_code (str): 父级代码（一级为0）
    src (str): 指数来源（SW2014：申万2014年版本，SW2021：申万2021年版本），默认SW2021
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据条数，默认1000

    返回:
    DataFrame: 包含申万行业分类信息的DataFrame
    """
    pro = ts.pro_api()
    # 转换symbol为ts_code
    if index_code:
        matcher = TsCodeMatcher()
        index_code = matcher[index_code]

    # 调用API获取数据
    df = pro.index_classify(index_code=index_code, level=level, parent_code=parent_code, src=src, offset=offset, limit=limit)

    # 去除ts_code中的后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_mutual_fund_managers(offset=0, limit=1000):
    """
    获取公募基金管理人列表

    参数:
    offset (int): 数据偏移量，默认值为0
    limit (int): 每次获取的数据量，默认值为1000

    返回:
    DataFrame: 包含公募基金管理人信息的DataFrame
    """
    pro = ts.pro_api()
    df = pro.fund_company(offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_industry_components(
    l1_code: str = None,
    l2_code: str = None,
    l3_code: str = None,
    symbol: str = None,
    is_new: str = 'Y',
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    按三级分类提取申万行业成分，可提供某个分类的所有成分，也可按股票代码提取所属分类。

    参数:
    l1_code (str): 一级行业代码
    l2_code (str): 二级行业代码
    l3_code (str): 三级行业代码
    symbol (str): 股票代码（支持A股、港股、美股、指数、期货）
    is_new (str): 是否最新（默认为“Y是”）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认2000（单次最大2000行）

    返回:
    pd.DataFrame: 包含行业分类及成分股的DataFrame
    """
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    df = pro.index_member_all(
        l1_code=l1_code,
        l2_code=l2_code,
        l3_code=l3_code,
        ts_code=ts_code,
        is_new=is_new,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

def get_ths_index_daily(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 3000
) -> pd.DataFrame:
    """
    获取同花顺板块指数行情数据。

    参数:
    symbol (str): 指数代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    fields (str): 需要返回的字段，多个字段用逗号分隔。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次返回数据的最大行数，默认3000行。

    返回:
    pd.DataFrame: 包含指数行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.ths_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_stock_trade_stats(
    trade_date: str = None,
    symbol: str = None,
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 4000
) -> pd.DataFrame:
    """
    获取交易所股票交易统计数据，包括各板块明细。

    参数:
    trade_date (str): 交易日期（YYYYMMDD格式）
    symbol (str): 板块代码或股票代码（A股、港股、美股、指数、期货）
    exchange (str): 股票市场（SH上交所 SZ深交所）
    start_date (str): 开始日期
    end_date (str): 结束日期
    fields (str): 指定提取字段
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认4000

    返回:
    pd.DataFrame: 包含交易统计数据的DataFrame
    """
    pro = ts.pro_api()
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()

    # 将symbol转换为ts_code
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    df = pro.daily_info(
        trade_date=trade_date,
        ts_code=ts_code,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_ci_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 4000,
    fields: str = 'ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount'
) -> pd.DataFrame:
    """
    获取中信行业指数日线行情数据

    参数:
    symbol (str): 行业代码，支持A股、港股、美股、指数、期货代码
    trade_date (str): 交易日期（YYYYMMDD格式）
    start_date (str): 开始日期（YYYYMMDD格式）
    end_date (str): 结束日期（YYYYMMDD格式）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据条数，默认4000
    fields (str): 需要返回的字段，默认返回所有字段

    返回:
    pd.DataFrame: 包含中信行业指数日线行情数据的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.ci_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        fields=fields
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_sw_industry_daily(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 4000
) -> pd.DataFrame:
    """
    获取申万行业日线行情（默认是申万2021版行情）

    参数:
    symbol (str): 行业代码，支持A股、港股、美股、指数、期货代码
    trade_date (str): 交易日期，格式为YYYYMMDD
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    fields (str): 需要返回的字段，多个字段用逗号分隔
    offset (int): 数据偏移量，默认0
    limit (int): 单次最大返回数据行数，默认4000

    返回:
    pd.DataFrame: 包含申万行业日线行情的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.sw_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)
    
    return df

@tsdata
def get_global_index_daily(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=4000):
    """
    获取国际主要指数日线行情数据

    参数:
    symbol (str): 指数代码，如 'XIN9' 对应富时中国A50指数
    trade_date (str): 交易日期，YYYYMMDD格式
    start_date (str): 开始日期，YYYYMMDD格式
    end_date (str): 结束日期，YYYYMMDD格式
    offset (int): 数据偏移量，默认0
    limit (int): 单次提取数据量，默认4000，最大4000

    返回:
    DataFrame: 包含指数日线行情数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.index_global(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_fund_managers(symbol=None, ann_date=None, name=None, offset=0, limit=5000):
    """
    获取公募基金经理数据，包括基金经理简历等数据。

    参数:
    symbol (str, optional): 基金代码，支持多只基金，逗号分隔。
    ann_date (str, optional): 公告日期，格式：YYYYMMDD。
    name (str, optional): 基金经理姓名。
    offset (int, optional): 开始行数，默认值为0。
    limit (int, optional): 每页行数，默认值为5000。

    返回:
    pandas.DataFrame: 包含基金经理数据的DataFrame。
    """
    pro = ts.pro_api()
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    # 调用接口获取数据
    df = pro.fund_manager(ts_code=ts_code, ann_date=ann_date, name=name, offset=offset, limit=limit)

    # 去掉ts_code列中的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_mutual_fund_data(
    symbol: str = None,
    market: str = 'E',
    status: str = None,
    offset: int = 0,
    limit: int = 15000
) -> pd.DataFrame:
    """
    获取公募基金数据列表，包括场内和场外基金。

    参数:
    symbol (str, optional): 基金代码，支持A股、港股、美股、指数、期货。默认为None。
    market (str, optional): 交易市场，E表示场内，O表示场外。默认为'E'。
    status (str, optional): 存续状态，D表示摘牌，I表示发行，L表示上市中。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据的最大条数，默认为15000。

    返回:
    pd.DataFrame: 包含基金数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.fund_basic(ts_code=ts_code, market=market, status=status)

    # 去掉ts_code列中的后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_fund_dividends(
    ann_date: str = None,
    ex_date: str = None,
    pay_date: str = None,
    symbol: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取公募基金分红数据。

    参数:
    ann_date (str, optional): 公告日（以下参数四选一）
    ex_date (str, optional): 除息日
    pay_date (str, optional): 派息日
    symbol (str, optional): 基金代码（A股，港股，美股，指数，期货）
    offset (int, optional): 数据偏移量，默认0
    limit (int, optional): 单次获取数据量，默认1000

    返回:
    pd.DataFrame: 包含基金分红数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.fund_div(
        ann_date=ann_date,
        ex_date=ex_date,
        pay_date=pay_date,
        ts_code=ts_code,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_fund_holdings(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取公募基金持仓数据，季度更新。

    参数:
    symbol (str): 股票代码，可选
    ann_date (str): 公告日期（YYYYMMDD格式），可选
    start_date (str): 报告期开始日期（YYYYMMDD格式），可选
    end_date (str): 报告期结束日期（YYYYMMDD格式），可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次请求的数据量，默认1000

    返回:
    pd.DataFrame: 包含基金持仓数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None
    
    pro = ts.pro_api()
    df = pro.fund_portfolio(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_fund_nav_data(
    symbol: str = None,
    nav_date: str = None,
    market: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取公募基金净值数据。

    参数:
    symbol (str): 基金代码（A股、港股、美股、指数、期货），可选。
    nav_date (str): 净值日期，可选。
    market (str): 市场类型（E场内 O场外），可选。
    start_date (str): 净值开始日期，可选。
    end_date (str): 净值结束日期，可选。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据量，默认1000。

    返回:
    pd.DataFrame: 包含基金净值数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.fund_nav(
        ts_code=ts_code,
        nav_date=nav_date,
        market=market,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_etf_fund_share(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=2000):
    """
    获取上海和深圳ETF基金的规模数据，包含基金份额信息。

    参数:
    symbol (str, optional): 基金代码，支持多只基金同时提取，用逗号分隔。默认为None。
    trade_date (str, optional): 交易日期，格式YYYYMMDD。默认为None。
    start_date (str, optional): 开始日期，格式YYYYMMDD。默认为None。
    end_date (str, optional): 结束日期，格式YYYYMMDD。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据的最大行数，默认为2000。

    返回:
    pd.DataFrame: 包含基金代码、交易日期和基金份额的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    pro = ts.pro_api()
    matcher = TsCodeMatcher()
    
    # 将symbol转换为ts_code
    if symbol:
        symbols = symbol.split(',')
        ts_codes = [matcher[s] for s in symbols]
        ts_code = ','.join(ts_codes)
    else:
        ts_code = None
    
    # 调用接口获取数据
    df = pro.fund_share(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    # 去掉ts_code中的后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_fund_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取场内基金日线行情数据，类似股票日行情，包括ETF行情。

    参数:
    symbol (str, optional): 基金代码，支持A股、港股、美股、指数、期货代码。默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大行数，默认为2000。

    返回:
    pd.DataFrame: 包含基金日线行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.fund_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_futures_minute_data(symbol: str, freq: str, offset: int = 0, limit: int = 500) -> pd.DataFrame:
    """
    获取全市场期货合约实时分钟数据，支持1min/5min/15min/30min/60min行情。

    参数:
    symbol (str): 股票代码，支持多个合约（逗号分隔），例如：'CU2501.SHF,CU2502.SHF'
    freq (str): 分钟频度（1MIN/5MIN/15MIN/30MIN/60MIN）
    offset (int): 数据偏移量，默认从0开始
    limit (int): 单次请求的数据量，默认500条

    返回:
    pd.DataFrame: 包含期货合约分钟数据的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.rt_fut_min(ts_code=ts_code, freq=freq, offset=offset, limit=limit)

    # 去掉ts_code列中的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_fund_adjustment_factors(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=2000):
    """
    获取基金复权因子数据，用于计算基金复权行情。

    参数:
    symbol (str, optional): 基金代码（支持多只基金输入），默认为None。
    trade_date (str, optional): 交易日期（格式：yyyymmdd），默认为None。
    start_date (str, optional): 开始日期（格式：yyyymmdd），默认为None。
    end_date (str, optional): 结束日期（格式：yyyymmdd），默认为None。
    offset (int, optional): 开始行数，默认为0。
    limit (int, optional): 最大行数，默认为2000（单次最大提取行数）。

    返回:
    DataFrame: 包含基金复权因子数据的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.fund_adj(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    # 去掉ts_code列中的后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_futures_contracts(exchange: str, fut_type: str = None, fut_code: str = None, offset: int = 0, limit: int = 10000):
    """
    获取期货合约列表数据

    :param exchange: 交易所代码，必填，如'CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', 'GFEX'
    :param fut_type: 合约类型，可选，'1'表示普通合约，'2'表示主力与连续合约，默认取全部
    :param fut_code: 标准合约代码，可选，如'AG'表示白银，'AP'表示鲜苹果等
    :param offset: 数据偏移量，默认0
    :param limit: 单次获取数据的最大数量，默认10000
    :return: DataFrame，包含期货合约列表数据
    """
    pro = ts.pro_api()
    
    # 如果提供了fut_code，将其转换为ts_code
    if fut_code:
        matcher = TsCodeMatcher()
        ts_code = matcher[fut_code]
    else:
        ts_code = None
    
    # 调用Tushare接口获取数据
    df = pro.fut_basic(exchange=exchange, fut_type=fut_type, fut_code=ts_code, fields='ts_code,symbol,name,list_date,delist_date')
    
    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_futures_daily_data(
    symbol: str = None,
    trade_date: str = None,
    exchange: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 2000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取期货日线行情数据。

    参数:
    symbol (str): 合约代码或股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期 (YYYYMMDD格式)。
    exchange (str): 交易所代码。
    start_date (str): 开始日期 (YYYYMMDD格式)。
    end_date (str): 结束日期 (YYYYMMDD格式)。
    limit (int): 单次获取数据的最大条数，默认2000条。
    offset (int): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含期货日线行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.fut_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_futures_mapping(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=2000):
    """
    获取期货主力（或连续）合约与月合约映射数据

    参数:
    symbol (str): 合约代码，支持A股、港股、美股、指数、期货等。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大条数，默认值为2000。

    返回:
    DataFrame: 包含连续合约代码、起始日期、期货合约代码的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.fut_mapping(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)

    return df

@tsdata
def get_sge_basic_info(symbol=None, offset=0, limit=20):
    """
    获取上海黄金交易所现货合约基础信息。

    参数:
    symbol (str, optional): 合约代码，支持多个，逗号分隔，不输入为获取全部。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大条数，默认值为20（根据文档说明）。

    返回:
    pd.DataFrame: 包含现货合约基础信息的DataFrame。
    """
    # 如果提供了symbol参数，将其转换为ts_code
    if symbol:
        from core.stock.ts_code_matcher import TsCodeMatcher
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol]
    else:
        ts_code = None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.sge_basic(ts_code=ts_code)

    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_futures_holding_data(trade_date=None, symbol=None, start_date=None, end_date=None, exchange=None, offset=0, limit=2000):
    """
    获取每日成交持仓排名数据

    参数:
    trade_date (str): 交易日期 (YYYYMMDD格式)
    symbol (str): 合约或产品代码
    start_date (str): 开始日期 (YYYYMMDD格式)
    end_date (str): 结束日期 (YYYYMMDD格式)
    exchange (str): 交易所代码
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认2000

    返回:
    DataFrame: 包含每日成交持仓排名数据的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    pro = ts.pro_api()
    df = pro.fut_holding(trade_date=trade_date, symbol=ts_code, start_date=start_date, end_date=end_date, exchange=exchange, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_futures_minute_data(symbol: str, freq: str, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 8000) -> pd.DataFrame:
    """
    获取全市场期货合约分钟数据，支持1min/5min/15min/30min/60min行情。

    参数:
    symbol (str): 期货合约代码，如CU2310.SHF
    freq (str): 分钟频度（1min/5min/15min/30min/60min）
    start_date (str, optional): 开始日期，格式：2023-08-25 09:00:00
    end_date (str, optional): 结束日期，格式：2023-08-25 19:00:00
    offset (int, optional): 数据偏移量，默认0
    limit (int, optional): 单次获取数据量，默认8000（单次最大8000行数据）

    返回:
    pd.DataFrame: 包含分钟数据的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.ft_mins(ts_code=ts_code, freq=freq, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_warehouse_receipt_daily(pro, trade_date=None, symbol=None, start_date=None, end_date=None, exchange=None, offset=0, limit=1000):
    """
    获取仓单日报数据，了解各仓库/厂库的仓单变化。

    参数:
    pro (ts.pro_api): Tushare API对象
    trade_date (str): 交易日期 (YYYYMMDD)
    symbol (str): 产品代码
    start_date (str): 开始日期 (YYYYMMDD)
    end_date (str): 结束日期 (YYYYMMDD)
    exchange (str): 交易所代码
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认1000

    返回:
    pd.DataFrame: 包含仓单日报数据的DataFrame
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare API获取数据
    df = pro.fut_wsr(trade_date=trade_date, symbol=ts_code, start_date=start_date, end_date=end_date, exchange=exchange)

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_daily_settlement_params(
    trade_date: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    exchange: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取每日结算参数数据，包括交易和交割费率等。

    参数:
    trade_date (str, 可选): 交易日期 (YYYYMMDD格式)
    symbol (str, 可选): 合约代码或股票代码 (A股, 港股, 美股, 指数, 期货)
    start_date (str, 可选): 开始日期 (YYYYMMDD格式)
    end_date (str, 可选): 结束日期 (YYYYMMDD格式)
    exchange (str, 可选): 交易所代码
    offset (int, 可选): 数据偏移量，默认0
    limit (int, 可选): 单次获取数据量，默认1000

    返回:
    pd.DataFrame: 包含每日结算参数数据的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.fut_settle(
        trade_date=trade_date,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df


@tsdata
def fetch_futures_weekly_monthly(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = 'week',
    exchange: str = None,
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取期货的周/月线行情数据。

    参数:
    symbol (str, optional): 期货代码或简称。默认为None。
    trade_date (str, optional): 交易日期。默认为None。
    start_date (str, optional): 开始日期。默认为None。
    end_date (str, optional): 结束日期。默认为None。
    freq (str, optional): 频率，'week'表示周，'month'表示月。默认为'week'。
    exchange (str, optional): 交易所。默认为None。
    offset (int, optional): 数据偏移量。默认为0。
    limit (int, optional): 单次获取数据的最大数量。默认为6000。

    返回:
    pd.DataFrame: 包含期货周/月线行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    pro = ts.pro_api()

    # 调用底层API获取数据
    data = pro.fut_weekly_monthly(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        exchange=exchange,
        offset=offset,
        limit=limit
    )

    # 处理数据
    df = pd.DataFrame(data)
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_futures_weekly_stats(
    week: str = None,
    prd: str = None,
    start_week: str = None,
    end_week: str = None,
    exchange: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 4000
) -> pd.DataFrame:
    """
    获取期货交易所主要品种每周交易统计信息。

    参数:
    week (str): 周期（每年第几周，e.g. 202001 表示2020第1周）
    prd (str): 期货品种（支持多品种输入，逗号分隔）
    start_week (str): 开始周期
    end_week (str): 结束周期
    exchange (str): 交易所（请参考[交易所说明](https://tushare.pro/document/2?doc_id=134)）
    fields (str): 提取的字段，e.g. fields='prd,name,vol'
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认4000（根据文档单次最大获取4000行数据）

    返回:
    pd.DataFrame: 包含期货每周交易统计信息的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    pro = ts.pro_api()

    if prd:
        prd = ','.join([matcher[symbol] for symbol in prd.split(',')])

    df = pro.fut_weekly_detail(
        week=week,
        prd=prd,
        start_week=start_week,
        end_week=end_week,
        exchange=exchange,
        fields=fields
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_sge_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 2000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取上海黄金交易所现货合约日线行情数据。

    参数:
    symbol (str, optional): 合约代码，可通过[基础信息](https://tushare.pro/document/2?doc_id=284)获得。默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。默认为None。
    limit (int, optional): 单次提取的最大数据量，默认为2000。
    offset (int, optional): 数据偏移量，默认为0。

    返回:
    pd.DataFrame: 包含现货合约日线行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.sge_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_option_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    exchange: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取期权日线行情数据

    参数:
    symbol (str): 合约代码（输入代码或时间至少任意一个参数）
    trade_date (str): 交易日期
    start_date (str): 开始日期
    end_date (str): 结束日期
    exchange (str): 交易所(SSE/SZSE/CFFEX/DCE/SHFE/CZCE）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认1000

    返回:
    pd.DataFrame: 包含期权日线行情数据的DataFrame
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare的opt_daily接口
    pro = ts.pro_api()
    df = pro.opt_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        offset=offset,
        limit=limit
    )

    # 去掉ts_code中的后缀，生成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_option_contract_info(
    symbol: str = None,
    exchange: str = None,
    opt_code: str = None,
    call_put: str = None,
    limit: int = 1000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取期权合约信息

    参数:
    symbol (str): 股票、指数、期货的代码，可选
    exchange (str): 交易所代码，可选
    opt_code (str): 标准合约代码，可选
    call_put (str): 期权类型，可选
    limit (int): 单次返回数据条数，默认1000
    offset (int): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含期权合约信息的DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.opt_basic(
        ts_code=ts_code,
        exchange=exchange,
        opt_code=opt_code,
        call_put=call_put,
        fields='ts_code,exchange,name,per_unit,opt_code,opt_type,call_put,exercise_type,exercise_price,s_month,maturity_date,list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg'
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_convertible_bond_price_changes(symbol: str, offset: int = 0, limit: int = 2000) -> pd.DataFrame:
    """
    获取可转债转股价变动数据

    :param symbol: 转债代码，支持多值输入，格式为逗号分隔的字符串
    :param offset: 数据偏移量，默认从0开始
    :param limit: 单次获取数据的最大数量，默认2000
    :return: 包含转股价变动信息的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.cb_price_chg(ts_code=ts_code, fields="ts_code,bond_short_name,change_date,convert_price_initial,convertprice_bef,convertprice_aft")

    # 去掉ts_code中的后缀，形成新的code列
    df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_convertible_bond_rates(symbol: str, limit: int = 2000, offset: int = 0) -> pd.DataFrame:
    """
    获取可转债的票面利率信息。

    参数:
    symbol (str): 转债代码或符号，支持多值输入，用逗号分隔。
    limit (int): 单次获取数据的最大数量，默认值为2000。
    offset (int): 数据偏移量，默认值为0。

    返回:
    pd.DataFrame: 包含可转债票面利率信息的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.cb_rate(ts_code=ts_code, fields="ts_code,rate_freq,rate_start_date,rate_end_date,coupon_rate")

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_convertible_bond_info(
    symbol: str = None,
    list_date: str = None,
    exchange: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取可转债基本信息列表。

    参数:
    symbol (str, optional): 转债代码或简称，默认为None。
    list_date (str, optional): 上市日期，格式为YYYY-MM-DD，默认为None。
    exchange (str, optional): 上市地点，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为2000。

    返回:
    pd.DataFrame: 包含可转债基本信息的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.cb_basic(
        ts_code=ts_code,
        list_date=list_date,
        exchange=exchange,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_convertible_bond_conversion_results(symbol: str, limit: int = 2000, offset: int = 0) -> pd.DataFrame:
    """
    获取可转债转股结果

    :param symbol: 转债代码，支持多值输入，如 "113001.SH,110027.SH"
    :param limit: 单次获取数据的最大数量，默认2000
    :param offset: 数据偏移量，默认0
    :return: DataFrame，包含可转债转股结果的数据
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.cb_share(ts_code=ts_code, fields="ts_code,end_date,convert_price,convert_val,convert_ratio,acc_convert_ratio", limit=limit, offset=offset)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_option_minute_data(symbol: str, freq: str, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 8000) -> pd.DataFrame:
    """
    获取全市场期权合约分钟数据，支持1min/5min/15min/30min/60min行情。

    :param symbol: 股票代码，支持A股、港股、美股、指数、期货等。
    :param freq: 分钟频度（1min/5min/15min/30min/60min）。
    :param start_date: 开始日期，格式：'2024-08-25 09:00:00'。
    :param end_date: 结束时间，格式：'2024-08-25 19:00:00'。
    :param offset: 数据偏移量，默认从0开始。
    :param limit: 单次请求的最大数据行数，默认8000行。
    :return: 包含分钟数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]
    
    pro = ts.pro_api()
    df = pro.opt_mins(ts_code=ts_code, freq=freq, start_date=start_date, end_date=end_date, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_convertible_bond_issue_data(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 2000,
    offset: int = 0,
) -> pd.DataFrame:
    """
    获取可转债发行数据

    参数:
    symbol (str): 股票代码，可选，默认为None
    ann_date (str): 发行公告日，可选，默认为None
    start_date (str): 公告开始日期，可选，默认为None
    end_date (str): 公告结束日期，可选，默认为None
    limit (int): 单次提取数据的最大数量，默认为2000
    offset (int): 数据偏移量，默认为0

    返回:
    pd.DataFrame: 包含可转债发行数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.cb_issue(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_otc_bond_quotes(trade_date=None, start_date=None, end_date=None, symbol=None, bank=None, offset=0, limit=2000):
    """
    获取柜台流通式债券报价数据

    参数:
    trade_date (str): 交易日期 (YYYYMMDD格式)
    start_date (str): 开始日期 (YYYYMMDD格式)
    end_date (str): 结束日期 (YYYYMMDD格式)
    symbol (str): 债券代码 (A股、港股、美股、指数、期货)
    bank (str): 报价机构
    offset (int): 数据偏移量，默认0
    limit (int): 单次提取数据量，默认2000

    返回:
    DataFrame: 包含债券报价数据的DataFrame
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.bc_otcqt(trade_date=trade_date, start_date=start_date, end_date=end_date, ts_code=ts_code, bank=bank, offset=offset, limit=limit)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_convertible_bond_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    获取可转债行情数据。

    参数:
    symbol (str, optional): 可转债代码或简称，默认为None。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD，默认为None。
    start_date (str, optional): 开始日期，格式为YYYYMMDD，默认为None。
    end_date (str, optional): 结束日期，格式为YYYYMMDD，默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大条数，默认为2000。

    返回:
    pd.DataFrame: 包含可转债行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.cb_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_bond_block_trades(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取沪深交易所债券大宗交易数据。

    参数:
    symbol (str): 债券代码，可选。
    trade_date (str): 交易日期，格式为YYYYMMDD，可选。
    start_date (str): 开始日期，格式为YYYYMMDD，可选。
    end_date (str): 结束日期，格式为YYYYMMDD，可选。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大数量，默认值为1000。

    返回:
    DataFrame: 包含债券大宗交易数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.bond_blk_detail(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_repo_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取债券回购日行情数据。

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大条数，默认2000条。

    返回:
    pd.DataFrame: 包含债券回购日行情数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.repo_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)

    return df

@tsdata
def fetch_bond_block_trades(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取沪深交易所债券大宗交易数据。

    参数:
    symbol (str, optional): 债券代码或简称，默认为None。
    trade_date (str, optional): 交易日期（YYYYMMDD格式），默认为None。
    start_date (str, optional): 开始日期（YYYYMMDD格式），默认为None。
    end_date (str, optional): 结束日期（YYYYMMDD格式），默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为1000。

    返回:
    pandas.DataFrame: 包含债券大宗交易数据的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.bond_blk(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_best_otc_bond_quotes(
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    symbol: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取柜台流通式债券最优报价

    :param trade_date: 报价日期(YYYYMMDD格式)
    :param start_date: 开始日期(YYYYMMDD格式)
    :param end_date: 结束日期(YYYYMMDD格式)
    :param symbol: 债券代码或简称
    :param offset: 数据偏移量，默认0
    :param limit: 单次提取数据量，默认2000
    :return: DataFrame
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.bc_bestotcqt(
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        ts_code=ts_code,
        offset=offset,
        limit=limit,
        fields='trade_date,ts_code,name,remain_maturity,best_buy_bank,best_buy_yield,best_sell_bank,best_sell_yield'
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_convertible_bond_call_info(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取可转债到期赎回、强制赎回等信息。

    数据来源于公开披露渠道，供个人和机构研究使用，请不要用于数据商业目的。

    参数:
    symbol (str, optional): 转债代码，支持多值输入。默认为None。
    ann_date (str, optional): 公告日期(YYYYMMDD格式)。默认为None。
    start_date (str, optional): 公告开始日期(YYYYMMDD格式)。默认为None。
    end_date (str, optional): 公告结束日期(YYYYMMDD格式)。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大条数，默认为1000。

    返回:
    pd.DataFrame: 包含可转债赎回信息的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()

    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    pro = ts.pro_api()
    df = pro.cb_call(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,call_vol,call_amount,payment_date,call_reg_date'
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_fxcm_basic_info(exchange=None, classify=None, symbol=None, offset=0, limit=1000):
    """
    获取FXCM交易商的外汇基础信息。

    参数:
    exchange (str): 交易商，默认为None。
    classify (str): 分类，默认为None。
    symbol (str): 标的代码，默认为None。如果提供，将自动转换为ts_code。
    offset (int): 数据偏移量，默认为0。
    limit (int): 单次提取数据的最大数量，默认为1000。

    返回:
    DataFrame: 包含外汇基础信息的DataFrame。
    """
    # 如果提供了symbol，将其转换为ts_code
    if symbol:
        matcher = TsCodeMatcher()
        ts_code = matcher[symbol]
    else:
        ts_code = None

    # 调用Tushare的pro_api获取数据
    pro = ts.pro_api()
    df = pro.fx_obasic(exchange=exchange, classify=classify, ts_code=ts_code, fields='ts_code,name,min_unit,max_unit,pip,pip_cost,traget_spread,min_stop_distance,trading_hours,break_time')

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_hk_stock_list(symbol=None, list_status='L', offset=0, limit=5000):
    """
    获取港股列表信息

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    list_status (str, optional): 上市状态，L表示上市，D表示退市，P表示暂停上市。默认为'L'。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据量，默认为5000。

    返回:
    pandas.DataFrame: 包含港股列表信息的DataFrame，包含去除后缀的code列。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None
    
    pro = ts.pro_api()
    df = pro.hk_basic(ts_code=ts_code, list_status=list_status)
    
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_cb_yield_curve(
    symbol: str = None,
    curve_type: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    curve_term: float = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取中债收益率曲线数据。

    参数:
    symbol (str): 收益率曲线编码，例如 '1001.CB'。如果为A股、港股、美股、指数、期货，则自动转换为对应的ts_code。
    curve_type (str): 曲线类型：'0'-到期，'1'-即期。
    trade_date (str): 交易日期，格式为YYYYMMDD。
    start_date (str): 查询起始日期，格式为YYYYMMDD。
    end_date (str): 查询结束日期，格式为YYYYMMDD。
    curve_term (float): 期限(年)。
    offset (int): 数据偏移量，默认为0。
    limit (int): 单次获取数据的最大数量，默认为2000。

    返回:
    pd.DataFrame: 包含收益率曲线数据的DataFrame。
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    pro = ts.pro_api()
    df = pro.yc_cb(
        ts_code=ts_code,
        curve_type=curve_type,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        curve_term=curve_term,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_hk_trade_calendar(start_date: str = None, end_date: str = None, is_open: str = None, offset: int = 0, limit: int = 2000) -> pd.DataFrame:
    """
    获取香港交易所的交易日历数据。

    参数:
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    is_open (str, optional): 是否交易，'0'表示休市，'1'表示交易。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大数量，默认为2000。

    返回:
    pd.DataFrame: 包含交易日历数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.hk_tradecal(start_date=start_date, end_date=end_date, is_open=is_open)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_fx_daily_data(symbol=None, trade_date=None, start_date=None, end_date=None, exchange='FXCM', offset=0, limit=1000):
    """
    获取外汇日线行情数据

    参数:
    symbol (str): 外汇代码或股票代码，默认为None
    trade_date (str): 交易日期（GMT），默认为None
    start_date (str): 开始日期（GMT），默认为None
    end_date (str): 结束日期（GMT），默认为None
    exchange (str): 交易商，默认为'FXCM'
    offset (int): 数据偏移量，默认为0
    limit (int): 单次提取数据量，默认为1000

    返回:
    pd.DataFrame: 包含外汇日线行情数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    
    if symbol:
        ts_code = matcher[symbol]
    else:
        ts_code = None

    pro = ts.pro_api()
    df = pro.fx_daily(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, exchange=exchange, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_hk_stock_data(symbol=None, trade_date=None, start_date=None, end_date=None, offset=0, limit=5000):
    """
    获取港股每日增量和历史行情数据。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。
    trade_date (str, optional): 交易日期，格式为YYYYMMDD。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次提取数据的最大行数，默认值为5000。

    返回:
    pandas.DataFrame: 包含港股行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.hk_daily(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def fetch_global_economic_calendar(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    currency: str = None,
    country: str = None,
    event: str = None,
    limit: int = 100,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取全球财经日历，包括经济事件数据更新。

    参数:
    date (str, optional): 日期（YYYYMMDD格式）
    start_date (str, optional): 开始日期
    end_date (str, optional): 结束日期
    currency (str, optional): 货币代码
    country (str, optional): 国家（比如：中国、美国）
    event (str, optional): 事件 （支持模糊匹配： *非农*）
    limit (int, optional): 单次获取的最大行数，默认100
    offset (int, optional): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含全球财经日历数据的DataFrame
    """
    pro = ts.pro_api()
    
    params = {
        'date': date,
        'start_date': start_date,
        'end_date': end_date,
        'currency': currency,
        'country': country,
        'event': event,
        'limit': limit,
        'offset': offset
    }
    
    df = pro.eco_cal(**params)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_us_stock_list(
    symbol: str = None, 
    classify: str = None, 
    offset: int = 0, 
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取美股列表信息。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    classify (str, optional): 股票分类，可选值为ADR/GDR/EQ。默认为None。
    offset (int, optional): 开始行数，默认为0。
    limit (int, optional): 每页最大行数，默认为6000（单次取数据上限）。

    返回:
    pd.DataFrame: 包含美股列表信息的DataFrame。
    """
    from core.utils.ts_data import tsdata
    from core.stock.ts_code_matcher import TsCodeMatcher

    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.us_basic(ts_code=ts_code, classify=classify, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)

    return df

@tsdata
def fetch_weekly_box_office(date: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取周度票房数据

    参数:
    date (str): 日期（每周一日期，格式YYYYMMDD）
    offset (int): 数据偏移量，默认0
    limit (int): 数据条数限制，默认1000

    返回:
    pd.DataFrame: 包含周度票房数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.bo_weekly(date=date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_us_trade_calendar(start_date: str = None, end_date: str = None, is_open: str = None, offset: int = 0, limit: int = 6000) -> pd.DataFrame:
    """
    获取美股交易日历信息

    参数:
    start_date (str, optional): 开始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    is_open (str, optional): 是否交易，'0'表示休市，'1'表示交易
    offset (int, optional): 数据偏移量，默认为0
    limit (int, optional): 单次获取数据的最大数量，默认为6000

    返回:
    pd.DataFrame: 包含美股交易日历信息的DataFrame
    """
    pro = ts.pro_api()
    df = pro.us_tradecal(start_date=start_date, end_date=end_date, is_open=is_open)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_monthly_box_office(date: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取电影月度票房数据

    参数:
    date (str): 日期（每月1号，格式YYYYMMDD）
    offset (int): 数据偏移量，默认从0开始
    limit (int): 每次获取的数据量，默认1000条

    返回:
    pd.DataFrame: 包含电影月度票房数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.bo_monthly(date=date, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_daily_box_office(date: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取电影日度票房数据

    参数:
    date (str): 日期，格式为YYYYMMDD
    offset (int): 数据偏移量，默认值为0
    limit (int): 每次获取的数据量，默认值为1000

    返回:
    pd.DataFrame: 包含电影日度票房数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.bo_daily(date=date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_tmt_twincome(item, start_date=None, end_date=None, offset=0, limit=30):
    """
    获取台湾TMT电子产业领域各类产品月度营收数据。

    参数:
    item (str): 产品代码，必选。
    start_date (str, optional): 报告期开始日期，格式为YYYYMMDD。
    end_date (str, optional): 报告期结束日期，格式为YYYYMMDD。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次获取数据的最大数量，默认值为30。

    返回:
    pandas.DataFrame: 包含月度营收数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.tmt_twincome(item=item, start_date=start_date, end_date=end_date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_daily_cinema_box_office(date: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取每日各影院的票房数据

    参数:
    date (str): 日期，格式为YYYYMMDD
    offset (int): 数据偏移量，默认为0
    limit (int): 每次获取的数据量，默认为1000

    返回:
    pd.DataFrame: 包含每日各影院票房数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.bo_cinema(date=date, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_hk_minute_data(symbol: str, freq: str, start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 8000) -> pd.DataFrame:
    """
    获取港股分钟级别行情数据

    参数:
    symbol (str): 股票代码，支持A股、港股、美股、指数、期货等
    freq (str): 分钟频度（1min/5min/15min/30min/60min）
    start_date (datetime, optional): 开始日期，格式：2023-03-13 09:00:00
    end_date (datetime, optional): 结束时间，格式：2023-03-13 19:00:00
    offset (int, optional): 数据偏移量，默认0
    limit (int, optional): 单次获取数据量，默认8000（根据文档单次最大8000行数据）

    返回:
    pd.DataFrame: 包含分钟级别行情数据的DataFrame
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol]

    pro = ts.pro_api()
    df = pro.hk_mins(ts_code=ts_code, freq=freq, start_date=start_date, end_date=end_date, offset=offset, limit=limit)

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)

    return df

@tsdata
def get_hk_adjusted_daily_data(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取港股复权行情数据，包含股票股本、市值和成交及换手等多个数据指标。

    参数:
    symbol (str, optional): 股票代码（e.g. 00001），默认为None。
    trade_date (str, optional): 交易日期（YYYYMMDD），默认为None。
    start_date (str, optional): 开始日期（YYYYMMDD），默认为None。
    end_date (str, optional): 结束日期（YYYYMMDD），默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次提取数据的最大条数，默认为6000。

    返回:
    pd.DataFrame: 包含港股复权行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.hk_daily_adj(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_national_film_record(
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 500,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取全国电影剧本备案的公示数据。

    参数:
    ann_date (str): 公布日期（格式：YYYYMMDD，日期不连续，定期公布）
    start_date (str): 开始日期
    end_date (str): 结束日期
    limit (int): 单次获取数据的最大数量，默认500
    offset (int): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含电影剧本备案公示数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.film_record(
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_taiwan_tmt_monthly_revenue(
    date: str = None,
    item: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    source: str = None,
    offset: int = 0,
    limit: int = 1000
) -> pd.DataFrame:
    """
    获取台湾TMT行业上市公司各类产品月度营收情况。

    参数:
    date (str): 报告期
    item (str): 产品代码
    symbol (str): 公司代码
    start_date (str): 报告期开始日期
    end_date (str): 报告期结束日期
    source (str): 数据来源 (默认None)
    offset (int): 数据偏移量 (默认0)
    limit (int): 单次获取数据上限 (默认1000)

    返回:
    pd.DataFrame: 包含月度营收数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.tmt_twincomedetail(
        date=date,
        item=item,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        source=source
    )
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_us_stock_daily_adjusted(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    exchange: str = None,
    offset: int = 0,
    limit: int = 8000
) -> pd.DataFrame:
    """
    获取美股复权行情数据，支持美股全市场股票，提供股本、市值、复权因子和成交信息等多个数据指标。

    参数:
    symbol (str): 股票代码（e.g. AAPL），支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期（YYYYMMDD）。
    start_date (str): 开始日期（YYYYMMDD）。
    end_date (str): 结束日期（YYYYMMDD）。
    exchange (str): 交易所（NAS/NYS/OTC）。
    offset (int): 开始行数，默认值为0。
    limit (int): 每页行数，默认值为8000（单次取数据上限）。

    返回:
    pd.DataFrame: 包含复权行情数据的DataFrame。
    """
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.us_daily_adj(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def get_us_stock_history(
    symbol: str = None,
    trade_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 6000
) -> pd.DataFrame:
    """
    获取美股行情（未复权），包括全部股票全历史行情，以及重要的市场和估值指标。

    参数:
    symbol (str): 股票代码（e.g. AAPL），支持A股、港股、美股、指数、期货。
    trade_date (str): 交易日期（YYYYMMDD）。
    start_date (str): 开始日期（YYYYMMDD）。
    end_date (str): 结束日期（YYYYMMDD）。
    offset (int): 数据偏移量，默认0。
    limit (int): 单次获取数据的最大行数，默认6000。

    返回:
    pd.DataFrame: 包含美股行情数据的DataFrame。
    """
    # 将symbol转换为ts_code
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    # 调用Tushare接口获取数据
    pro = ts.pro_api()
    df = pro.us_daily(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    # 如果结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df

@tsdata
def fetch_shibor_rates(start_date: str = None, end_date: str = None, offset: int = 0, limit: int = 2000) -> pd.DataFrame:
    """
    获取上海银行间同业拆放利率（Shibor）数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大数量，默认2000条。

    返回:
    pd.DataFrame: Shibor利率数据。
    """
    pro = ts.pro_api()
    df = pro.shibor(start_date=start_date, end_date=end_date)
    
    # 如果输出结果包含ts_code列，去掉后缀.XX
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_shibor_quote_data(start_date=None, end_date=None, bank=None, offset=0, limit=4000):
    """
    获取Shibor报价数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    bank (str): 银行名称，中文名称，例如“农业银行”。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大行数，默认4000行。

    返回:
    DataFrame: 包含Shibor报价数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.shibor_quote(start_date=start_date, end_date=end_date, bank=bank)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_lpr_data(start_date=None, end_date=None, date=None, fields=None, offset=0, limit=1000):
    """
    获取LPR贷款基础利率数据

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    date (str): 具体日期，格式为YYYYMMDD
    fields (str): 需要获取的字段，多个字段用逗号分隔
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大数量，默认1000

    返回:
    DataFrame: 包含LPR数据的DataFrame
    """
    pro = ts.pro_api()
    
    # 调用Tushare接口获取LPR数据
    df = pro.shibor_lpr(start_date=start_date, end_date=end_date, date=date, fields=fields)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_libor_rates(
    start_date: str = None,
    end_date: str = None,
    curr_type: str = 'USD',
    limit: int = 4000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取Libor拆借利率数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    curr_type (str): 货币代码，可选值为USD（美元）、EUR（欧元）、JPY（日元）、GBP（英镑）、CHF（瑞郎），默认是USD。
    limit (int): 单次获取数据的最大行数，默认值为4000。
    offset (int): 数据偏移量，默认值为0。

    返回:
    pd.DataFrame: 包含Libor拆借利率数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.libor(
        start_date=start_date,
        end_date=end_date,
        curr_type=curr_type,
        limit=limit,
        offset=offset
    )
    
    return df

@tsdata
def fetch_wz_index_data(start_date=None, end_date=None, offset=0, limit=1000):
    """
    获取温州民间借贷利率指数（温州指数）的历史数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 每次获取的数据量，默认1000条。

    返回:
    pd.DataFrame: 包含温州指数历史数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.wz_index(start_date=start_date, end_date=end_date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.replace(r'\.\w+$', '', regex=True)
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_hibor_data(start_date=None, end_date=None, date=None, offset=0, limit=1000):
    """
    获取香港银行同业拆借利率（HIBOR）数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    date (str): 特定日期，格式为YYYYMMDD。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 每次获取的数据量，默认1000条。

    返回:
    pandas.DataFrame: 包含HIBOR数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.hibor(start_date=start_date, end_date=end_date, date=date)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_teleplay_records(
    report_date: str = None,
    start_date: str = None,
    end_date: str = None,
    org: str = None,
    name: str = None,
    limit: int = 1000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取2009年以来全国拍摄制作电视剧备案公示数据。

    参数:
    report_date (str): 备案月份（YYYYMM）
    start_date (str): 备案开始月份（YYYYMM）
    end_date (str): 备案结束月份（YYYYMM）
    org (str): 备案机构
    name (str): 电视剧名称
    limit (int): 单次获取数据的最大数量，默认1000
    offset (int): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含电视剧备案信息的DataFrame
    """
    pro = ts.pro_api()
    
    params = {
        'report_date': report_date,
        'start_date': start_date,
        'end_date': end_date,
        'org': org,
        'name': name,
        'limit': limit,
        'offset': offset
    }
    
    df = pro.teleplay_record(**params)
    
    # 如果结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_guangzhou_private_loan_rates(date=None, start_date=None, end_date=None, limit=1000, offset=0):
    """
    获取广州民间借贷利率数据

    参数:
    date (str): 日期，格式为YYYYMMDD
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    limit (int): 每次获取的数据条数，默认1000
    offset (int): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含广州民间借贷利率数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.gz_index(date=date, start_date=start_date, end_date=end_date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_monthly_social_finance_data(start_m=None, end_m=None, m=None, offset=0, limit=2000):
    """
    获取月度社会融资数据

    参数:
    start_m (str): 开始月份 (YYYYMM)
    end_m (str): 结束月份 (YYYYMM)
    m (str): 月份 (YYYYMM)，支持多个月份同时输入，逗号分隔
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大条数，默认2000

    返回:
    DataFrame: 包含月度社会融资数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.sf_month(start_m=start_m, end_m=end_m, m=m)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_cpi_data(m=None, start_m=None, end_m=None, limit=5000, offset=0):
    """
    获取CPI居民消费价格数据，包括全国、城市和农村的数据。

    参数:
    m (str, optional): 月份（YYYYMM），支持多个月份同时输入，逗号分隔。
    start_m (str, optional): 开始月份（YYYYMM）。
    end_m (str, optional): 结束月份（YYYYMM）。
    limit (int, optional): 每次请求的最大行数，默认5000行。
    offset (int, optional): 数据偏移量，默认0。

    返回:
    pandas.DataFrame: 包含CPI数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.cn_cpi(m=m, start_m=start_m, end_m=end_m, limit=limit, offset=offset)
    
    # 如果输出结果包含ts_code列，去掉000000.XX的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_cn_gdp_data(q=None, start_q=None, end_q=None, fields=None, offset=0, limit=10000):
    """
    获取中国GDP及相关数据。

    参数:
    q (str): 指定季度，格式为'YYYYQX'，例如'2019Q1'。
    start_q (str): 开始季度，格式为'YYYYQX'，例如'2018Q1'。
    end_q (str): 结束季度，格式为'YYYYQX'，例如'2019Q3'。
    fields (str): 指定输出字段，例如'quarter,gdp,gdp_yoy'。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大数量，默认10000。

    返回:
    pd.DataFrame: 包含GDP及相关数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.cn_gdp(q=q, start_q=start_q, end_q=end_q, fields=fields)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_ppi_data(m=None, start_m=None, end_m=None, fields=None, offset=0, limit=5000):
    """
    获取中国工业生产者出厂价格指数（PPI）数据。

    参数:
    m (str, optional): 月份（YYYYMM），支持多个月份同时输入，逗号分隔。
    start_m (str, optional): 开始月份（YYYYMM）。
    end_m (str, optional): 结束月份（YYYYMM）。
    fields (str, optional): 需要返回的字段，逗号分隔。
    offset (int, optional): 数据偏移量，默认值为0。
    limit (int, optional): 单次返回数据的最大条数，默认值为5000。

    返回:
    pandas.DataFrame: 包含PPI数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.cn_ppi(m=m, start_m=start_m, end_m=end_m, fields=fields, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉000000.XX的.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_monthly_money_supply(m=None, start_m=None, end_m=None, fields=None, offset=0, limit=5000):
    """
    获取货币供应量之月度数据

    参数:
    m (str): 月度（202001表示，2020年1月）
    start_m (str): 开始月度
    end_m (str): 结束月度
    fields (str): 指定输出字段（e.g. fields='month,m0,m1,m2'）
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据量，默认5000

    返回:
    DataFrame: 包含货币供应量月度数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.cn_m(m=m, start_m=start_m, end_m=end_m, fields=fields, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀，形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_us_treasury_yields(start_date=None, end_date=None, fields=None, offset=0, limit=2000):
    """
    获取美国国债实际收益率曲线利率数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    fields (str): 指定输出字段，如'y5,y20'。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大行数，默认2000行。

    返回:
    pd.DataFrame: 包含美国国债实际收益率曲线利率数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.us_trycr(start_date=start_date, end_date=end_date, fields=fields, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_pmi_data(
    m: str = None,
    start_m: str = None,
    end_m: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取中国采购经理人指数（PMI）数据。

    参数:
    m (str): 指定月份，格式为YYYYMM。
    start_m (str): 开始月份，格式为YYYYMM。
    end_m (str): 结束月份，格式为YYYYMM。
    fields (str): 需要获取的字段，多个字段用逗号分隔。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 每次获取的数据量，默认2000条。

    返回:
    pd.DataFrame: 包含PMI数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.cn_pmi(
        m=m,
        start_m=start_m,
        end_m=end_m,
        fields=fields,
        offset=offset,
        limit=limit
    )
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_us_treasury_rates(start_date=None, end_date=None, fields=None, limit=2000, offset=0):
    """
    获取美国国债长期利率数据

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    fields (str): 指定字段，多个字段用逗号分隔
    limit (int): 每次获取的数据行数，默认2000
    offset (int): 数据偏移量，默认0

    返回:
    pd.DataFrame: 包含美国国债长期利率数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.us_tltr(start_date=start_date, end_date=end_date, fields=fields, limit=limit, offset=offset)
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_us_treasury_rates(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取美国短期国债利率数据。

    参数:
    date (str): 日期 (YYYYMMDD格式)
    start_date (str): 开始日期 (YYYYMMDD格式)
    end_date (str): 结束日期 (YYYYMMDD格式)
    fields (str): 指定输出字段 (e.g. fields='w4_bd,w52_ce')
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大行数，默认2000

    返回:
    pd.DataFrame: 包含美国短期国债利率数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.us_tbr(
        date=date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        offset=offset,
        limit=limit
    )
    
    return df

@tsdata
def fetch_news_data(start_date: str, end_date: str, src: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
    """
    获取主流新闻网站的快讯新闻数据。

    :param start_date: 开始日期，格式为 'YYYY-MM-DD HH:MM:SS'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD HH:MM:SS'
    :param src: 新闻来源标识，可选值见文档
    :param offset: 数据偏移量，默认从0开始
    :param limit: 单次获取数据的最大条数，默认1000条
    :return: 包含新闻数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.news(src=src, start_date=start_date, end_date=end_date,fields=['datetime','content','title'], offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_us_trltr_data(start_date=None, end_date=None, fields=None, offset=0, limit=2000):
    """
    获取美国国债实际长期利率平均值数据。

    参数:
    start_date (str): 开始日期，格式为YYYYMMDD。
    end_date (str): 结束日期，格式为YYYYMMDD。
    fields (str): 指定字段，多个字段用逗号分隔。
    offset (int): 数据偏移量，默认从0开始。
    limit (int): 单次获取数据的最大行数，默认2000行。

    返回:
    pd.DataFrame: 包含国债实际长期利率平均值的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.us_trltr(start_date=start_date, end_date=end_date, fields=fields, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉后缀.XX
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df


@tsdata
def fetch_us_treasury_yields(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    fields: str = None,
    limit: int = 2000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取美国每日国债收益率曲线利率数据。

    参数:
    date (str, optional): 日期，格式为YYYYMMDD。
    start_date (str, optional): 开始日期，格式为YYYYMMDD。
    end_date (str, optional): 结束日期，格式为YYYYMMDD。
    fields (str, optional): 指定输出字段，例如'm1,y1'。
    limit (int, optional): 单次获取数据的最大条数，默认2000条。
    offset (int, optional): 数据偏移量，默认0。

    返回:
    pd.DataFrame: 包含美国国债收益率曲线利率的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.us_tycr(
        date=date,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        limit=limit,
        offset=offset
    )
    
    # 如果输出结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_cctv_news(date: str, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
    """
    获取新闻联播文字稿数据。

    参数:
    date (str): 日期，格式为YYYYMMDD，例如：20181211。
    limit (int): 每次请求的数据量，默认值为1000。
    offset (int): 数据偏移量，默认值为0。

    返回:
    pd.DataFrame: 包含新闻联播文字稿数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.cctv_news(date=date)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_fund_sales_ratio(year: str = None, offset: int = 0, limit: int = 100) -> pd.DataFrame:
    """
    获取各渠道公募基金销售保有规模占比数据，年度更新。

    参数:
    year (str): 年度，可选参数。
    offset (int): 数据偏移量，默认值为0。
    limit (int): 单次获取数据的最大行数，默认值为100。

    返回:
    pd.DataFrame: 包含各渠道公募基金销售保有规模占比数据的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.fund_sales_ratio(year=year, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_fund_sales_vol_data(year: str = None, quarter: str = None, name: str = None, offset: int = 0, limit: int = 500) -> pd.DataFrame:
    """
    获取销售机构公募基金销售保有规模数据。

    参数:
    year (str): 年度，可选
    quarter (str): 季度，可选
    name (str): 机构名称，可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大行数，默认500

    返回:
    pd.DataFrame: 包含销售机构公募基金销售保有规模数据的DataFrame
    """
    pro = ts.pro_api()
    df = pro.fund_sales_vol(year=year, quarter=quarter, name=name, offset=offset, limit=limit)
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def fetch_major_news(src='', start_date='', end_date='', offset=0, limit=200, fields='title,pub_time,src'):
    """
    获取主要新闻资讯网站的长篇通讯信息。

    参数:
    src (str): 新闻来源（新华网、凤凰财经、同花顺、新浪财经、华尔街见闻、中证网），默认为空。
    start_date (str): 新闻发布开始时间，格式为 'YYYY-MM-DD HH:MM:SS'，默认为空。
    end_date (str): 新闻发布结束时间，格式为 'YYYY-MM-DD HH:MM:SS'，默认为空。
    offset (int): 数据偏移量，默认为0。
    limit (int): 单次提取数据的最大行数，默认为200。
    fields (str): 需要返回的字段，默认为 'title,pub_time,src'。

    返回:
    DataFrame: 包含新闻标题、发布时间、来源网站等信息的DataFrame。
    """
    pro = ts.pro_api()
    df = pro.major_news(src=src, start_date=start_date, end_date=end_date, offset=offset, limit=limit, fields=fields)
    
    # 如果结果包含ts_code列，去掉后缀形成新的code列
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def get_covid_stats(
    area_name: str = None,
    level: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取新冠状肺炎疫情感染人数统计数据

    参数:
    area_name (str): 地区名称，可选
    level (str): 级别：2-中国内地，3-省级，4-地区市级别，可选
    ann_date (str): 公告日期，可选
    start_date (str): 查询开始日期，可选
    end_date (str): 查询结束日期，可选
    offset (int): 数据偏移量，默认0
    limit (int): 单次获取数据的最大数量，默认2000

    返回:
    pd.DataFrame: 包含疫情统计数据的DataFrame
    """
    pro = ts.pro_api()
    
    df = pro.ncov_num(
        area_name=area_name,
        level=level,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
        df.drop(columns=['ts_code'], inplace=True)
    
    return df

@tsdata
def fetch_global_covid_data(
    country: str = None,
    province: str = None,
    publish_date: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 10000,
    offset: int = 0
) -> pd.DataFrame:
    """
    获取全球新冠疫情数据，包括国家和地区。

    参数:
    - country (str): 国家名称，可选
    - province (str): 省份简称（如北京、上海），可选
    - publish_date (datetime): 公布日期，可选
    - start_date (datetime): 开始日期（YYYYMMDD），可选
    - end_date (datetime): 结束日期（YYYYMMDD），可选
    - limit (int): 单次提取数据的最大行数，默认10000
    - offset (int): 数据偏移量，默认0

    返回:
    - pd.DataFrame: 包含全球新冠疫情数据的DataFrame
    """
    pro = ts.pro_api()
    
    # 获取数据
    df = pro.ncov_global(
        country=country,
        province=province,
        publish_date=publish_date,
        start_date=start_date,
        end_date=end_date,
        fields='area_id,publish_date,country,country_enname,province,province_short,province_enname,confirmed_num,confirmed_num_now,suspected_num,cured_num,dead_num,update_time'
    )
    
    # 如果输出结果包含ts_code列，去掉.XX后缀
    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]
    
    return df

@tsdata
def get_full_announcements(
    symbol: str = None,
    ann_date: str = None,
    start_date: str = None,
    end_date: str = None,
    offset: int = 0,
    limit: int = 2000
) -> pd.DataFrame:
    """
    获取全量公告数据，并提供PDF下载URL。

    参数:
    symbol (str, optional): 股票代码，支持A股、港股、美股、指数、期货。默认为None。
    ann_date (str, optional): 公告日期，格式为yyyymmdd。默认为None。
    start_date (str, optional): 公告开始日期，格式为yyyymmdd。默认为None。
    end_date (str, optional): 公告结束日期，格式为yyyymmdd。默认为None。
    offset (int, optional): 数据偏移量，默认为0。
    limit (int, optional): 单次获取数据的最大条数，默认为2000。

    返回:
    pd.DataFrame: 包含公告数据的DataFrame，包含以下列：
        - ann_date: 公告日期
        - ts_code: 股票代码
        - name: 股票名称
        - title: 标题
        - url: 原文下载链接
        - rec_time: 发布时间
        - code: 去除后缀的股票代码
    """
    from core.stock.ts_code_matcher import TsCodeMatcher
    matcher = TsCodeMatcher()
    ts_code = matcher[symbol] if symbol else None

    pro = ts.pro_api()
    df = pro.anns_d(
        ts_code=ts_code,
        ann_date=ann_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit
    )

    if 'ts_code' in df.columns:
        df['code'] = df['ts_code'].str.split('.').str[0]

    return df