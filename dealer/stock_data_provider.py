import asyncio
from threading import Lock,Event
import logging
import pickle
import re
from traceback import format_exc
from typing import Any, Callable, List, Dict, Tuple, Optional, Union ,Literal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import json
import akshare as ak
from core.llms._llm_api_client import LLMApiClient
from core.interpreter.data_summarizer import DataSummarizer
from core.interpreter.ast_code_runner import ASTCodeRunner
from core.utils.shared_cache import cache,lg_limiter,bg_limiter,cd_limiter,sn_limiter
from .baidu_news import BaiduFinanceAPI
from .index_finder import index_finder
from .stock_symbol_provider import StockSymbolProvider
from core.utils.query_executor import QueryExecutor
import ta
from tenacity import before_sleep_log, retry,retry_if_exception,stop_after_attempt,wait_fixed,wait_exponential
from core.utils.cache import CodeNameCache, load_pickle_cache, save_pickle_cache
from core.utils.path_akshare import patch_akshare
from core.utils.thread_safe import ThreadSafeCache
patch_akshare()
thread_safe_cache = ThreadSafeCache()


class StockDataProvider:
    def __init__(self,llm_client:LLMApiClient):
        self.llm_client = llm_client
        self.data_summarizer = DataSummarizer()
        self.code_runner = ASTCodeRunner()
        self.baidu_news_api = BaiduFinanceAPI()
        self.index_finder = index_finder
        self.stock_finder = StockSymbolProvider()
        
        self._code_name_list: Dict[str, str] = {}
        self._lock_code_name_list = Lock()
        self._cache_initialized_code_name_list = False
        self._initialization_event = Event()  # 用于线程协调
        self._initialization_in_progress = False  # 标记初始化状态

        self.previous_trading_date_cache = None
        self.previous_trading_date_cache_time = None
        self.latest_trading_date_cache = None
        self.latest_trading_date_cache_time = None
        self.cash_flow_cache = {}
        self.profit_cache = {}
        self.balance_sheet_cache = {}
        self.forecast_cache = {}
        self.report_cache = {}
        self.comment_cache = {}
        self.historical_data_cache = {}
        self.rebound_stock_pool_cache = {}  
        self.new_stock_pool_cache = {} 
        self.strong_stock_pool_cache = {}  
        self.previous_day_stock_pool_cache = {}
        self.institutional_holdings_cache = {}
        self.big_deal_cache = {}
        self.sector_cache = {}
        self.stock_lg_code_cache = {}
        self.stock_sector_cache = {}
        self.stock_sector_update_time = None
        self.sector_cache_file_path = './json/sector.pickle'
        self.task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logging.getLogger(__name__)

    def search_index_code(self,name:str)->str:
        """
        通过名称模糊查询指数代码。参数name:str,返回值str
        参数：
            name:str  用于搜索的名字
        返回值
            指数代码
        """
        return self.index_finder[name]

    def search_stock_code(self,name:str)->str:
        """
        通过名称模糊查询股票代码.参数name:str 返回值str
        参数
            name:str    用于搜索的名字
        返回值
            股票代码
        """
        return self.stock_finder[name]

    def get_stock_sector(self, symbol: str) -> str:
        """
        查询指定股票(symbol)的所属行业。使用ThreadSafeCache实现线程安全的缓存机制。
        
        参数:
            symbol (str): 股票代码
        
        返回:
            str: 股票所属行业，如果未找到则返回"未找到所属行业"
        """
        @thread_safe_cache.thread_safe_cached(cache_key="all_sector_data")
        def load_sector_data():
            """
            加载所有股票的行业数据。使用装饰器确保线程安全和缓存。
            """
            sector_data = {}
            
            try:
                # 先尝试从文件加载缓存
                if os.path.exists(self.sector_cache_file_path):
                    with open(self.sector_cache_file_path, 'rb') as f:
                        sector_data = pickle.load(f)
                    self.logger.info(f"Loaded {len(sector_data)} items from sector cache file.")
                    return sector_data
                    
            except Exception as e:
                self.logger.warning(f"Failed to load sector cache from file: {str(e)}")
            
            try:
                # 如果文件加载失败，从API获取数据
                industry_df = ak.stock_board_industry_name_em()
                
                for _, industry in industry_df.iterrows():
                    industry_name = industry['板块名称']
                    try:
                        cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
                        for code in cons_df['代码'].values:
                            code_str = str(code)
                            if code_str in sector_data:
                                sector_data[code_str] += f", {industry_name}"
                            else:
                                sector_data[code_str] = industry_name
                    except Exception as e:
                        self.logger.warning(f"Error fetching constituents for industry {industry_name}: {str(e)}")
                
                # 保存到文件以便下次使用
                try:
                    with open(self.sector_cache_file_path, 'wb') as f:
                        pickle.dump(sector_data, f)
                    self.logger.info(f"Saved {len(sector_data)} items to sector cache file.")
                except Exception as e:
                    self.logger.warning(f"Failed to save sector cache to file: {str(e)}")
                
                return sector_data
                
            except Exception as e:
                self.logger.error(f"Error fetching sector data: {str(e)}")
                return {}
        
        try:
            # 获取所有行业数据的缓存
            sector_data = load_sector_data()
            
            # 返回指定股票的行业信息
            return sector_data.get(symbol, "未找到所属行业")
            
        except Exception as e:
            self.logger.error(f"Error getting sector for symbol {symbol}: {str(e)}")
            return "未找到所属行业"

    def clear_sector_cache(self):
        """
        清除行业数据缓存。
        """
        try:
            # 使用装饰器的clear_cache方法清除缓存
            self.get_stock_sector.__wrapped__.clear_cache()  # type: ignore
            self.logger.info("Sector cache cleared successfully.")
        except Exception as e:
            self.logger.error(f"Error clearing sector cache: {str(e)}")

    def _load_sector_cache(self):
        """从文件加载缓存"""
        try:
            if os.path.exists(self.sector_cache_file_path):
                with open(self.sector_cache_file_path, 'rb') as f:
                    self.sector_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.sector_cache)} items from sector cache.")
            else:
                self.logger.info("Sector cache file does not exist. Will create a new one.")
        except Exception as e:
            self.logger.error(f"Error loading sector cache: {str(e)}")
            self.sector_cache = {}

    def _update_sector_cache(self):
        """更新整个行业缓存"""
        if self.stock_sector_update_time and datetime.now() - self.stock_sector_update_time < timedelta(days=1):
            self.logger.info("Sector cache is up to date. Skipping update.")
            return
        try:
            # 获取所有行业板块数据
            industry_df = ak.stock_board_industry_name_em()
            
            new_cache = {}
            for _, industry in industry_df.iterrows():
                industry_code = industry['板块代码']
                industry_name = industry['板块名称']

                try:
                    cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
                    for code in cons_df['代码'].values:
                        code_str = str(code)
                        if code_str in new_cache:
                            new_cache[code_str] += f", {industry_name}"
                        else:
                            new_cache[code_str] = industry_name
                except Exception as e:
                    self.logger.warning(f"Error fetching constituents for industry {industry_name}: {str(e)}")

            # 更新缓存
            self.sector_cache = new_cache
            self._save_sector_cache()
            self.stock_sector_update_time = datetime.now()
            self.logger.info(f"Updated sector cache with {len(self.sector_cache)} items.")
        except Exception as e:
            self.logger.error(f"Error updating sector cache: {str(e)}")

    def _save_sector_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.sector_cache_file_path, 'wb') as f:
                pickle.dump(self.sector_cache, f)
            self.logger.info(f"Saved {len(self.sector_cache)} items to sector cache file.")
        except Exception as e:
            self.logger.error(f"Error saving sector cache: {str(e)}")

    def get_macro_economic_indicators(self) -> str:

        if hasattr(self, 'macro_economic_indicators'):
            return self.macro_economic_indicators

        result = []
        
        # 中国宏观杠杆率
        try:
            df = ak.macro_cnbs()
            latest = df.iloc[-1]
            result.append(f"中国宏观杠杆率 (截至 {latest['年份']}):\n"
                        f"居民部门: {latest['居民部门']}%, 非金融企业部门: {latest['非金融企业部门']}%, "
                        f"政府部门: {latest['政府部门']}%, 实体经济部门: {latest['实体经济部门']}%")
        except Exception as e:
            result.append(f"获取中国宏观杠杆率数据失败: {str(e)}")

        # 企业商品价格指数
        try:
            df = ak.macro_china_qyspjg()
            latest = df.iloc[-1]
            result.append(f"企业商品价格指数 ({latest['月份']}):\n"
                        f"总指数: {latest['总指数-指数值']}, 同比增长: {latest['总指数-同比增长']}%, "
                        f"环比增长: {latest['总指数-环比增长']}%")
        except Exception as e:
            result.append(f"获取企业商品价格指数数据失败: {str(e)}")

        # 外商直接投资数据
        try:
            df = ak.macro_china_fdi()
            latest = df.iloc[-1]
            result.append(f"外商直接投资 ({latest['月份']}):\n"
                        f"当月: {latest['当月']}美元, 同比增长: {latest['当月-同比增长']}%, "
                        f"累计: {latest['累计']}美元, 同比增长: {latest['累计-同比增长']}%")
        except Exception as e:
            result.append(f"获取外商直接投资数据失败: {str(e)}")

        # LPR数据
        try:
            df = ak.macro_china_lpr()
            latest = df.iloc[-1]
            result.append(f"LPR利率 ({latest['TRADE_DATE']}):\n"
                        f"1年期: {latest['LPR1Y']}%, 5年期: {latest['LPR5Y']}%")
        except Exception as e:
            result.append(f"获取LPR数据失败: {str(e)}")

        # 城镇调查失业率
        try:
            df = ak.macro_china_urban_unemployment()
            latest_month = df['date'].max()
            latest = df[df['date'] == latest_month]
            result.append(f"城镇调查失业率 ({latest_month}):")
            for _, row in latest.iterrows():
                result.append(f"{row['item']}: {row['value']}%")
        except Exception as e:
            result.append(f"获取城镇调查失业率数据失败: {str(e)}")

        # 社会融资规模增量统计
        try:
            df = ak.macro_china_shrzgm()
            latest = df.iloc[0]
            result.append(f"社会融资规模增量 ({latest['月份']}):\n"
                        f"当月: {latest['社会融资规模增量']}亿元, "
                        f"人民币贷款: {latest['其中-人民币贷款']}亿元")
        except Exception as e:
            result.append(f"获取社会融资规模增量数据失败: {str(e)}")

        # GDP年率
        try:
            df = ak.macro_china_gdp_yearly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"GDP年率 ({latest['日期']}):\n"
                        f"同比增长: {latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取GDP年率数据失败: {str(e)}")

        # CPI年率
        try:
            df = ak.macro_china_cpi_yearly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"CPI年率 ({latest['日期']}):\n"
                        f"同比增长: {latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取CPI年率数据失败: {str(e)}")

        # CPI月率
        try:
            df = ak.macro_china_cpi_monthly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"CPI月率 ({latest['日期']}):\n"
                        f"环比增长: {latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取CPI月率数据失败: {str(e)}")

        # PPI年率
        try:
            df = ak.macro_china_ppi_yearly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"PPI年率 ({latest['日期']}):\n"
                        f"同比增长: {latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取PPI年率数据失败: {str(e)}")

        # 进出口年率
        try:
            exports_df = ak.macro_china_exports_yoy()
            imports_df = ak.macro_china_imports_yoy()
            latest_exports = exports_df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            latest_imports = imports_df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"进出口年率:\n"
                        f"出口 ({latest_exports['日期']}): {latest_exports['今值']}%, "
                        f"进口 ({latest_imports['日期']}): {latest_imports['今值']}%")
        except Exception as e:
            result.append(f"获取进出口年率数据失败: {str(e)}")

        # 贸易帐
        try:
            df = ak.macro_china_trade_balance()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"贸易帐 ({latest['日期']}):\n"
                        f"{latest['今值']}亿美元, 预期: {latest['预测值']}亿美元")
        except Exception as e:
            result.append(f"获取贸易帐数据失败: {str(e)}")

        # 工业增加值增长
        try:
            df = ak.macro_china_industrial_production_yoy()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"工业增加值增长 ({latest['日期']}):\n"
                        f"同比增长: {latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取工业增加值增长数据失败: {str(e)}")

        # PMI数据
        try:
            pmi_df = ak.macro_china_pmi_yearly()
            cx_pmi_df = ak.macro_china_cx_pmi_yearly()
            cx_services_pmi_df = ak.macro_china_cx_services_pmi_yearly()
            
            latest_pmi = pmi_df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            latest_cx_pmi = cx_pmi_df.iloc[-2]
            latest_cx_services_pmi = cx_services_pmi_df.iloc[-2]
            
            result.append(f"PMI数据:\n"
                        f"官方制造业PMI ({latest_pmi['日期']}): {latest_pmi['今值']}\n"
                        f"财新制造业PMI ({latest_cx_pmi['日期']}): {latest_cx_pmi['今值']}\n"
                        f"财新服务业PMI ({latest_cx_services_pmi['日期']}): {latest_cx_services_pmi['今值']}")
        except Exception as e:
            result.append(f"获取PMI数据失败: {str(e)}")

        # 外汇储备
        try:
            df = ak.macro_china_fx_reserves_yearly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"外汇储备 ({latest['日期']}):\n"
                        f"{latest['今值']}亿美元, 预期: {latest['预测值']}亿美元")
        except Exception as e:
            result.append(f"获取外汇储备数据失败: {str(e)}")

        # M2货币供应年率
        try:
            df = ak.macro_china_m2_yearly()
            latest = df.iloc[-2]  # 使用倒数第二行，因为最后一行可能是NaN
            result.append(f"M2货币供应年率 ({latest['日期']}):\n"
                        f"{latest['今值']}%, 预期: {latest['预测值']}%")
        except Exception as e:
            result.append(f"获取M2货币供应年率数据失败: {str(e)}")

        self.macro_economic_indicators = "\n\n".join(result)
        return self.macro_economic_indicators

    def get_main_competitors(self, symbol: str) -> str:
        """
        获取主要竞争对手的股票代码和相关信息。

        参数：
            symbol: str  股票代码

        返回值：
            str: 格式化的竞争对手信息字符串，如果没有找到竞争对手则返回相应消息
        """
        competitors = self.baidu_news_api.get_stock_recommendations(symbol)
        
        if not competitors:
            return "未找到该股票的主要竞争对手信息。"
        
        formatted_output = "主要竞争对手信息：\n"
        for comp in competitors:
            price_status = "↑" if comp['price']['status'] == 'up' else "↓"
            ratio_status = "↑" if comp['ratio']['status'] == 'up' else "↓"
            
            formatted_output += (
                f"代码: {comp['code']} | 名称: {comp['name']} | 市场: {comp['market']} | "
                f"交易所: {comp['exchange']}\n"
                f"价格: {comp['price']['value']} {price_status} | "
                f"涨跌幅: {comp['ratio']['value']} {ratio_status}\n"
                f"------------------------\n"
            )
        
        return formatted_output.strip()

    def get_global_economic_indicators(self) -> str:
        if hasattr(self, 'global_economic_indicators'):
            return self.global_economic_indicators

        result = []
        
        # 美国 GDP 月率
        df = ak.macro_usa_gdp_monthly()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"美国GDP月率: {latest['今值']}% (日期: {latest['日期']}, 前值: {latest['前值']}%)")
        
        # 美国失业率
        df = ak.macro_usa_unemployment_rate()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"美国失业率: {latest['今值']}% (日期: {latest['日期']}, 前值: {latest['前值']}%)")
        
        # 美国CPI月率
        df = ak.macro_usa_cpi_monthly()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"美国CPI月率: {latest['今值']}% (日期: {latest['日期']}, 前值: {latest['前值']}%)")
        
        # 欧元区GDP季率
        df = ak.macro_euro_gdp_yoy()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"欧元区GDP季率: {latest['今值']}% (日期: {latest['日期']}, 前值: {latest['前值']}%)")
        
        # 欧元区失业率
        df = ak.macro_euro_unemployment_rate_mom()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"欧元区失业率: {latest['今值']}% (日期: {latest['日期']}, 前值: {latest['前值']}%)")
        
        # 英国GDP年率
        df = ak.macro_uk_gdp_yearly()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"英国GDP年率: {latest['现值']}% (时间: {latest['时间']}, 前值: {latest['前值']}%)")
        
        # 英国失业率
        df = ak.macro_uk_unemployment_rate()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"英国失业率: {latest['现值']}% (时间: {latest['时间']}, 前值: {latest['前值']}%)")
        
        # 中国GDP年率
        df = ak.macro_china_gdp_yearly()
        if not df.empty:
            latest = df.iloc[-1]
            result.append(f"中国GDP年率: {latest['今值']}% (日期: {latest['统计时间']}, 前值: {latest['前值']}%)")
        
        # 中国失业率
        df = ak.macro_china_urban_unemployment()
        if not df.empty:
            # 筛选出全国城镇调查失业率的最新数据
            latest = df[df['item'] == '全国城镇调查失业率'].iloc[0]
            result.append(f"中国城镇调查失业率: {latest['value']}% (日期: {latest['date']})")
    
        
        self.global_economic_indicators =  "\n".join(result)
        return self.global_economic_indicators

    def get_esg_score(self, symbol: str) -> str:
        """
        获取ESG评分
        
        :param symbol: 股票代码（不包含后缀）
        :return: ESG评分或未找到数据的提示
        """
        if not hasattr(self, 'esg_rate_cache') or self.esg_rate_cache is None:
            self._load_esg_data()
        
        if symbol in self.esg_rate_cache:
            return self.esg_rate_cache[symbol]
        else:
            return f"No ESG data found for {symbol}"

    def _load_esg_data(self):
        df = ak.stock_esg_hz_sina()
        # 移除股票代码中的后缀（如.SZ或.SH）
        df['股票代码'] = df['股票代码'].str.split('.').str[0]
        self.esg_rate_cache = dict(zip(df['股票代码'], df['ESG评分'].astype(str)))

    def get_esg_score_sina(self, symbol: str) -> str:
        # 检查缓存是否存在
        if hasattr(self, 'esg_rate_cache'):
            if symbol in self.esg_rate_cache:
                return self.esg_rate_cache[symbol]
            else:
                return f"No ESG data found for {symbol}"
        
        # 如果缓存不存在或symbol不在缓存中，则重新获取数据
        df = ak.stock_esg_rate_sina()
        
        # 筛选最新季度
        df['评级季度'] = pd.to_datetime(df['评级季度'], format='%YQ%q')
        latest_quarter = df['评级季度'].max()
        df = df[df['评级季度'] == latest_quarter]
        
        # 筛选交易市场为cn
        df = df[df['交易市场'] == 'cn']
        
        # 处理成分股代码
        df['symbol'] = df['成分股代码'].str.replace(r'^[A-Z]+', '', regex=True)
        
        # 创建结果字典
        result_dict = {}
        
        for sym in df['symbol'].unique():
            sym_data = df[df['symbol'] == sym]
            ratings = []
            for _, row in sym_data.iterrows():
                rating = f"{row['评级机构']}: {row['评级']}"
                if pd.notna(row['标识']):
                    rating += f" ({row['标识']})"
                ratings.append(rating)
            
            result_str = f"ESG Ratings for {sym} (as of {latest_quarter.strftime('%Y Q%q')}):\n"
            result_str += "\n".join(ratings)
            result_dict[sym] = result_str
        
        # 保存缓存
        self.esg_rate_cache = result_dict
        
        # 返回请求的symbol的数据
        return result_dict.get(symbol, f"No ESG data found for {symbol}")

    def get_cctv_news(self, days=30) -> List[dict]:
        """
        获取最近指定天数的CCTV新闻联播内容，参数需要获取的天数 days=30 ，返回列表，包含date、title、content

        参数:
        days (int): 要获取的天数，默认为30天

        返回:
        List[dict]: 包含新闻数据的字典列表，每个字典包含date（日期）、title（标题）和content（内容）
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news_list = []
        current_date = end_date
        
        while current_date >= start_date:
            date_str = current_date.strftime("%Y%m%d")
            try:
                news_df = ak.news_cctv(date=date_str)
                if not news_df.empty:
                    for _, row in news_df.iterrows():
                        news_item = {
                            "date": row['date'],
                            "title": row['title'],
                            "content": row['content']
                        }
                        news_list.append(news_item)
                else:
                    print(f"No news data available for {date_str}")
            except Exception as e:
                print(f"Error fetching news for {date_str}: {str(e)}")
            
            current_date -= timedelta(days=1)
        
        return news_list

    def _fetch_trading_dates(self):
        # 获取当前时间
        now = datetime.now()

        # 定义今天的日期
        today_str = now.strftime("%Y%m%d")
        start_date_str = (now - timedelta(days=10)).strftime("%Y%m%d")

        # 获取最近10天的交易数据，假设使用上证指数（000001.SH）
        stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=start_date_str, end_date=today_str, adjust="")

        # 提取交易日期
        trading_dates = stock_data['日期'].apply(lambda x: x.strftime( "%Y%m%d")).tolist()
        trading_dates.sort()
        return trading_dates

    def get_previous_trading_date(self) -> str:
        """
        获取最近一个交易日，不包含今天的日期,返回str 格式：YYYYMMDD
        """
        now = datetime.now()
        
        # 检查缓存是否有效
        if self.previous_trading_date_cache and self.previous_trading_date_cache_time.date() == now.date():
            return self.previous_trading_date_cache

        trading_dates = self._fetch_trading_dates()  # 假设这个函数返回字符串列表

        # 如果最近交易日期是今天，则返回上一个交易日期
        today_str = now.strftime("%Y%m%d")
        if trading_dates[-1] == today_str:
            previous_trading_date = trading_dates[-2]
        else:
            previous_trading_date = trading_dates[-1]

        self.previous_trading_date_cache = previous_trading_date
        self.previous_trading_date_cache_time = now

        return self.previous_trading_date_cache
    
    def get_latest_trading_date(self) -> str:
        """
        获取最近一个交易日。返回str 格式：YYYYMMDD
        如果当前时间是9:30之后，则最近包含今天，否则不包含。
        
        处理逻辑：
        1. 检查缓存是否有效
        2. 获取交易日期列表
        3. 根据当前时间判断是否包含今天
        4. 返回合适的交易日期
        
        返回:
            str: 格式为YYYYMMDD的日期字符串
        """
        now = datetime.now()
        cache_valid = False

        # 检查缓存是否有效
        if self.latest_trading_date_cache and self.latest_trading_date_cache_time.date() == now.date():
            if now.time() < datetime.strptime('09:30', '%H:%M').time():
                if self.latest_trading_date_cache_time.time() < datetime.strptime('09:30', '%H:%M').time():
                    cache_valid = True
            else:
                cache_valid = True

        if cache_valid:
            return self.latest_trading_date_cache

        # 获取交易日期列表
        trading_dates = self._fetch_trading_dates()
        
        # 判断是否包含今天
        include_today = now.time() >= datetime.strptime('09:30', '%H:%M').time()
        today_str = now.strftime("%Y%m%d")

        # 获取最后一个交易日（确保是字符串格式）
        latest_date = trading_dates[-1]
        if hasattr(latest_date, 'strftime'):  # 如果是datetime对象
            latest_date = latest_date.strftime("%Y%m%d")

        # 判断和处理逻辑
        if include_today and latest_date == today_str:
            latest_trading_date = latest_date
        else:
            # 如果最后一个日期是今天，取倒数第二个，否则取最后一个
            if latest_date == today_str:
                second_latest_date = trading_dates[-2]
                if hasattr(second_latest_date, 'strftime'):
                    latest_trading_date = second_latest_date.strftime("%Y%m%d")
                else:
                    latest_trading_date = second_latest_date
            else:
                latest_trading_date = latest_date

        # 更新缓存
        self.latest_trading_date_cache = latest_trading_date
        self.latest_trading_date_cache_time = now
        return self.latest_trading_date_cache

    def stock_fund_flow_big_deal_ak(self) -> pd.DataFrame:
        """
        同花顺-数据中心-资金流向-大单追踪
        https://data.10jqka.com.cn/funds/ddzz
        :return: 大单追踪
        :rtype: pandas.DataFrame
        """
        import py_mini_racer
        import requests
        from bs4 import BeautifulSoup
        from io import StringIO
        from akshare.utils.tqdm import get_tqdm
        js_code = py_mini_racer.MiniRacer()

        js_content = self._get_file_content_ths("ths.js")
        js_code.eval(js_content)
        v_code = js_code.call("v")
        headers = {
            "Accept": "text/html, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "hexin-v": v_code,
            "Host": "data.10jqka.com.cn",
            "Pragma": "no-cache",
            "Referer": "http://data.10jqka.com.cn/funds/hyzjl/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.85 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }
        url = "http://data.10jqka.com.cn/funds/ddzz/order/desc/ajax/1/free/1/"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, features="lxml")
        raw_page = soup.find(name="span", attrs={"class": "page_info"}).text
        page_num = raw_page.split("/")[1]
        url = "http://data.10jqka.com.cn/funds/ddzz/order/asc/page/{}/ajax/1/free/1/"
        big_df = pd.DataFrame()
        tqdm = get_tqdm()
        for page in tqdm(range(1, int(page_num) + 1), leave=False):
            js_code = py_mini_racer.MiniRacer()
            js_content = self._get_file_content_ths("ths.js")
            js_code.eval(js_content)
            v_code = js_code.call("v")
            headers = {
                "Accept": "text/html, */*; q=0.01",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "hexin-v": v_code,
                "Host": "data.10jqka.com.cn",
                "Pragma": "no-cache",
                "Referer": "http://data.10jqka.com.cn/funds/hyzjl/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.85 Safari/537.36",
                "X-Requested-With": "XMLHttpRequest",
            }
            r = requests.get(url.format(page), headers=headers)
            temp_df = pd.read_html(StringIO(r.text))[0]
            big_df = pd.concat(objs=[big_df, temp_df], ignore_index=True)

        big_df.columns = [
            "成交时间",
            "股票代码",
            "股票简称",
            "成交价格",
            "成交量",
            "成交额",
            "大单性质",
            "涨跌幅",
            "涨跌额",
            "详细",
        ]
        del big_df["详细"]
        return big_df

    @cache.cache(ttl=30*60,limiter=bg_limiter)
    def stock_fund_flow_big_deal(self):
        return self.stock_fund_flow_big_deal_ak()

    def get_stock_big_deal(self, symbol: str) -> str:
        """
        获取指定股票的大单追踪数据，使用线程安全的缓存机制。

        缓存策略：
        - 交易时间内（9:30 - 15:00）：缓存60秒
        - 非交易时间：缓存永久有效（至下一次刷新）
        
        参数:
            symbol (str): 股票代码

        返回:
            str: 大单数据的格式化字符串，包含：
                - 成交时间、股票代码、股票简称
                - 成交价格、成交量、成交额
                - 大单性质、涨跌幅、涨跌额
                如果没有数据则返回"暂时没有数据"
                如果获取失败则返回错误信息
        """
        from time import time
        def _format_big_deal_row(row) -> str:
            """格式化单行大单数据"""
            try:
                return (
                    f"成交时间: {row['成交时间']}, 股票代码: {row['股票代码']}, "
                    f"股票简称: {row['股票简称']}, 成交价格: {row['成交价格']}, "
                    f"成交量: {row['成交量']}股, 成交额: {row['成交额']}万元, "
                    f"大单性质: {row['大单性质']}, 涨跌幅: {row['涨跌幅']}, "
                    f"涨跌额: {row['涨跌额']}"
                )
            except Exception as e:
                self.logger.error(f"Error formatting big deal row: {str(e)}")
                raise RuntimeError(f"数据格式化失败: {str(e)}")

        # 检查市场时间并设置缓存时间
        now = datetime.now()
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
        is_market_hours = market_open_time <= now <= market_close_time
        
        # 生成缓存键，在交易时间内加入分钟信息以区分不同时间的缓存
        if is_market_hours:
            cache_key = f"stock_big_deal_{now.strftime('%Y%m%d_%H%M')}"
            cache_timeout = 60  # 交易时间内缓存60秒
        else:
            cache_key = f"stock_big_deal_{now.strftime('%Y%m%d')}"
            cache_timeout = None  # 非交易时间缓存永久有效
            
        @thread_safe_cache.thread_safe_cached(cache_key, timeout=cache_timeout)
        def fetch_all_big_deals():
            """获取并缓存所有股票的大单数据"""
            try:
                self.logger.info(f"Fetching all big deal data for caching at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 使用重试装饰器的函数来获取数据
                df = self.stock_fund_flow_big_deal()
                
                if df is None or df.empty:
                    self.logger.warning("Empty data received from stock_fund_flow_big_deal")
                    raise RuntimeError("未获取到任何大单数据")
                
                result = {}
                for _, row in df.iterrows():
                    try:
                        code = str(row['股票代码'])
                        result[code] = _format_big_deal_row(row)
                    except Exception as row_e:
                        self.logger.warning(f"Error processing row for stock {row.get('股票代码', 'unknown')}: {str(row_e)} {format_exc()}")
                        continue
                
                if not result:
                    self.logger.warning("No big deal data processed successfully")
                    raise RuntimeError("未获取到任何有效的大单数据")
                    
                self.logger.debug(f"Successfully cached big deal data for {len(result)} stocks")
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to fetch and cache big deal data: {str(e)} {format_exc()}")
                raise RuntimeError(f"获取大单数据失败: {str(e)}")
        
        max_retries = 3
        retry_delay = 5  # 重试延迟秒数
        waiting_time = 120  # 等待其他线程初始化的最大时间
        
        for retry_count in range(max_retries):
            try:
                self.logger.debug(f"Attempting to get big deal data for {symbol} from cache (Market hours: {is_market_hours})")
                all_deals = fetch_all_big_deals()
                return all_deals.get(symbol, "暂时没有数据")
                
            except TimeoutError as te:
                self.logger.error(f"Cache initialization timeout for {symbol} (attempt {retry_count + 1}/{max_retries}): {str(te)}")
                # 只有在最后一次重试时才清除缓存
                if retry_count == max_retries - 1:
                    try:
                        fetch_all_big_deals.clear_cache()
                    except Exception as clear_e:
                        self.logger.error(f"Error clearing cache: {str(clear_e)}")
                    return f"获取数据超时: {str(te)}"
                time.sleep(retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error accessing big deal data for {symbol}: {str(e)}")
                # 清除缓存并立即返回特定错误
                if "未获取到任何大单数据" in str(e):
                    try:
                        fetch_all_big_deals.clear_cache()
                    except Exception as clear_e:
                        self.logger.error(f"Error clearing cache: {str(clear_e)}")
                    return "暂时没有大单数据"
                
                # 对于其他错误，在最后一次重试时清除缓存
                if retry_count == max_retries - 1:
                    try:
                        fetch_all_big_deals.clear_cache()
                    except Exception as clear_e:
                        self.logger.error(f"Error clearing cache: {str(clear_e)}")
                return f"获取数据失败: {str(e)}"
                
        return "获取数据失败: 达到最大重试次数"

    def get_stock_announcements_recent_days(self, days: int = 5) -> Dict[str, List[str]]:
        """
        获取最近n天内的所有股票公告信息。

        参数:
            days (int): 要获取的天数，默认为5天。

        返回:
            Dict[str, List[str]]: 键为股票代码，值为该股票的公告列表。
            如果某个日期没有数据或获取失败，会继续处理其他日期的数据。
        """
        cache_key = f"stock_announcements_{days}days"

        @thread_safe_cache.thread_safe_cached(cache_key)
        def fetch_recent_announcements() -> Dict[str, List[str]]:
            result = {}
            today = datetime.now()
            fetch_success = False  # 标记是否成功获取了任何数据
            
            for i in range(days):
                date = (today - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    daily_announcements = self.get_stock_announcements(symbols=[], date=date)

                    if daily_announcements is None:
                        self.logger.debug(f"{date}没有公告数据")
                        continue
                    
                    # 验证返回的数据
                    if not daily_announcements:
                        self.logger.debug(f"{date}没有公告数据")
                        continue
                    
                    # 确保返回的是字典类型
                    if not isinstance(daily_announcements, dict):
                        self.logger.warning(f"{date}返回的数据格式不正确: {type(daily_announcements)}")
                        continue
                    
                    # 合并公告数据
                    for symbol, announcements in daily_announcements.items():
                        if not announcements:  # 跳过空的公告列表
                            continue
                            
                        if symbol not in result:
                            result[symbol] = []
                        # 为每条公告添加日期前缀
                        dated_announcements = [f"[{date}] {announcement}" 
                                            for announcement in announcements 
                                            if announcement]  # 只添加非空公告
                        if dated_announcements:  # 只在有实际公告时添加
                            result[symbol].extend(dated_announcements)
                            fetch_success = True
                    
                except Exception as e:
                    self.logger.error(f"获取{date}的公告数据时出错: {str(e)}")
                    continue
            
            if not fetch_success:
                self.logger.warning(f"在过去{days}天内没有获取到任何有效的公告数据")
                
            return result

        try:
            announcements = fetch_recent_announcements()
            # 清理空的股票条目
            return {k: v for k, v in announcements.items() if v}
        except Exception as e:
            self.logger.error(f"获取最近{days}天公告数据时出错: {str(e)}")
            return {}

    def get_one_stock_announcements(self, symbol: str, days: int = 5) -> List[str]:
        """
        获取指定股票最近的公告信息。

        参数:
            symbol (str): 股票代码
            days (int): 要查询的天数，默认5天

        返回:
            List[str]: 该股票的公告列表，按时间倒序排列。
            如果没有数据返回空列表。
        """
        try:
            # 尝试从缓存中获取所有公告
            all_announcements = self.get_stock_announcements_recent_days(days)
            
            # 检查是否获取到数据
            if not all_announcements:
                #self.logger.info("未获取到任何公告数据")
                return []
                
            # 提取指定股票的公告
            announcements = all_announcements.get(symbol, [])
            if not announcements:
                self.logger.info(f"未找到股票{symbol}的公告信息")
                return []
                
            # 按日期倒序排序
            try:
                sorted_announcements = sorted(
                    [a for a in announcements if a and '[' in a],  # 只处理有效的公告
                    key=lambda x: x.split(']')[0].strip('['),  # 提取日期部分进行排序
                    reverse=True
                )
                return sorted_announcements
            except Exception as e:
                self.logger.error(f"排序公告数据时出错: {str(e)}")
                return announcements  # 如果排序失败，返回未排序的列表
                
        except Exception as e:
            self.logger.error(f"获取股票{symbol}的公告数据时出错: {str(e)}")
            return []

    def get_rebound_stock_pool(self, date: str = None) -> dict:
        """
        获取炸板股池数据并返回格式化结果。返回dict[symbol,str]
        
        参数:
            date (str): 交易日期，格式为 'yyyymmdd'。如果未提供，则获取最近一个交易日的数据。
        
        返回:
            dict: 键为股票代码，值为该股票的相关信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_previous_trading_date()
        
        # 检查缓存
        if date in self.rebound_stock_pool_cache:
            return self.rebound_stock_pool_cache[date]

        # 获取数据
        stock_pool_df = ak.stock_zt_pool_zbgc_em(date=date)

        # 处理数据
        result = {}
        for _, row in stock_pool_df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"最新价: {row['最新价']}, "
                f"涨停价: {row['涨停价']}, "
                f"成交额: {row['成交额']}元, "
                f"流通市值: {row['流通市值']}亿, "
                f"总市值: {row['总市值']}亿, "
                f"换手率: {row['换手率']}%, "
                f"涨速: {row['涨速']}, "
                f"首次封板时间: {row['首次封板时间']}, "
                f"炸板次数: {row['炸板次数']}, "
                f"涨停统计: {row['涨停统计']}, "
                f"振幅: {row['振幅']}, "
                f"所属行业: {row['所属行业']}"
            )
            result[row['代码']] = stock_info

        # 缓存结果
        self.rebound_stock_pool_cache[date] = result

        return result

    def get_new_stock_pool(self, date: str = None) -> dict:
        """
        获取次新股池数据并返回格式化结果。返回dict[symbol,str]
        
        参数:
            date (str): 交易日期，格式为 'yyyymmdd'。如果未提供，则获取最近一个交易日的数据。
        
        返回:
            dict: 键为股票代码，值为该股票的相关信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_previous_trading_date()
        
        # 检查缓存
        if date in self.new_stock_pool_cache:
            return self.new_stock_pool_cache[date]

        # 获取数据
        new_stock_pool_df = ak.stock_zt_pool_sub_new_em(date=date)

        # 处理数据
        result = {}
        for _, row in new_stock_pool_df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"最新价: {row['最新价']}, "
                f"涨停价: {row['涨停价']}, "
                f"成交额: {row['成交额']}元, "
                f"流通市值: {row['流通市值']}亿, "
                f"总市值: {row['总市值']}亿, "
                f"转手率: {row['转手率']}%, "
                f"开板几日: {row['开板几日']}, "
                f"开板日期: {row['开板日期']}, "
                f"上市日期: {row['上市日期']}, "
                f"是否新高: {row['是否新高']}, "
                f"涨停统计: {row['涨停统计']}, "
                f"所属行业: {row['所属行业']}"
            )
            result[row['代码']] = stock_info

        # 缓存结果
        self.new_stock_pool_cache[date] = result

        return result

    def get_strong_stock_pool(self, date: str = None) -> dict:
        """
        获取强势股池数据并返回格式化结果。返回dict[symbol,str]
        
        参数:
            date (str): 交易日期，格式为 'yyyymmdd'。如果未提供，则获取最近一个交易日的数据。
        
        返回:
            dict: 键为股票代码，值为该股票的相关信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_previous_trading_date()
        
        # 检查缓存
        if date in self.strong_stock_pool_cache:
            return self.strong_stock_pool_cache[date]

        # 获取数据
        strong_stock_pool_df = ak.stock_zt_pool_strong_em(date=date)

        # 处理数据
        result = {}
        for _, row in strong_stock_pool_df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"最新价: {row['最新价']}, "
                f"涨停价: {row['涨停价']}, "
                f"成交额: {row['成交额']}元, "
                f"流通市值: {row['流通市值']}亿, "
                f"总市值: {row['总市值']}亿, "
                f"换手率: {row['换手率']}%, "
                f"涨速: {row['涨速']}%, "
                f"是否新高: {row['是否新高']}, "
                f"量比: {row['量比']}, "
                f"涨停统计: {row['涨停统计']}, "
                f"入选理由: {row['入选理由']}, "
                f"所属行业: {row['所属行业']}"
            )
            result[row['代码']] = stock_info

        # 缓存结果
        self.strong_stock_pool_cache[date] = result

        return result

    def get_previous_day_stock_pool(self, date: str = None) -> dict:
        """
        获取昨日涨停股池数据并返回格式化结果。返回dict[symbol,str]
        
        参数:
            date (str): 交易日期，格式为 'yyyymmdd'。如果未提供，则获取最近一个交易日的数据。
        
        返回:
            dict: 键为股票代码，值为该股票的相关信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_previous_trading_date()
        
        # 检查缓存
        if date in self.previous_day_stock_pool_cache:
            return self.previous_day_stock_pool_cache[date]

        # 获取数据
        previous_day_stock_pool_df = ak.stock_zt_pool_previous_em(date)

        # 处理数据
        result = {}
        for _, row in previous_day_stock_pool_df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"最新价: {row['最新价']}, "
                f"涨停价: {row['涨停价']}, "
                f"成交额: {row['成交额']}元, "
                f"流通市值: {row['流通市值']}亿, "
                f"总市值: {row['总市值']}亿, "
                f"换手率: {row['换手率']}%, "
                f"涨速: {row['涨速']}%, "
                f"振幅: {row['振幅']}%, "
                f"昨日封板时间: {row['昨日封板时间']}, "
                f"昨日连板数: {row['昨日连板数']}, "
                f"涨停统计: {row['涨停统计']}, "
                f"所属行业: {row['所属行业']}"
            )
            result[row['代码']] = stock_info

        # 缓存结果
        self.previous_day_stock_pool_cache[date] = result

        return result

    def get_market_anomaly(self, indicator: Literal['火箭发射', '快速反弹', '大笔买入', '封涨停板', '打开跌停板', '有大买盘', '竞价上涨', '高开5日线', '向上缺口', '60日新高', '60日大幅上涨', '加速下跌', '高台跳水', '大笔卖出', '封跌停板', '打开涨停板', '有大卖盘', '竞价下跌', '低开5日线', '向下缺口', '60日新低', '60日大幅下跌'] = '大笔买入') -> dict:
        """
        获取指定类型的盘口异动信息，并返回格式化结果。indicator:str="大笔买入" 返回dict[symbol,str]
        
        参数:
            indicator (str): 盘口异动的类型，可以从以下选项中选择:
                - '火箭发射', '快速反弹', '大笔买入', '封涨停板', '打开跌停板', '有大买盘', 
                - '竞价上涨', '高开5日线', '向上缺口', '60日新高', '60日大幅上涨', 
                - '加速下跌', '高台跳水', '大笔卖出', '封跌停板', '打开涨停板', 
                - '有大卖盘', '竞价下跌', '低开5日线', '向下缺口', '60日新低', '60日大幅下跌'
        
        返回:
            dict: 键为股票代码，值为该股票的相关异动信息，格式化为易于读取的字符串。
        """
        # 获取数据
        market_anomaly_df = ak.stock_changes_em(symbol=indicator)

        # 处理数据
        result = {}
        for _, row in market_anomaly_df.iterrows():
            anomaly_info = (
                f"时间: {row['时间']}, "
                f"名称: {row['名称']}, "
                f"板块: {row['板块']}, "
                f"相关信息: {row['相关信息']}"
            )
            result[row['代码']] = anomaly_info

        return result

    def get_active_a_stock_stats(self, indicator: Literal['近一月', '近三月', '近六月', '近一年'] = "近一月") -> dict:
        """
        获取活跃 A 股统计数据，并返回格式化结果。参数indicator:str="近一月" 返回dict[symbol,str]
        
        参数:
            indicator (str): 统计时间范围，可以选择以下选项:
                - '近一月', '近三月', '近六月', '近一年'
        
        返回:
            dict: 键为股票代码，值为该股票的统计信息，格式化为易于读取的字符串。
        """
        # 获取数据
        active_stock_stats_df = ak.stock_dzjy_hygtj(symbol=indicator)

        # 处理数据
        result = {}
        for _, row in active_stock_stats_df.iterrows():
            stats_info = (
                f"证券简称: {row['证券简称']}, "
                f"最新价: {row['最新价']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"最近上榜日: {row['最近上榜日']}, "
                f"上榜次数-总计: {row['上榜次数-总计']}, "
                f"上榜次数-溢价: {row['上榜次数-溢价']}, "
                f"上榜次数-折价: {row['上榜次数-折价']}, "
                f"总成交额: {row['总成交额']}万元, "
                f"折溢率: {row['折溢率']}%, "
                f"成交总额/流通市值: {row['成交总额/流通市值']}%, "
                f"上榜日后平均涨跌幅-1日: {row['上榜日后平均涨跌幅-1日']}%, "
                f"上榜日后平均涨跌幅-5日: {row['上榜日后平均涨跌幅-5日']}%, "
                f"上榜日后平均涨跌幅-10日: {row['上榜日后平均涨跌幅-10日']}%, "
                f"上榜日后平均涨跌幅-20日: {row['上榜日后平均涨跌幅-20日']}%"
            )
            result[row['证券代码']] = stats_info

        return result

    def get_daily_lhb_details(self,date:str=None) -> dict:
        """
        获取龙虎榜每日详情数据，并返回格式化结果。返回dict[symbol,str]
        
        返回:
            dict: 键为股票代码，值为该股票的龙虎榜详情信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_latest_trading_date()
        
        # 获取数据
        lhb_details_df = ak.stock_lhb_detail_daily_sina(date=date)

        # 处理数据
        result = {}
        for _, row in lhb_details_df.iterrows():
            lhb_info = (
                f"股票名称: {row['股票名称']}, "
                f"收盘价: {row['收盘价']}元, "
                f"对应值: {row['对应值']}%, "
                f"成交量: {row['成交量']}万股, "
                f"成交额: {row['成交额']}万元, "
                f"指标: {row['指标']}"
            )
            result[row['股票代码']] = lhb_info

        return result

    def get_stock_report_fund_hold(self, indicator: Literal["基金持仓", "QFII持仓", "社保持仓", "券商持仓", "保险持仓", "信托持仓"] = "基金持仓") -> Dict[str, str]:
        """
        获取东方财富网的机构持股报告数据。indicator="基金持仓" 返回dict[symbol,str]

        参数:
        indicator (str): 持股类型，可选值为 "基金持仓", "QFII持仓", "社保持仓", "券商持仓", "保险持仓", "信托持仓"，默认为 "基金持仓"

        返回:
        Dict[str, str]: 键为股票代码，值为格式化的持股信息字符串
        """
        try:
            # 获取最近的财报发布日期
            current_date = datetime.now()
            if current_date.month <= 4:
                report_date = f"{current_date.year - 1}-12-31"
            elif current_date.month <= 8:
                report_date = f"{current_date.year}-03-31"
            elif current_date.month <= 10:
                report_date = f"{current_date.year}-06-30"
            else:
                report_date = f"{current_date.year}-09-30"

            # 获取持股数据
            df = ak.stock_report_fund_hold(symbol=indicator, date=report_date)
            
            # 格式化数据为字典
            result = {}
            for _, row in df.iterrows():
                stock_info = (
                    f"股票简称: {row['股票简称']}, "
                    f"持有{indicator[:2]}家数: {row['持有基金家数']}家, "
                    f"持股总数: {row['持股总数']}股, "
                    f"持股市值: {row['持股市值']}元, "
                    f"持股变化: {row['持股变化']}, "
                    f"持股变动数值: {row['持股变动数值']}股, "
                    f"持股变动比例: {row['持股变动比例']}%"
                )
                result[row['股票代码']] = stock_info
            
            return result

        except Exception as e:
            return {"error": f"获取{indicator}数据时发生错误: {str(e)}"}
    
    def get_next_financial_report_date(self):
        """
        获取下一个财报发布日期(即将发生的). 返回值：str 格式：yyyymmdd

        """
        # 当前日期
        today = datetime.today()
        year = today.year
        month = today.month

        # 定义财报发行日期
        report_dates = [
            datetime(year, 3, 31),
            datetime(year, 6, 30),
            datetime(year, 9, 30),
            datetime(year, 12, 31)
        ]

        # 查找下一个财报发行日期
        for report_date in report_dates:
            if today < report_date:
                return report_date.strftime("%Y%m%d")

        # 如果当前日期在10月1日至12月31日之间，返回下一年的3月31日
        return datetime(year + 1, 3, 31).strftime("%Y%m%d")

    def get_latest_financial_report_date(self):
        """
        获取最近的财报发布日期(已经发生的)。 返回值：str 格式：yyyymmdd

        """
        # 当前日期
        today = datetime.today()
        year = today.year
        month = today.month

        # 定义财报发行日期
        report_dates = [
            datetime(year, 3, 31),
            datetime(year, 6, 30),
            datetime(year, 9, 30),
            datetime(year, 12, 31)
        ]

        # 查找最近的财报发行日期
        for report_date in reversed(report_dates):
            if today >= report_date:
                return report_date.strftime("%Y%m%d")

        # 如果当前日期在1月1日至3月30日之间，返回上一年的12月31日
        return datetime(year - 1, 12, 31).strftime("%Y%m%d")

    @cache.cache(ttl= 30*60, limiter=lg_limiter)
    def stock_market_pe_lg(self, symbol: str):
        return ak.stock_market_pe_lg(symbol=symbol)

    @thread_safe_cache.thread_safe_cached("stock_market_desc")
    def stock_market_desc(self)->str:
        """
        获取市场总体描述信息，每个市场的市盈率，指数等信息。返回值str。
        """
        market_descriptions = []
        markets = ["上证", "深证", "创业板", "科创版"]
        for market in markets:
            try:
                df = self.stock_market_pe_lg(symbol=market)
                if not df.empty:
                    latest_data = df.iloc[-1]
                    if market == "科创版":
                        description = f"{market}最新市值: {latest_data['总市值']:.2f}亿元，市盈率: {latest_data['市盈率']:.2f}"
                    else:
                        description = f"{market}最新指数: {latest_data['指数']:.2f}，平均市盈率: {latest_data['平均市盈率']:.2f}"
                    market_descriptions.append(description)
                else:
                    market_descriptions.append(f"{market}无数据")
            except Exception as e:
                market_descriptions.append(f"{market}数据获取失败: {e}")

        return "当前市场整体概况: " + "; ".join(market_descriptions)

    def get_a_stock_pb_stats(self) -> str:
        """
        获取A股等权重与中位数市净率数据。返回值str
        
        返回:
            str: A股市净率统计信息，包含日期、上证指数、市净率中位数、等权平均等信息。
        """
        # 获取数据
        pb_stats_df = ak.stock_a_all_pb()
        pb_stats_df =pb_stats_df.tail(10)
        # 处理数据并生成易于理解的字符串
        result = []
        for _, row in pb_stats_df.iterrows():
            stats_info = (
                f"日期: {row['date']}, "
                f"全部A股市净率中位数: {row['middlePB']}, "
                f"全部A股市净率等权平均: {row['equalWeightAveragePB']}, "
                f"上证指数: {row['close']}, "
                f"当前市净率中位数在历史数据上的分位数: {row['quantileInAllHistoryMiddlePB']}, "
                f"当前市净率中位数在最近10年数据上的分位数: {row['quantileInRecent10YearsMiddlePB']}, "
                f"当前市净率等权平均在历史数据上的分位数: {row['quantileInAllHistoryEqualWeightAveragePB']}, "
                f"当前市净率等权平均在最近10年数据上的分位数: {row['quantileInRecent10YearsEqualWeightAveragePB']}"
            )
            result.append(stats_info)

        return "\n".join(result)

    def get_a_stock_pe_ratios(self) -> str:
        """
        获取A股等权重与中位数市盈率数据。返回值str
        
        返回:
            str: A股市盈率统计信息，包含日期、沪深300指数、市盈率中位数、等权平均等信息。
        """
        # 获取数据
        pe_ratios_df = ak.stock_a_ttm_lyr()
        pe_ratios_df = pe_ratios_df.tail(10)

        # 处理数据并生成易于理解的字符串
        result = []
        for _, row in pe_ratios_df.iterrows():
            ratios_info = (
                f"日期: {row['date']}, "
                f"全A股滚动市盈率(TTM)中位数: {row['middlePETTM']}, "
                f"全A股滚动市盈率(TTM)等权平均: {row['averagePETTM']}, "
                f"全A股静态市盈率(LYR)中位数: {row['middlePELYR']}, "
                f"全A股静态市盈率(LYR)等权平均: {row['averagePELYR']}, "
                f"当前TTM(滚动市盈率)中位数在历史数据上的分位数: {row['quantileInAllHistoryMiddlePeTtm']}, "
                f"当前TTM(滚动市盈率)中位数在最近10年数据上的分位数: {row['quantileInRecent10YearsMiddlePeTtm']}, "
                f"当前TTM(滚动市盈率)等权平均在历史数据上的分位数: {row['quantileInAllHistoryAveragePeTtm']}, "
                f"当前TTM(滚动市盈率)等权平均在最近10年数据上的分位数: {row['quantileInRecent10YearsAveragePeTtm']}, "
                f"当前LYR(静态市盈率)中位数在历史数据上的分位数: {row['quantileInAllHistoryMiddlePeLyr']}, "
                f"当前LYR(静态市盈率)中位数在最近10年数据上的分位数: {row['quantileInRecent10YearsMiddlePeLyr']}, "
                f"当前LYR(静态市盈率)等权平均在历史数据上的分位数: {row['quantileInAllHistoryAveragePeLyr']}, "
                f"当前LYR(静态市盈率)等权平均在最近10年数据上的分位数: {row['quantileInRecent10YearsAveragePeLyr']}, "
                f"沪深300指数: {row['close']}"
            )
            result.append(ratios_info)

        return "\n".join(result)
    
    @cache.cache(ttl= 30*60, limiter=lg_limiter)
    def stock_buffett_index_lg(self) -> pd.DataFrame:
        """
        获取巴菲特指数的最新数据。返回值pd.DataFrame
        
        返回:
            pd.DataFrame: 包含巴菲特指数的最新数据，包括收盘价、总市值、GDP等信息。
        """
        return ak.stock_buffett_index_lg()

    @thread_safe_cache.thread_safe_cached("current_buffett_index")
    @retry(stop=stop_after_attempt(3),wait=wait_fixed(3))
    def get_current_buffett_index(self)->str:
        """
        获取当前巴菲特指数的最新数据.返回值str
        
        返回值:
            一个字符串，包含以下信息：
            - 收盘价
            - 总市值
            - GDP
            - 近十年分位数
            - 总历史分位数
        """
        # 获取数据
        try:
            data = self.stock_buffett_index_lg()
        except:
            return "查询巴菲特指数遇到错误"
        
        # 获取最后一行数据
        latest_data = data.iloc[-1]
        
        # 将最后一行数据转换为字符串
        buffett_index_info = (
            f"当前巴菲特指数: "
            f"收盘价: {latest_data['收盘价']}, "
            f"总市值: {latest_data['总市值']}, "
            f"GDP: {latest_data['GDP']}, "
            f"近十年分位数: {latest_data['近十年分位数']}, "
            f"总历史分位数: {latest_data['总历史分位数']}"
        )
        
        return buffett_index_info

    @cache.cache(ttl=30*60, limiter=lg_limiter)
    def stock_a_indicator_lg(self, symbol: str):
        return ak.stock_a_indicator_lg(symbol=symbol)

    @retry(retry=retry_if_exception(Exception),stop=stop_after_attempt(3),wait=wait_fixed(3))
    def get_stock_a_indicators(self, symbol: str) -> str:
        """
        获取指定股票的A股个股指标的最新数据.参数symbol:str 返回值str
        
        输入参数:
            symbol (str): 股票代码
            
        返回值 字符串类型:
            一个字符串，包含以下信息的描述：
            - 市盈率
            - 市盈率TTM
            - 市净率
            - 市销率
            - 市销率TTM
            - 股息率
            - 股息率TTM
            - 总市值
        """
        try:
            if len(self.stock_lg_code_cache) == 0:
                df = self.stock_a_indicator_lg(symbol="all")
                for _, row in df.iterrows():
                    self.stock_lg_code_cache[row['code']] = row['stock_name']
            if symbol not in self.stock_lg_code_cache:
                return f"股票代码{symbol}暂无数据"
            # 获取数据
            data = self.stock_a_indicator_lg(symbol=symbol)
            
            # 获取最后一行数据
            latest_data = data.iloc[-1]
            
            # 将最后一行数据转换为字符串
            stock_indicators_info = (
                f"A股个股指标"
                f"股票代码: {symbol} 的最新A股个股指标: "
                f"市盈率: {latest_data['pe']}, "
                f"市盈率TTM: {latest_data['pe_ttm']}, "
                f"市净率: {latest_data['pb']}, "
                f"市销率: {latest_data['ps']}, "
                f"市销率TTM: {latest_data['ps_ttm']}, "
                f"股息率: {latest_data['dv_ratio']}, "
                f"股息率TTM: {latest_data['dv_ttm']}, "
                f"总市值: {latest_data['total_mv']}"
            )
            
            return stock_indicators_info
        except Exception as e:
            return f"获取股票{symbol}的A股个股指标数据时发生错误:暂无数据"

    def get_industry_pe_ratio(self, symbol: str, date: str = None) -> Dict[str, str]:
        """
        获取指定日期和行业分类的行业市盈率数据。

        输入参数:
            symbol (str): 行业分类，选择 {"证监会行业分类", "国证行业分类"}
            date (str): 交易日，格式为 "YYYYMMDD"。如果未提供，则使用最近的一个交易日。
            
        返回值:
            Dict[str, str]: 一个字典，键为行业名称，值为包含该行业信息的字符串
        """
        if not date:
            date = self.get_previous_trading_date()
        
        # 获取数据
        data = ak.stock_industry_pe_ratio_cninfo(symbol=symbol, date=date)
        
        # 初始化结果字典
        result = {}
        
        # 遍历所有行，生成字符串并添加到字典中
        for _, row in data.iterrows():
            industry_name = row['行业名称']
            industry_pe_ratio_info = (
                f"行业分类: {row['行业分类']}, "
                f"行业层级: {row['行业层级']}, "
                f"行业编码: {row['行业编码']}, "
                f"公司数量: {row['公司数量']}, "
                f"纳入计算公司数量: {row['纳入计算公司数量']}, "
                f"总市值-静态: {row['总市值-静态']}亿元, "
                f"净利润-静态: {row['净利润-静态']}亿元, "
                f"静态市盈率-加权平均: {row['静态市盈率-加权平均']}, "
                f"静态市盈率-中位数: {row['静态市盈率-中位数']}, "
                f"静态市盈率-算术平均: {row['静态市盈率-算术平均']}"
            )
            result[industry_name] = industry_pe_ratio_info
        
        return result

    def get_institute_recommendations(self, indicator: Literal['最新投资评级', '上调评级股票', '下调评级股票', '股票综合评级', '首次评级股票', '目标涨幅排名', '机构关注度', '行业关注度', '投资评级选股'] = "投资评级选股") -> dict:
        """
        获取机构推荐池数据，并返回格式化结果。 参数indicator:str="投资评级选股" 返回值Dict[symbol,str]
        
        参数:
            indicator (str): 选择的机构推荐类型，可以选择以下选项:
                - '最新投资评级', '上调评级股票', '下调评级股票', '股票综合评级', 
                - '首次评级股票', '目标涨幅排名', '机构关注度', '行业关注度', '投资评级选股'
        
        返回:
            dict: 键为股票代码，值为该股票的推荐信息，格式化为易于读取的字符串。
        """
        # 获取数据
        recommendations_df = ak.stock_institute_recommend(symbol=indicator)

        # 寻找股票代码列
        code_columns = ['股票代码', 'symbol', 'code']
        code_column = next((col for col in code_columns if col in recommendations_df.columns), None)

        if not code_column:
            raise ValueError("无法找到股票代码的列。")

        # 处理数据
        result = {}
        for _, row in recommendations_df.iterrows():
            recommendation_info = ", ".join([f"{col}: {row[col]}" for col in recommendations_df.columns if col != code_column])
            result[row[code_column]] = recommendation_info

        return result

    def get_recent_recommendations_summary(self, symbol: str) -> str:
        """
        获取指定股票的最近半年的评级记录统计.参数symbol:str 返回值str
        
        输入参数:
            symbol (str): 股票代码
            
        返回值:
            一个描述性的字符串，包含以下信息的统计：
            - 股票名称
            - 最近半年内的评级次数
            - 各种评级的次数统计（例如：买入、增持等）
            - 涉及的分析师数量
            - 涉及的评级机构数量
            - 目标价的最高值、最低值、平均值
            - 目标价的分布情况（最多的目标价区间）
        """
        # 获取数据
        data = ak.stock_institute_recommend_detail(symbol=symbol)
        
        # 计算最近半年的日期
        six_months_ago = datetime.now() - timedelta(days=180)
        
        # 过滤最近半年的数据
        recent_data = data[data['评级日期'] >= six_months_ago.strftime('%Y-%m-%d')]
        
        # 统计股票名称
        stock_name = recent_data['股票名称'].iloc[0] if not recent_data.empty else "未知"
        
        # 统计评级次数
        total_recommendations = recent_data.shape[0]
        
        # 统计各种评级的次数
        rating_counts = recent_data['最新评级'].value_counts().to_dict()
        
        # 统计涉及的分析师数量
        analysts = recent_data['分析师'].str.split(',').explode().unique()
        num_analysts = len(analysts)
        
        # 统计涉及的评级机构数量
        institutions = recent_data['评级机构'].unique()
        num_institutions = len(institutions)
        
        # 统计目标价
        target_prices = recent_data['目标价'].replace('NaN', np.nan).dropna().astype(float)
        if not target_prices.empty:
            max_target_price = target_prices.max()
            min_target_price = target_prices.min()
            avg_target_price = target_prices.mean()
            
            # 计算目标价的分布情况
            bins = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
            target_price_distribution = np.histogram(target_prices, bins=bins)
            most_common_range_index = np.argmax(target_price_distribution[0])
            most_common_range = f"{bins[most_common_range_index]}-{bins[most_common_range_index + 1]}"
        else:
            max_target_price = min_target_price = avg_target_price = most_common_range = "无数据"
        
        # 生成描述性的字符串
        recommendation_summary = (
            f"股票代码: {symbol}, 股票名称: {stock_name}\n"
            f"最近半年内的评级次数: {total_recommendations}\n"
            f"评级统计:\n"
        )
        
        for rating, count in rating_counts.items():
            recommendation_summary += f" - {rating}: {count}次\n"
        
        recommendation_summary += (
            f"涉及的分析师数量: {num_analysts}\n"
            f"涉及的评级机构数量: {num_institutions}\n"
            f"目标价统计:\n"
            f" - 最高目标价: {max_target_price}\n"
            f" - 最低目标价: {min_target_price}\n"
            f" - 平均目标价: {avg_target_price}\n"
            f" - 最多的目标价区间: {most_common_range}\n"
        )
        
        return recommendation_summary

    def _get_file_content_cninfo(self, file: str) -> str:
        """
        获取 JS 文件的内容
        :param file: 文件名
        :type file: str
        :return: 文件内容
        :rtype: str
        """
        import pathlib
        module_path = pathlib.Path(__file__).resolve().parent
        file_path = module_path / file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果 UTF-8 解码失败，尝试使用 ISO-8859-1
            with open(file_path, "r", encoding="iso-8859-1") as f:
                return f.read()

    def stock_rank_forecast_cninfo(self,date: str = "20240630") -> pd.DataFrame:
        """
        巨潮资讯-数据中心-评级预测-投资评级
        http://webapi.cninfo.com.cn/#/thematicStatistics?name=%E6%8A%95%E8%B5%84%E8%AF%84%E7%BA%A7
        :param date: 查询日期
        :type date: str
        :return: 投资评级
        :rtype: pandas.DataFrame
        """
        import py_mini_racer
        import requests
        url = "http://webapi.cninfo.com.cn/api/sysapi/p_sysapi1089"
        params = {"tdate": "-".join([date[:4], date[4:6], date[6:]])}
        js_code = py_mini_racer.MiniRacer()
        js_content = self._get_file_content_cninfo("cninfo.js")
        js_code.eval(js_content)
        mcode = js_code.call("getResCode1")
        headers = {
            "Accept": "*/*",
            "Accept-Enckey": mcode,
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Content-Length": "0",
            "Host": "webapi.cninfo.com.cn",
            "Origin": "http://webapi.cninfo.com.cn",
            "Pragma": "no-cache",
            "Proxy-Connection": "keep-alive",
            "Referer": "http://webapi.cninfo.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = requests.post(url, params=params, headers=headers)
        data_json = r.json()
        temp_df = pd.DataFrame(data_json["records"])
        temp_df.columns = [
            "证券简称",
            "发布日期",
            "前一次投资评级",
            "评级变化",
            "目标价格-上限",
            "是否首次评级",
            "投资评级",
            "研究员名称",
            "研究机构简称",
            "目标价格-下限",
            "证券代码",
        ]
        temp_df = temp_df[
            [
                "证券代码",
                "证券简称",
                "发布日期",
                "研究机构简称",
                "研究员名称",
                "投资评级",
                "是否首次评级",
                "评级变化",
                "前一次投资评级",
                "目标价格-下限",
                "目标价格-上限",
            ]
        ]
        temp_df["目标价格-上限"] = pd.to_numeric(temp_df["目标价格-上限"], errors="coerce")
        temp_df["目标价格-下限"] = pd.to_numeric(temp_df["目标价格-下限"], errors="coerce")
        return temp_df

    @cache.cache(ttl=60*60*7)
    def get_investment_ratings(self, date: str = None) -> dict:
        """
        获取投资评级数据，并返回格式化结果。使用线程安全的缓存机制。
        
        参数:
            date (str): 交易日期，格式为 'yyyymmdd'。如果未提供，则获取最近一个交易日的数据。
        
        返回:
            dict: 键为证券代码，值为该证券的投资评级信息，格式化为易于读取的字符串。
        """
        if not date:
            date = self.get_previous_trading_date()
        
        @thread_safe_cache.thread_safe_cached(f'investment_ratings_{date}')
        def fetch_ratings(query_date: str) -> dict:
            """
            使用线程安全的方式获取并格式化投资评级数据。
            """
            # 获取数据
            ratings_df = ak.stock_rank_forecast_cninfo(date=query_date)

            # 确定证券代码列
            code_columns = ['证券代码', 'symbol', 'code']
            code_column = next((col for col in code_columns if col in ratings_df.columns), None)

            if not code_column:
                raise ValueError("无法找到证券代码的列。")

            # 处理数据
            result = {}
            for _, row in ratings_df.iterrows():
                rating_info = ", ".join([f"{col}: {row[col]}" for col in ratings_df.columns if col != code_column])
                result[row[code_column]] = rating_info

            return result

        # 调用带有缓存的函数获取数据
        return fetch_ratings(date)
    
    @retry(retry=retry_if_exception(Exception),stop=stop_after_attempt(3),wait=wait_fixed(3))
    def stock_financial_analysis_indicator(self,
        symbol: str = "600004", start_year: str = "1900"
    ) -> pd.DataFrame:
        """
        新浪财经-财务分析-财务指标
        https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/stockid/600004/ctrl/2019/displaytype/4.phtml
        :param symbol: 股票代码
        :type symbol: str
        :param start_year: 开始年份
        :type start_year: str
        :return: 新浪财经-财务分析-财务指标
        :rtype: pandas.DataFrame
        """
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup
        from akshare.utils.tqdm import get_tqdm
        from io import StringIO
        url = (
            f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/"
            f"stockid/{symbol}/ctrl/2020/displaytype/4.phtml"
        )
        r = requests.get(url)
        soup = BeautifulSoup(r.text, features="lxml")
        year_context = soup.find(attrs={"id": "con02-1"}).find("table").find_all("a")
        year_list = [item.text for item in year_context]
        if start_year in year_list:
            year_list = year_list[: year_list.index(start_year) + 1]
        out_df = pd.DataFrame()
        tqdm = get_tqdm()
        for year_item in tqdm(year_list, leave=False):
            url = (
                f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/"
                f"stockid/{symbol}/ctrl/{year_item}/displaytype/4.phtml"
            )
            r = requests.get(url)
            temp_df = pd.read_html(StringIO(r.text))[12].iloc[:, :-1]
            temp_df.columns = temp_df.iloc[0, :]
            temp_df = temp_df.iloc[1:, :]
            big_df = pd.DataFrame()
            indicator_list = [
                "每股指标",
                "盈利能力",
                "成长能力",
                "营运能力",
                "偿债及资本结构",
                "现金流量",
                "其他指标",
            ]
            for i in range(len(indicator_list)):
                if i == 6:
                    inner_df = temp_df[
                        temp_df.loc[
                            temp_df.iloc[:, 0].str.find(indicator_list[i]) == 0, :
                        ].index[0] :
                    ].T
                else:
                    inner_df = temp_df[
                        temp_df.loc[
                            temp_df.iloc[:, 0].str.find(indicator_list[i]) == 0, :
                        ].index[0] : temp_df.loc[
                            temp_df.iloc[:, 0].str.find(indicator_list[i + 1]) == 0, :
                        ].index[0]
                        - 1
                    ].T
                inner_df = inner_df.reset_index(drop=True)
                big_df = pd.concat(objs=[big_df, inner_df], axis=1)
            big_df.columns = big_df.iloc[0, :].tolist()
            big_df = big_df.iloc[1:, :]
            big_df.index = temp_df.columns.tolist()[1:]
            out_df = pd.concat(objs=[out_df, big_df])

        out_df.dropna(inplace=True)
        out_df.reset_index(inplace=True)
        out_df.rename(columns={"index": "日期"}, inplace=True)
        out_df.sort_values(by=["日期"], ignore_index=True, inplace=True)
        out_df["日期"] = pd.to_datetime(out_df["日期"], errors="coerce").dt.date
        for item in out_df.columns[1:]:
            out_df[item] = pd.to_numeric(out_df[item], errors="coerce")
        return out_df

    @cache.cache(ttl=60*60*24,limiter=sn_limiter)
    def get_financial_analysis_summary(self, symbol: str, start_year: str = "2024") -> str:
        """
        获取指定股票的财务分析指标，并返回易于理解的字符串形式的结果。参数symbol:str 返回值str

        参数:
            symbol (str): 股票代码。
            start_year (str): 查询的起始年份，默认为 "2024"。

        返回:
            str: 股票的最新财务分析指标，格式化为易于理解的字符串。
        """
        now = datetime.now()
        year = now.year
        if year > int(start_year):
            start_year = year
        # 获取数据
        df = self.stock_financial_analysis_indicator(symbol, start_year)
        
        # 取最后一行数据（最新数据）
        latest_data = df.tail(1).squeeze()

        # 生成易于理解的字符串
        result = "\n".join([f"{index}: {value}" for index, value in latest_data.items()])

        return result
 
    @cache.cache(ttl=60*60*24*7)
    def get_key_financial_indicators(self, symbol: str, indicator: Literal["按报告期", "按年度", "按单季度"] = "按报告期") -> str:
        """
        获取指定股票的关键财务指标摘要，并返回易于理解的字符串形式的结果。参数symbol:str,indicator:str="按报告期" 返回值str

        参数:
            symbol (str): 股票代码。
            indicator (str): 财务指标的时间范围，可选值为 "按报告期", "按年度", "按单季度"。默认值为 "按报告期"。

        返回:
            str: 股票的关键财务指标摘要，格式化为易于理解的字符串。
        """
        # 获取数据
        df = ak.stock_financial_abstract_ths(symbol, indicator)

        # 取最新一行数据
        latest_data = df.tail(1).squeeze()

        # 生成易于理解的字符串
        result = "\n".join([f"{index}: {value}" for index, value in latest_data.items()])

        return result

    def get_stock_balance_sheet_by_report_em(self, symbol: str) -> str:
        """
        获取指定股票的最新资产负债表，并将所有列的数据拼接成一个字符串返回。参数symbol:str 返回值str

        参数:
        symbol (str): 股票代码。

        返回:
        str: 资产负债表的格式化字符串，包括所有319项数据。
        """
        df = ak.stock_balance_sheet_by_report_em(symbol)
        
        # 选择最新的一行数据
        latest_row = df.tail(1).iloc[0]
        
        # 将所有列的数据拼接成一个字符串
        balance_sheet_str = "\n".join([f"{col}: {latest_row[col]}" for col in df.columns])
        
        return balance_sheet_str

    def get_individual_stock_fund_flow_rank(self, indicator: str = "今日") -> dict:
        """
        获取个股资金流排名，并返回格式化结果。参数indicator:str="今日" 返回值Dict[symbol,str]

        参数:
            indicator (str): 资金流动的时间范围，可选值为 "今日", "3日", "5日", "10日"。默认值为 "今日"。

        返回:
            dict: 键为股票代码，值为该股票的资金流动信息，格式化为易于读取的字符串。
        """
        # 获取数据
        fund_flow_df = ak.stock_individual_fund_flow_rank(indicator=indicator)

        # 确定股票代码列
        code_columns = ['代码', 'symbol', 'code']
        code_column = next((col for col in code_columns if col in fund_flow_df.columns), None)

        if not code_column:
            raise ValueError("无法找到股票代码的列。")

        # 处理数据
        result = {}
        for _, row in fund_flow_df.iterrows():
            fund_flow_info = ", ".join([f"{col}: {row[col]}" for col in fund_flow_df.columns if col != code_column])
            result[row[code_column]] = fund_flow_info

        return result

    def get_individual_stock_fund_flow(self, symbol: str, market: str = "sh") -> dict:
        """
        获取指定股票的资金流动信息，并返回格式化结果。参数symbol:str market:str="sh" 返回值Dict[symbol,str]

        参数:
            symbol (str): 股票代码。
            market (str): 证券市场代码，可选值为 "sh"（上海证券交易所）、"sz"（深证证券交易所）、"bj"（北京证券交易所）。默认值为 "sh"。

        返回:
            dict: 键为股票代码，值为该股票的资金流动信息，格式化为易于读取的字符串。
        """
        # 获取数据
        fund_flow_df = ak.stock_individual_fund_flow(symbol=symbol, market=market)

        # 确定股票代码列
        code_columns = ['股票代码', 'symbol', 'code']
        code_column = next((col for col in code_columns if col in fund_flow_df.columns), None)

        if not code_column:
            raise ValueError("无法找到股票代码的列。")

        # 处理数据
        result = {}
        for _, row in fund_flow_df.iterrows():
            fund_flow_info = ", ".join([f"{col}: {row[col]}" for col in fund_flow_df.columns if col != code_column])
            result[row[code_column]] = fund_flow_info

        return result

    def get_cash_flow_statement_summary(self) -> dict:
        """
        获取最近一个财报发行日期的现金流量表数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 股票简称
            - 净现金流
            - 净现金流同比增长
            - 经营性现金流净额
            - 经营性现金流净额占比
            - 投资性现金流净额
            - 投资性现金流净额占比
            - 融资性现金流净额
            - 融资性现金流净额占比
            - 公告日期
        """
        # 获取最近的财报发行日期
        date = self.get_latest_financial_report_date()

        # 检查缓存是否存在
        if date in self.cash_flow_cache:
            return self.cash_flow_cache[date]
        
        # 获取数据
        data = ak.stock_xjll_em(date=date)
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"股票简称: {row['股票简称']}, "
                f"净现金流: {row['净现金流-净现金流']}元, "
                f"净现金流同比增长: {row['净现金流-同比增长']}%, "
                f"经营性现金流净额: {row['经营性现金流-现金流量净额']}元, "
                f"经营性现金流净额占比: {row['经营性现金流-净现金流占比']}%, "
                f"投资性现金流净额: {row['投资性现金流-现金流量净额']}元, "
                f"投资性现金流净额占比: {row['投资性现金流-净现金流占比']}%, "
                f"融资性现金流净额: {row['融资性现金流-现金流量净额']}元, "
                f"融资性现金流净额占比: {row['融资性现金流-净现金流占比']}%, "
            )
            summary_dict[row['股票代码']] = description
        
        # 缓存结果
        self.cash_flow_cache[date] = summary_dict
        
        return summary_dict

    def get_profit_statement_summary(self) -> dict:
        """
        获取最近一个财报发行日期的利润表数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 股票简称
            - 净利润
            - 净利润同比
            - 营业总收入
            - 营业总收入同比
            - 营业总支出-营业支出
            - 营业总支出-销售费用
            - 营业总支出-管理费用
            - 营业总支出-财务费用
            - 营业总支出-营业总支出
            - 营业利润
            - 利润总额
            - 公告日期
        """
        date = self.get_latest_financial_report_date()

        # 检查缓存是否存在
        if date in self.profit_cache:
            return self.profit_cache[date]
        
        # 获取数据
        data = ak.stock_lrb_em(date=date)
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"股票简称: {row['股票简称']}, "
                f"净利润: {row['净利润']}元, "
                f"净利润同比: {row['净利润同比']}%, "
                f"营业总收入: {row['营业总收入']}元, "
                f"营业总收入同比: {row['营业总收入同比']}%, "
                f"营业总支出-营业支出: {row['营业总支出-营业支出']}元, "
                f"营业总支出-销售费用: {row['营业总支出-销售费用']}元, "
                f"营业总支出-管理费用: {row['营业总支出-管理费用']}元, "
                f"营业总支出-财务费用: {row['营业总支出-财务费用']}元, "
                f"营业总支出-营业总支出: {row['营业总支出-营业总支出']}元, "
                f"营业利润: {row['营业利润']}元, "
                f"利润总额: {row['利润总额']}元, "
            )
            summary_dict[row['股票代码']] = description
        
        # 缓存结果
        self.profit_cache[date] = summary_dict
        
        return summary_dict

    def get_latest_market_fund_flow(self) -> Dict:
        """
        获取大盘资金流数据，返回值Dict
        """
        # 获取大盘资金流数据
        stock_market_fund_flow_df = ak.stock_market_fund_flow()
        
        # 获取最后一行数据
        latest_row = stock_market_fund_flow_df.iloc[-1]
        
        # 将最后一行数据转换为字典
        result = latest_row.to_dict()
        
        return result

    def get_balance_sheet_summary(self) -> dict:
        """
        获取最近一个财报发行日期的资产负债表数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 股票简称
            - 资产-货币资金
            - 资产-应收账款
            - 资产-存货
            - 资产-总资产
            - 资产-总资产同比
            - 负债-应付账款
            - 负债-总负债
            - 负债-预收账款
            - 负债-总负债同比
            - 资产负债率
            - 股东权益合计
            - 公告日期
        """
        date = self.get_latest_financial_report_date()

        # 检查缓存是否存在
        if date in self.balance_sheet_cache:
            return self.balance_sheet_cache[date]
        
        # 获取数据
        data = ak.stock_zcfz_em(date=date)
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"股票简称: {row['股票简称']}, "
                f"资产-货币资金: {row['资产-货币资金']}元, "
                f"资产-应收账款: {row['资产-应收账款']}元, "
                f"资产-存货: {row['资产-存货']}元, "
                f"资产-总资产: {row['资产-总资产']}元, "
                f"资产-总资产同比: {row['资产-总资产同比']}%, "
                f"负债-应付账款: {row['负债-应付账款']}元, "
                f"负债-总负债: {row['负债-总负债']}元, "
                f"负债-预收账款: {row['负债-预收账款']}元, "
                f"负债-总负债同比: {row['负债-总负债同比']}%, "
                f"资产负债率: {row['资产负债率']}%, "
                f"股东权益合计: {row['股东权益合计']}元, "
                f"公告日期: {row['公告日期']}"
            )
            summary_dict[row['股票代码']] = description
        
        # 缓存结果
        self.balance_sheet_cache[date] = summary_dict
        
        return summary_dict

    def get_stock_concept_fund_flow_top(self,indicator:Literal["即时", "3日排行", "5日排行", "10日排行", "20日排行"] = "即时",num:int=25)->list:
        """
        获取概念板块资金流数据，参数indicator:str="即时"(可选:3日排行, 5日排行, 10日排行, 20日排行) num:int=25 返回值list
        """
        return ak.stock_concept_fund_flow_em(symbol=indicator).head(num).to_dict(orient='records')

    def stock_gdfx_free_top_10_em(self, symbol: str = "sh688686", date: str = "20210630") -> pd.DataFrame:
        """
        获取东方财富网-个股-十大流通股东数据。
        
        参数:
            symbol (str): 带市场标识的股票代码，如 "sh688686"
            date (str): 报告期，格式为 "YYYYMMDD"
        
        返回:
            pandas.DataFrame: 包含以下列的数据框：
                - 名次：股东排名 (HOLDER_RANK)
                - 股东名称：股东的名称 (HOLDER_NAME)
                - 股东性质：股东的性质类别 (HOLDER_TYPE)
                - 股份类型：所持股份的类型 (SHARES_TYPE)
                - 持股数：持股数量 (HOLD_NUM)
                - 占总流通股本持股比例：持股占比 (FREE_HOLDNUM_RATIO)
                - 增减：持股的增减变动 (HOLD_NUM_CHANGE)
                - 变动比率：变动的比率 (CHANGE_RATIO)
        """
        import requests
        url = "https://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/PageSDLTGD"
        params = {
            "code": f"{symbol.upper()}",
            "date": f"{'-'.join([date[:4], date[4:6], date[6:]])}"
        }

        try:
            r = requests.get(url, params=params)
            data_json = r.json()
            
            # 从返回的JSON中获取sdltgd数据
            if "sdltgd" not in data_json or not data_json["sdltgd"]:
                return pd.DataFrame(columns=[
                    "名次", "股东名称", "股东性质", "股份类型", 
                    "持股数", "占总流通股本持股比例", "增减", "变动比率"
                ])

            temp_df = pd.DataFrame(data_json["sdltgd"])
            
            # 定义列映射
            column_mapping = {
                'HOLDER_RANK': '名次',
                'HOLDER_NAME': '股东名称',
                'HOLDER_TYPE': '股东性质',
                'SHARES_TYPE': '股份类型',
                'HOLD_NUM': '持股数',
                'FREE_HOLDNUM_RATIO': '占总流通股本持股比例',
                'HOLD_NUM_CHANGE': '增减',
                'CHANGE_RATIO': '变动比率'
            }
            
            # 重命名列
            temp_df = temp_df.rename(columns=column_mapping)
            
            # 选择需要的列
            temp_df = temp_df[[
                "名次", "股东名称", "股东性质", "股份类型", 
                "持股数", "占总流通股本持股比例", "增减", "变动比率"
            ]]

            # 转换数值类型
            numeric_columns = ["持股数", "占总流通股本持股比例", "变动比率"]
            for col in numeric_columns:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            return temp_df

        except Exception as e:
            self.logger.error(f"获取股东数据时出错: {str(e)}")
            # 返回空DataFrame，保持列结构一致
            return pd.DataFrame(columns=[
                "名次", "股东名称", "股东性质", "股份类型", 
                "持股数", "占总流通股本持股比例", "增减", "变动比率"
            ])

    def stock_gdfx_top_10_em(self, symbol: str = "sh688686", date: str = "20210630") -> pd.DataFrame:
        """
        获取东方财富网-个股-十大股东数据。
        
        参数:
            symbol (str): 带市场标识的股票代码，如 "sh688686"
            date (str): 报告期，格式为 "YYYYMMDD"
        
        返回:
            pandas.DataFrame: 包含以下列的数据框：
                - 名次：股东排名 (HOLDER_RANK)
                - 股东名称：股东的名称 (HOLDER_NAME)
                - 股份类型：所持股份的类型 (SHARES_TYPE)
                - 持股数：持股数量 (HOLD_NUM)
                - 占总股本持股比例：持股占比 (HOLD_NUM_RATIO)
                - 增减：持股的增减变动 (HOLD_NUM_CHANGE)
                - 变动比率：变动的比率 (CHANGE_RATIO)
        """
        import requests
        url = "https://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/PageSDGD"
        params = {
            "code": f"{symbol.upper()}",
            "date": f"{'-'.join([date[:4], date[4:6], date[6:]])}"
        }

        try:
            r = requests.get(url, params=params)
            data_json = r.json()
            
            # 检查是否有sdgd数据
            if "sdgd" not in data_json or not data_json["sdgd"]:
                return pd.DataFrame(columns=[
                    "名次", "股东名称", "股份类型", "持股数",
                    "占总股本持股比例", "增减", "变动比率"
                ])

            # 创建DataFrame
            temp_df = pd.DataFrame(data_json["sdgd"])
            
            # 定义字段映射
            column_mapping = {
                'HOLDER_RANK': '名次',
                'HOLDER_NAME': '股东名称',
                'SHARES_TYPE': '股份类型',
                'HOLD_NUM': '持股数',
                'HOLD_NUM_RATIO': '占总股本持股比例',
                'HOLD_NUM_CHANGE': '增减',
                'CHANGE_RATIO': '变动比率'
            }
            
            # 重命名列
            temp_df = temp_df.rename(columns=column_mapping)
            
            # 选择需要的列
            temp_df = temp_df[[
                "名次", "股东名称", "股份类型", "持股数",
                "占总股本持股比例", "增减", "变动比率"
            ]]

            # 转换数值类型
            numeric_columns = ["持股数", "占总股本持股比例", "变动比率"]
            for col in numeric_columns:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            return temp_df

        except Exception as e:
            self.logger.error(f"获取十大股东数据时出错: {str(e)}")
            # 返回空DataFrame，保持列结构一致
            return pd.DataFrame(columns=[
                "名次", "股东名称", "股份类型", "持股数",
                "占总股本持股比例", "增减", "变动比率"
            ])

    def fetch_top_10_tradable_shareholders(self, symbol: str, date: str = None) -> List[Dict]:
        """
        获取指定股票的前10大流通股股东信息。

        参数:
            symbol (str): 股票代码，如 "sh600000"。
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            List[Dict]: 包含股东信息的字典列表。
        """
        if not date:
            date = self.get_latest_financial_report_date()
        symbol = self.add_market_prefix(symbol)
        df = self.stock_gdfx_free_top_10_em(symbol=symbol, date=date)
        return df.to_dict('records')

    def fetch_top_10_shareholders(self, symbol: str, date: str = None) -> List[Dict]:
        """
        获取指定股票的前10大股东信息。

        参数:
            symbol (str): 股票代码，如 "600000"。
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            List[Dict]: 包含股东信息的字典列表。
        """
        if not date:
            date = self.get_latest_financial_report_date()
        symbol = self.add_market_prefix(symbol)
        symbol = symbol.upper()
        df = self.stock_gdfx_top_10_em(symbol=symbol, date=date)
        return df.to_dict('records')

    @cache.cache(ttl=60*60*24)
    def get_shareholders_data(self, symbol: str, date: str = None) -> Dict[str, List[Dict]]:
        """
        获取指定股票的前10大流通股股东和前10大股东信息。

        参数:
            symbol (str): 股票代码，如 "600000"。
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            Dict[str, List[Dict]]: 包含两类股东数据的字典：
                - "top_10_tradable": 前10大流通股股东信息
                - "top_10_total": 前10大股东信息
        """
        if not date:
            date = self.get_latest_financial_report_date()
        symbol = self.add_market_prefix(symbol)
        return {
            "top_10_tradable": self.fetch_top_10_tradable_shareholders(symbol, date),
            "top_10_total": self.fetch_top_10_shareholders(symbol, date)
        }

    def fetch_top_10_tradable_shareholders_analysis(self, date: str = None) -> List[Dict]:
        """
        获取所有股票前10大流通股股东的分析数据。

        参数:
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            List[Dict]: 包含前10大流通股股东分析数据的字典列表。
        """
        if not date:
            date = self.get_latest_trading_date()
        df = ak.stock_gdfx_free_holding_analyse_em(date=date)
        return df.to_dict('records')

    def fetch_top_10_shareholders_analysis(self, date: str = None) -> List[Dict]:
        """
        获取所有股票前10大股东的分析数据。

        参数:
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            List[Dict]: 包含前10大股东分析数据的字典列表。
        """
        if not date:
            date = self.get_latest_trading_date()
        df = ak.stock_gdfx_holding_analyse_em(date=date)
        return df.to_dict('records')

    def get_shareholders_analysis(self, date: str = None) -> Dict[str, List[Dict]]:
        """
        获取所有股票的前10大流通股股东和前10大股东的分析数据。

        参数:
            date (str): 财报日期，格式为"YYYYMMDD"。如果未提供，则使用最近的交易日期。

        返回:
            Dict[str, List[Dict]]: 包含两类股东分析数据的字典：
                - "top_10_tradable_analysis": 前10大流通股股东分析数据
                - "top_10_total_analysis": 前10大股东分析数据
        """
        if not date:
            date = self.get_latest_trading_date()
        return {
            "top_10_tradable_analysis": self.fetch_top_10_tradable_shareholders_analysis(date),
            "top_10_total_analysis": self.fetch_top_10_shareholders_analysis(date)
        }

    @cache.cache(ttl=60*60*24*7)
    def get_stock_info_dict(self,symbol:str)->dict:
        """
        公司概况
        输入参数:symbol	str	股票代码
        返回值:
            名称	类型	描述
            公司名称	object	-
            英文名称	object	-
            曾用简称	object	-
            A股代码	object	-
            A股简称	object	-
            B股代码	object	-
            B股简称	object	-
            H股代码	object	-
            H股简称	object	-
            入选指数	object	-
            所属市场	object	-
            所属行业	object	-
            法人代表	object	-
            注册资金	object	-
            成立日期	object	-
            上市日期	object	-
            官方网站	object	-
            电子邮箱	object	-
            联系电话	object	-
            传真	object	-
            注册地址	object	-
            办公地址	object	-
            邮政编码	object	-
            主营业务	object	-
            经营范围	object	-
            机构简介	object	-
        """
        return ak.stock_profile_cninfo(symbol=symbol).to_dict(orient='records')[0]

    def get_stock_report(self, symbol: str) -> str:
        """
        获取指定股票的个股研报数据，过滤超过180天的数据，并进行统计分析（仅包含2024年数据）。返回值str

        参数:
            symbol (str): 股票代码。

        返回:
            str: 经过过滤和统计的个股研报数据，格式化为易于阅读的字符串。
        """
        # 获取数据
        reports_df = ak.stock_research_report_em(symbol)

        # 过滤超过180天的数据
        cutoff_date = datetime.now() - timedelta(days=180)
        reports_df['日期'] = pd.to_datetime(reports_df['日期'], errors='coerce')
        filtered_df = reports_df[reports_df['日期'] >= cutoff_date]

        if filtered_df.empty:
            return "没有找到最近180天内的研报数据。"

        # 统计分析，仅针对2024年数据
        stats = {
            "2024-盈利预测-收益": {
                "max": filtered_df["2024-盈利预测-收益"].max(),
                "min": filtered_df["2024-盈利预测-收益"].min(),
                "mean": filtered_df["2024-盈利预测-收益"].mean(),
            },
            "2024-盈利预测-市盈率": {
                "max": filtered_df["2024-盈利预测-市盈率"].max(),
                "min": filtered_df["2024-盈利预测-市盈率"].min(),
                "mean": filtered_df["2024-盈利预测-市盈率"].mean(),
            }
        }

        # 返回易于阅读的字符串
        result = [
            f"2024年盈利预测-收益: 最高值: {stats['2024-盈利预测-收益']['max']}, 最低值: {stats['2024-盈利预测-收益']['min']}, 平均值: {stats['2024-盈利预测-收益']['mean']}",
            f"2024年盈利预测-市盈率: 最高值: {stats['2024-盈利预测-市盈率']['max']}, 最低值: {stats['2024-盈利预测-市盈率']['min']}, 平均值: {stats['2024-盈利预测-市盈率']['mean']}"
        ]

        return "\n".join(result)

    def get_financial_forecast_summary(self) -> dict:
        """
        获取最近一个财报发行日期的业绩预告数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 股票简称
            - 预测指标
            - 业绩变动
            - 预测数值
            - 业绩变动幅度
            - 业绩变动原因
            - 预告类型
            - 上年同期值
            - 公告日期
        """
        date = self.get_latest_financial_report_date()

        # 检查缓存是否存在
        if date in self.forecast_cache:
            return self.forecast_cache[date]
        
        # 获取数据
        data = ak.stock_yjyg_em(date=date)
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"股票简称: {row['股票简称']}, "
                f"预测指标: {row['预测指标']}, "
                f"业绩变动: {row['业绩变动']}, "
                f"预测数值: {row['预测数值']}元, "
                f"业绩变动幅度: {row['业绩变动幅度']}%, "
                f"业绩变动原因: {row['业绩变动原因']}, "
                f"预告类型: {row['预告类型']}, "
                f"上年同期值: {row['上年同期值']}元, "
                f"公告日期: {row['公告日期']}"
            )
            summary_dict[row['股票代码']] = description
        
        # 缓存结果
        self.forecast_cache[date] = summary_dict
        
        return summary_dict

    def get_financial_report_summary(self) -> dict:
        """
        获取最近一个财报发行日期的业绩报表数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 股票简称
            - 每股收益
            - 营业收入
            - 营业收入同比增长
            - 营业收入季度环比增长
            - 净利润
            - 净利润同比增长
            - 净利润季度环比增长
            - 每股净资产
            - 净资产收益率
            - 每股经营现金流量
            - 销售毛利率
            - 所处行业
            - 最新公告日期
        """
        date = self.get_latest_financial_report_date()

        # 检查缓存是否存在
        if date in self.report_cache:
            return self.report_cache[date]
        
        # 获取数据
        data = ak.stock_yjbb_em(date=date)
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"股票简称: {row['股票简称']}, "
                f"每股收益: {row['每股收益']}元, "
                f"营业收入: {row['营业收入-营业收入']}元, "
                f"营业收入同比增长: {row['营业收入-同比增长']}%, "
                f"营业收入季度环比增长: {row['营业收入-季度环比增长']}%, "
                f"净利润: {row['净利润-净利润']}元, "
                f"净利润同比增长: {row['净利润-同比增长']}%, "
                f"净利润季度环比增长: {row['净利润-季度环比增长']}%, "
                f"每股净资产: {row['每股净资产']}元, "
                f"净资产收益率: {row['净资产收益率']}%, "
                f"每股经营现金流量: {row['每股经营现金流量']}元, "
                f"销售毛利率: {row['销售毛利率']}%, "
                f"所处行业: {row['所处行业']}, "
                f"最新公告日期: {row['最新公告日期']}"
            )
            summary_dict[row['股票代码']] = description
        
        # 缓存结果
        self.report_cache[date] = summary_dict
        
        return summary_dict

    def get_top_holdings_by_market(self, market: Literal["北向", "沪股通", "深股通"] = "北向", indicator: Literal["今日排行", "3日排行", "5日排行", "10日排行", "月排行", "季排行", "年排行"] = "月排行") -> dict:
        """
        获取指定市场的持股个股排行，并返回格式化后的结果。参数： market:str="北向" indicator="月排行"  返回值Dict[symbol,str]

        参数:
            market (str): 市场类型，选择 "北向", "沪股通", "深股通" 之一。默认值为 "北向"。
            indicator (str): 排行时间范围，选择 "今日排行", "3日排行", "5日排行", "10日排行", "月排行", "季排行", "年排行" 之一。默认值为 "月排行"。

        返回:
            dict: 键为股票代码，值为该股票的详细信息，格式化为易于阅读的字符串。

        示例:
        >>> get_top_holdings_by_market(market="沪股通", indicator="月排行")
        {'000001': '名称: 平安银行, 今日收盘价: 10.5, 今日涨跌幅: 1.2%, ...', ...}
        """
        # 获取数据
        df = ak.stock_hsgt_hold_stock_em(indicator=indicator, market=market)

        # 处理数据，将每行数据转换为易于阅读的字符串
        result = {}
        for _, row in df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"今日收盘价: {row['今日收盘价']}, "
                f"今日涨跌幅: {row['今日涨跌幅']}%, "
                f"今日持股-股数: {row['今日持股-股数']}万, "
                f"今日持股-市值: {row['今日持股-市值']}万, "
                f"今日持股-占流通股比: {row['今日持股-占流通股比']}%, "
                f"今日持股-占总股本比: {row['今日持股-占总股本比']}%, "
                f"增持估计-股数: {row['增持估计-股数']}万, "
                f"增持估计-市值: {row['增持估计-市值']}万, "
                f"增持估计-市值增幅: {row['增持估计-市值增幅']}%, "
                f"增持估计-占流通股比: {row['增持估计-占流通股比']}‰, "
                f"增持估计-占总股本比: {row['增持估计-占总股本比']}‰, "
                f"所属板块: {row['所属板块']}, "
                f"日期: {row['日期']}"
            )
            result[row['代码']] = stock_info

        return result

    @thread_safe_cache.thread_safe_cached("stock_comments_summary")
    def get_stock_comments_summary(self) -> dict:
        """
        获取东方财富网-数据中心-特色数据-千股千评数据摘要.返回值Dict[symbol,str]
        
        返回值:
            一个字典，键是股票代码，值是描述性的字符串，包含以下信息的统计：
            - 名称
            - 最新价
            - 涨跌幅
            - 换手率
            - 市盈率
            - 主力成本
            - 机构参与度
            - 综合得分
            - 上升
            - 目前排名
            - 关注指数
            - 交易日
        """
        # 检查缓存是否存在
        if "stock_comments" in self.comment_cache:
            return self.comment_cache["stock_comments"]
        
        # 获取数据
        data = ak.stock_comment_em()
        
        # 生成描述性字符串的字典
        summary_dict = {}
        for index, row in data.iterrows():
            description = (
                f"名称: {row['名称']}, "
                f"最新价: {row['最新价']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"换手率: {row['换手率']}%, "
                f"市盈率: {row['市盈率']}, "
                f"主力成本: {row['主力成本']}, "
                f"机构参与度: {row['机构参与度']}%, "
                f"综合得分: {row['综合得分']}, "
                f"上升: {row['上升']}, "
                f"目前排名: {row['目前排名']}, "
                f"关注指数: {row['关注指数']}, "
                f"交易日: {row['交易日']}"
            )
            summary_dict[row['代码']] = description
        
        # 缓存结果
        self.comment_cache["stock_comments"] = summary_dict
        
        return summary_dict

    @cache.cache(ttl=60*60*7)
    def get_stock_profit_forecast(self, symbol: str) -> str:
        """
        获取指定股票的盈利预测数据。symbol: str ,返回str 盈利预测字符串

        参数:
        symbol (str): 股票代码，例如 "600519"

        返回:
        str: 格式化的盈利预测信息字符串
        """
        if not hasattr(self, 'profit_forecast_cache'):
            self.profit_forecast_cache = {}

        if not self.profit_forecast_cache:
            try:
                df = ak.stock_profit_forecast_em()
                for _, row in df.iterrows():
                    code = row['代码']
                    forecast_info = (
                        f"名称: {row['名称']}, "
                        f"研报数: {row['研报数']}, "
                        f"机构投资评级(近六个月): 买入 {row['机构投资评级(近六个月)-买入']}%, "
                        f"增持 {row['机构投资评级(近六个月)-增持']}%, "
                        f"中性 {row['机构投资评级(近六个月)-中性']}%, "
                        f"减持 {row['机构投资评级(近六个月)-减持']}%, "
                        f"卖出 {row['机构投资评级(近六个月)-卖出']}%, "
                        f"2022预测每股收益: {row['2022预测每股收益']:.4f}, "
                        f"2023预测每股收益: {row['2023预测每股收益']:.4f}, "
                        f"2024预测每股收益: {row['2024预测每股收益']:.4f}, "
                        f"2025预测每股收益: {row['2025预测每股收益']:.4f}"
                    )
                    self.profit_forecast_cache[code] = forecast_info

            except Exception as e:
                return f"获取盈利预测数据时发生错误: {str(e)}"

        return self.profit_forecast_cache.get(symbol, f"未找到股票代码 {symbol} 的盈利预测数据")

    def _calculate_ratios(self, data: Dict) -> Dict[str, float]:
        """计算各种市场比率
        
        Args:
            data: 原始市场数据字典
            
        Returns:
            包含计算后比率的字典
        """
        total_stocks = data['up_count'] + data['down_count'] + data['flat_count']
        
        if total_stocks == 0:
            return {
                'up_down_ratio': 0.0,
                'limit_up_down_ratio': 0.0,
                'real_limit_up_down_ratio': 0.0
            }
            
        return {
            'up_down_ratio': round(data['up_count'] / total_stocks * 100, 2),
            'limit_up_down_ratio': round(data['limit_up_count'] / data['limit_down_count'] if data['limit_down_count'] != 0 else float('inf'), 2),
            'real_limit_up_down_ratio': round(data['real_limit_up_count'] / data['real_limit_down_count'] if data['real_limit_down_count'] != 0 else float('inf'), 2)
        }

    @cache.cache(ttl=60*60)
    def get_market_activity(self) -> Dict[str, Union[int, float, str]]:
        """获取市场活跃度数据
        
        Returns:
            Dict: 包含以下字段的字典：
                up_count (int): 上涨家数
                limit_up_count (int): 涨停家数
                real_limit_up_count (int): 真实涨停家数
                st_limit_up_count (int): ST股涨停家数
                down_count (int): 下跌家数
                limit_down_count (int): 跌停家数
                real_limit_down_count (int): 真实跌停家数
                st_limit_down_count (int): ST股跌停家数
                flat_count (int): 平盘家数
                suspended_count (int): 停牌家数
                activity_rate (float): 市场活跃度(%)
                timestamp (str): 统计时间
                up_down_ratio (float): 涨跌比(%)
                limit_up_down_ratio (float): 涨跌停比
                real_limit_up_down_ratio (float): 真实涨跌停比
                
        Examples:
            >>> analyzer = MarketActivityAnalyzer()
            >>> activity_data = analyzer.get_market_activity()
            >>> print(f"市场活跃度: {activity_data['activity_rate']}%")
            >>> print(f"涨跌比: {activity_data['up_down_ratio']}%")
            
        Notes:
            1. 涨跌比 = 上涨家数 / (上涨家数 + 下跌家数 + 平盘家数) * 100%
            2. 活跃度反映市场整体交投活跃程度
            3. 涨跌停比反映市场强弱程度
            4. 真实涨跌停不包含一字板
        """
        try:
            # 获取原始数据
            df = ak.stock_market_activity_legu()
            
            # 转换为字典格式，方便处理
            data = {}
            for _, row in df.iterrows():
                item = row['item']
                value = row['value']
                
                # 处理不同类型的数据
                if isinstance(value, str) and '%' in value:
                    # 处理百分比
                    data['activity_rate'] = float(value.replace('%', ''))
                elif isinstance(value, str) and ':' in value:
                    # 处理时间戳
                    data['timestamp'] = value
                else:
                    # 映射字段名称
                    field_mapping = {
                        '上涨': 'up_count',
                        '涨停': 'limit_up_count',
                        '真实涨停': 'real_limit_up_count',
                        'st st*涨停': 'st_limit_up_count',
                        '下跌': 'down_count',
                        '跌停': 'limit_down_count',
                        '真实跌停': 'real_limit_down_count',
                        'st st*跌停': 'st_limit_down_count',
                        '平盘': 'flat_count',
                        '停牌': 'suspended_count'
                    }
                    
                    if item in field_mapping:
                        data[field_mapping[item]] = float(value) if isinstance(value, (int, float)) else 0
            
            # 计算额外的比率指标
            ratios = self._calculate_ratios(data)
            data.update(ratios)
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to get market activity data: {str(e)}")

    def _format_timestamp(self, timestamp: str) -> str:
        """格式化时间戳
        
        Args:
            timestamp: 原始时间戳字符串
            
        Returns:
            格式化后的时间字符串，格式为'YYYY-MM-DD HH:MM:SS'
        """
        try:
            dt = pd.to_datetime(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    
    def _calculate_statistics(self, ranks: List[int]) -> Dict[str, Union[int, float]]:
        """计算排名统计信息
        
        Args:
            ranks: 排名列表
            
        Returns:
            包含统计信息的字典
        """
        if not ranks:
            return {
                'max_rank': None,
                'min_rank': None,
                'avg_rank': None,
                'rank_volatility': None
            }
            
        return {
            'max_rank': max(ranks),
            'min_rank': min(ranks),
            'avg_rank': round(sum(ranks) / len(ranks), 2),
            'rank_volatility': round(pd.Series(ranks).std(), 2) if len(ranks) > 1 else 0
        }
    
    def get_popularity_ranks(self, symbol: str) -> List[Dict[str, Union[str, int]]]:
        """获取个股实时人气排名数据
        
        Args:
            symbol: 股票代码，格式如'SZ000665'，其中SZ代表深圳，SH代表上海
            
        Returns:
            List[Dict]: 包含以下字段的字典列表：
                timestamp (str): 时间戳，格式为'YYYY-MM-DD HH:MM:SS'
                rank (int): 排名
                rank_change (int): 排名变化（与上一时间点相比）
                is_rising (bool): 排名是否上升（与上一时间点相比）
                
        Examples:
            >>> tracker = StockPopularityTracker()
            >>> ranks = tracker.get_popularity_ranks("SZ000665")
            >>> for rank_data in ranks[:5]:  # 显示前5条数据
            >>>     print(f"时间: {rank_data['timestamp']}, 排名: {rank_data['rank']}, "
            >>>           f"变化: {rank_data['rank_change']}")
                     
        Notes:
            1. 排名为正整数，数值越小表示人气越高
            2. rank_change为负数表示排名上升（变好），正数表示排名下降（变差）
            3. is_rising=True表示排名上升（变好），False表示排名下降或不变
        """
        try:
            # 获取原始数据
            df = ak.stock_hot_rank_detail_realtime_em(symbol=symbol)
            
            # 转换数据格式
            result = []
            prev_rank = None
            
            for _, row in df.iterrows():
                current_rank = int(row['排名'])
                timestamp = self._format_timestamp(row['时间'])
                
                # 计算排名变化
                rank_change = current_rank - prev_rank if prev_rank is not None else 0
                is_rising = rank_change < 0 if prev_rank is not None else False
                
                result.append({
                    'timestamp': timestamp,
                    'rank': current_rank,
                    'rank_change': rank_change,
                    'is_rising': is_rising
                })
                
                prev_rank = current_rank
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to get popularity rank data: {str(e)}")
    
    def get_rank_analysis(self, symbol: str) -> Dict[str, Union[Dict, str]]:
        """获取个股人气排名分析报告
        
        Args:
            symbol: 股票代码，格式如'SZ000665'
            
        Returns:
            Dict: 包含以下内容的字典：
                statistics (Dict): 统计信息
                    - max_rank: 最差排名
                    - min_rank: 最好排名
                    - avg_rank: 平均排名
                    - rank_volatility: 排名波动性（标准差）
                summary (str): 分析总结
                
        Examples:
            >>> tracker = StockPopularityTracker()
            >>> analysis = tracker.get_rank_analysis("SZ000665")
            >>> print(f"分析总结：{analysis['summary']}")
        """
        try:
            ranks_data = self.get_popularity_ranks(symbol)
            
            # 提取排名数据
            ranks = [data['rank'] for data in ranks_data]
            
            # 计算统计信息
            stats = self._calculate_statistics(ranks)
            
            # 生成分析总结
            recent_trend = "上升" if ranks_data[-1]['is_rising'] else "下降" if ranks_data[-1]['rank_change'] > 0 else "保持稳定"
            
            summary = (
                f"个股人气排名在 {stats['min_rank']} 至 {stats['max_rank']} 之间波动，"
                f"平均排名 {stats['avg_rank']}，波动性 {stats['rank_volatility']}。"
                f"最近排名呈{recent_trend}趋势。"
            )
            
            return {
                'statistics': stats,
                'summary': summary
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate rank analysis: {str(e)}")
        
    def get_analysis_summary(self) -> str:
        """生成市场活跃度分析总结
        
        Returns:
            str: 市场活跃度分析总结文字
            
        Examples:
            >>> analyzer = MarketActivityAnalyzer()
            >>> summary = analyzer.get_analysis_summary()
            >>> print(summary)
        """
        try:
            data = self.get_market_activity()
            
            activity_level = '较高' if data['activity_rate'] > 80 else '一般' if data['activity_rate'] > 50 else '较低'
            market_sentiment = '多头' if data['up_down_ratio'] > 60 else '震荡' if data['up_down_ratio'] > 40 else '空头'
            
            summary = (
                f"市场活跃度{data['activity_rate']}%，处于{activity_level}水平。\n"
                f"市场上涨{data['up_count']}家，下跌{data['down_count']}家，"
                f"涨跌比{data['up_down_ratio']}%，呈现{market_sentiment}格局。\n"
                f"涨停{data['limit_up_count']}家（其中真实涨停{data['real_limit_up_count']}家），"
                f"跌停{data['limit_down_count']}家（其中真实跌停{data['real_limit_down_count']}家）。"
            )
            
            return summary
            
        except Exception as e:
            raise Exception(f"Failed to generate analysis summary: {str(e)}")

    def _convert_value(self, value: str) -> Optional[float]:
        """将字符串格式的数值转换为浮点数
        
        Args:
            value: 需要转换的字符串值
            
        Returns:
            转换后的浮点数，如果无法转换则返回None
        """
        if isinstance(value, (int, float)):
            return float(value)
        if not value or value == '-':
            return None
        # 处理"亿"单位
        if '亿' in value:
            value = float(value.replace('亿', ''))
        # 处理百分比
        elif '%' in value:
            value = float(value.replace('%', ''))
        else:
            try:
                value = float(value)
            except ValueError:
                return None
        return value

    def _process_df_to_dict(self, df: pd.DataFrame) -> List[Dict]:
        """将DataFrame转换为字典列表
        
        Args:
            df: 需要转换的DataFrame
            
        Returns:
            转换后的字典列表
        """
        return df.to_dict('records')

    def _clean_institution_forecasts(self, df: pd.DataFrame) -> List[Dict]:
        """处理机构预测数据
        
        Args:
            df: 机构预测原始DataFrame
            
        Returns:
            处理后的机构预测数据字典列表
        """
        records = []
        for _, row in df.iterrows():
            cleaned_record = {
                'institution': row['机构名称'],
                'analyst': row['研究员'],
                'report_date': row['报告日期']
            }
            # 处理各年度预测数据
            for year in ['2022', '2023', '2024']:
                eps_key = f'预测年报每股收益{year}预测'
                profit_key = f'预测年报净利润{year}预测'
                cleaned_record[f'eps_{year}'] = self._convert_value(row[eps_key])
                profit_val = row[profit_key]
                cleaned_record[f'profit_{year}'] = self._convert_value(profit_val.replace('亿', '')) if isinstance(profit_val, str) else None
            records.append(cleaned_record)
        return records

    def _clean_detailed_indicators(self, df: pd.DataFrame) -> List[Dict]:
        """处理详细指标预测数据
        
        Args:
            df: 详细指标原始DataFrame
            
        Returns:
            处理后的详细指标数据字典列表
        """
        records = []
        for _, row in df.iterrows():
            cleaned_record = {
                'indicator': row['预测指标'],
                'actual_2019': row['2019-实际值'],
                'actual_2020': row['2020-实际值'],
                'actual_2021': row['2021-实际值'],
                'forecast_2022': row['预测2022-平均'],
                'forecast_2023': row['预测2023-平均'],
                'forecast_2024': row['预测2024-平均']
            }
            records.append(cleaned_record)
        return records

    def _convert_amount(self, value: float) -> float:
        """
        转换金额，统一单位为亿元
        
        Args:
            value: 原始金额（元）
            
        Returns:
            float: 转换后的金额（亿元）
        """
        return round(value / 100000000, 4)

    @cache.cache(ttl=60*60*2)
    def get_technical_analysis(self, stock_code: Optional[str] = None, 
                            period_high: Literal["创月新高", "半年新高", "一年新高", "历史新高"] = "创月新高",
                            period_low: Literal["创月新低", "半年新低", "一年新低", "历史新低"] = "创月新低",
                            ma_period: str = "30日均线") -> Dict[str, List[Dict[str, Any]]]:
        """
        聚合获取股票的技术分析数据，包括创新高、创新低、连续上涨、连续下跌、持续放量、持续缩量、
        向上突破、向下突破、量价齐升、量价齐跌、险资举牌等信息。

        Parameters
        ----------
        stock_code : Optional[str]
            股票代码，如果指定，则只返回该股票的数据
        period_high : Literal["创月新高", "半年新高", "一年新高", "历史新高"]
            创新高周期，默认为"创月新高"
        period_low : Literal["创月新低", "半年新低", "一年新低", "历史新低"]
            创新低周期，默认为"创月新低"
        ma_period : str
            均线周期，默认为"30日均线"

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            返回包含各类技术分析数据的字典，键为分析类型，值为对应的股票列表

        Examples
        --------
        >>> data = stock_provider.get_technical_analysis("000001")
        >>> print(data['创新高'][0])
        {'股票代码': '000001', '股票简称': '平安银行', ...}
        """
        result = {}
        
        try:
            # 创新高、创新低
            result['创新高'] = self.get_new_high_stocks(stock_code=stock_code, period=period_high)
            result['创新低'] = self.get_new_low_stocks(stock_code=stock_code, period=period_low)
            
            # 连续上涨、连续下跌
            result['连续上涨'] = self.get_continuous_rise_stocks(symbol=stock_code)
            result['连续下跌'] = self.get_continuous_decline_stocks(symbol=stock_code)
            
            # 成交量分析
            result['持续放量'] = self.get_volume_increase_stocks(symbol=stock_code)
            result['持续缩量'] = self.get_volume_shrink_stocks(symbol=stock_code)
            
            # 均线突破
            result['向上突破'] = self.get_break_up_stocks(stock_code=stock_code, ma_period=ma_period)
            result['向下突破'] = self.get_break_down_stocks(stock_code=stock_code, ma_period=ma_period)
            
            # 量价关系
            result['量价齐升'] = self.get_volume_price_rise_stocks(symbol=stock_code)
            result['量价齐跌'] = self.get_volume_price_decline_stocks(symbol=stock_code)
            
            # 险资举牌
            result['险资举牌'] = self.get_insurance_holdings_alerts(symbol=stock_code)
            
            # 移除空列表
            return {k: v for k, v in result.items() if v}
            
        except Exception as e:
            self.logger.error(f"获取技术分析数据时出错: {str(e)} {format_exc()}")
            return {}

    def get_volume_increase_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取持续放量的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含持续放量数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 涨跌幅: 当日涨跌幅(%)
                - 最新价: 当前价格(元)
                - 成交量: 当日成交股数(股)
                - 基准日成交量: 放量起始日成交股数(股)
                - 放量天数: 连续放量天数
                - 阶段涨跌幅: 放量期间涨跌幅(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_cxfl_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_new_high_stocks(self, stock_code: Optional[str] = None , period: Literal["创月新高", "半年新高", "一年新高", "历史新高"] = "创月新高") -> List[Dict[str, Any]]:
        """
        获取创新高股票列表。

        Parameters
        ----------
        period : Literal["创月新高", "半年新高", "一年新高", "历史新高"]
            创新高的时间周期，默认为"创月新高"
            - "创月新高": 月新高
            - "半年新高": 半年新高
            - "一年新高": 一年新高
            - "历史新高": 历史新高
        stock_code : Optional[str]
            股票代码，如果指定，则只返回该股票的数据。默认为None，返回所有股票。

        Returns
        -------
        List[Dict[str, Any]]
            包含创新高股票信息的列表，每个字典包含以下字段：
            - 序号: int
            - 股票代码: str
            - 股票简称: str 
            - 涨跌幅: float
            - 换手率: float
            - 最新价: float
            - 前期高点: float
            - 前期高点日期: str

        Examples
        --------
        >>> data = stock_provider.get_new_high_stocks("创月新高", "000157")
        >>> print(data[0])
        {'序号': 1, '股票代码': '000157', '股票简称': '中联重科'...}
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data(period: str) -> pd.DataFrame:
            return ak.stock_rank_cxg_ths(symbol=period)

        try:
            df = fetch_data(period)
            if stock_code:
                df = df[df['股票代码'] == stock_code]
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"获取创新高股票数据时出错: {str(e)}")
            return []

    def get_new_low_stocks(self, stock_code: Optional[str] = None , period: Literal["创月新低", "半年新低", "一年新低", "历史新低"] = "创月新低") -> List[Dict[str, Any]]:
        """
        获取创新低股票列表。

        Parameters
        ----------
        period : Literal["创月新低", "半年新低", "一年新低", "历史新低"]
            创新低的时间周期，默认为"创月新低"
            - "创月新低": 月新低
            - "半年新低": 半年新低
            - "一年新低": 一年新低
            - "历史新低": 历史新低
        stock_code : Optional[str]
            股票代码，如果指定，则只返回该股票的数据。默认为None，返回所有股票。

        Returns
        -------
        List[Dict[str, Any]]
            包含创新低股票信息的列表，每个字典包含以下字段：
            - 序号: int
            - 股票代码: str
            - 股票简称: str 
            - 涨跌幅: float
            - 换手率: float
            - 最新价: float
            - 前期低点: float
            - 前期低点日期: str

        Examples
        --------
        >>> data = stock_provider.get_new_low_stocks("创月新低", "000004")
        >>> print(data[0])
        {'序号': 1, '股票代码': '000004', '股票简称': '国华网安'...}
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data(period: str) -> pd.DataFrame:
            return ak.stock_rank_cxd_ths(symbol=period)

        try:
            df = fetch_data(period)
            if stock_code:
                df = df[df['股票代码'] == stock_code]
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"获取创新低股票数据时出错: {str(e)}")
            return []

    def get_continuous_rise_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取连续上涨的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含连续上涨数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 收盘价: 当前收盘价(元)
                - 最高价: 阶段最高价(元)
                - 最低价: 阶段最低价(元)
                - 连涨天数: 连续上涨天数
                - 连续涨跌幅: 上涨阶段涨跌幅(%)
                - 累计换手率: 上涨阶段累计换手率(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_lxsz_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_continuous_decline_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取连续下跌的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含连续下跌数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 收盘价: 当前收盘价(元)
                - 最高价: 阶段最高价(元)
                - 最低价: 阶段最低价(元)
                - 连涨天数: 连续下跌天数
                - 连续涨跌幅: 下跌阶段涨跌幅(%)
                - 累计换手率: 下跌阶段累计换手率(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_lxxd_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_volume_shrink_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取持续缩量的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含持续缩量数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称 
                - 涨跌幅: 当日涨跌幅(%)
                - 最新价: 当前价格(元)
                - 成交量: 当日成交股数(股)
                - 基准日成交量: 缩量起始日成交股数(股)
                - 缩量天数: 连续缩量天数
                - 阶段涨跌幅: 缩量期间涨跌幅(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_cxsl_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_break_up_stocks(self, stock_code: str = None , ma_period: str = "30日均线") -> List[Dict[str, Any]]:
        """
        获取向上突破均线的股票数据。
        
        参数:
            ma_period (str): 均线周期,可选值:"5日均线","10日均线","20日均线","30日均线",
                            "60日均线","90日均线","250日均线","500日均线"
            stock_code (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含向上突破数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码 
                - 股票简称: 标的名称
                - 最新价: 当前价格(元)
                - 成交额: 成交金额(元)
                - 成交量: 成交股数(股)  
                - 涨跌幅: 价格涨跌幅(%)
                - 换手率: 换手率(%)
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data(period: str):
            try:
                df = ak.stock_rank_xstp_ths(symbol=period)
            except:
                return []
            return df.to_dict(orient='records')
        
        data = fetch_data(ma_period)
        if stock_code:
            data = [item for item in data if item['股票代码'] == stock_code]
            
        return data

    def get_break_down_stocks(self, stock_code: str = None , ma_period: str = "30日均线") -> List[Dict[str, Any]]:
        """
        获取向下突破均线的股票数据。
        
        参数:
            ma_period (str): 均线周期,可选值:"5日均线","10日均线","20日均线","30日均线",
                            "60日均线","90日均线","250日均线","500日均线"
            stock_code (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含向下突破数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码 
                - 股票简称: 标的名称
                - 最新价: 当前价格(元)
                - 成交额: 成交金额(元)
                - 成交量: 成交股数(股)  
                - 涨跌幅: 价格涨跌幅(%)
                - 换手率: 换手率(%)
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data(period: str):
            df = ak.stock_rank_xxtp_ths(symbol=period)
            return df.to_dict(orient='records')
        
        data = fetch_data(ma_period)
        if stock_code:
            data = [item for item in data if item['股票代码'] == stock_code]
            
        return data

    def get_volume_price_rise_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取量价齐升的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含量价齐升数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 最新价: 当前价格(元)
                - 量价齐升天数: 连续量价齐升的天数
                - 阶段涨幅: 量价齐升期间的涨跌幅(%) 
                - 累计换手率: 量价齐升期间的累计换手率(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_ljqs_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_volume_price_decline_stocks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取量价齐跌的股票数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含量价齐跌数据的字典列表,每个字典包含:
                - 序号: 编号
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 最新价: 当前价格(元)
                - 量价齐跌天数: 连续量价齐跌的天数
                - 阶段涨幅: 量价齐跌期间的涨跌幅(%) 
                - 累计换手率: 量价齐跌期间的累计换手率(%)
                - 所属行业: 股票所属行业
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_ljqd_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_insurance_holdings_alerts(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        获取险资举牌数据。
        
        参数:
            symbol (str, optional): 股票代码,如果传入则只返回该股票的数据
            
        返回:
            List[Dict[str, Any]]: 包含险资举牌数据的字典列表,每个字典包含:
                - 序号: 编号 
                - 举牌公告日: 公告日期
                - 股票代码: 标的代码
                - 股票简称: 标的名称
                - 现价: 当前价格(元)
                - 涨跌幅: 涨跌幅(%)
                - 举牌方: 举牌机构名称
                - 增持数量: 增持股数(股)
                - 交易均价: 增持均价(元)
                - 增持数量占总股本比例: 增持比例(%)
                - 变动后持股总数: 持股总数(股)
                - 变动后持股比例: 最新持股比例(%)
        """
        @cache.cache(ttl=60*60*2)
        def fetch_data():
            df = ak.stock_rank_xzjp_ths()
            return df.to_dict(orient='records')
        
        data = fetch_data()
        if symbol:
            data = [item for item in data if item['股票代码'] == symbol]
            
        return data

    def get_institutional_trading(self, start_date: str, end_date: str) -> Dict[str, Union[List[Dict], Dict]]:
        """
        获取机构买卖每日统计数据
        
        Args:
            start_date: 起始日期，格式为'YYYYMMDD'
            end_date: 结束日期，格式为'YYYYMMDD'
            
        Returns:
            Dict: 包含以下内容：
                transactions: List[Dict] - 交易记录列表
                    - stock_code: str - 股票代码
                    - stock_name: str - 股票名称
                    - close_price: float - 收盘价
                    - change_percent: float - 涨跌幅(%)
                    - buy_inst_count: int - 买方机构数
                    - sell_inst_count: int - 卖方机构数
                    - buy_amount: float - 机构买入总额(亿元)
                    - sell_amount: float - 机构卖出总额(亿元)
                    - net_amount: float - 机构买入净额(亿元)
                    - total_amount: float - 市场总成交额(亿元)
                    - net_ratio: float - 机构净买额占总成交额比(%)
                    - turnover_rate: float - 换手率(%)
                    - floating_mv: float - 流通市值(亿元)
                    - list_reason: str - 上榜原因
                    - trade_date: str - 上榜日期
                summary: Dict - 汇总数据
                    - total_stocks: int - 上榜股票数
                    - total_buy_amount: float - 总买入金额(亿元)
                    - total_sell_amount: float - 总卖出金额(亿元)
                    - total_net_amount: float - 总净买入金额(亿元)
                    - avg_turnover_rate: float - 平均换手率(%)
                
        Examples:
            >>> analyzer = StockTraderAnalyzer()
            >>> data = analyzer.get_institutional_trading("20240417", "20240430")
            >>> print(f"期间共有{data['summary']['total_stocks']}只股票上榜")
            >>> print(f"机构净买入{data['summary']['total_net_amount']}亿元")
        """
        try:
            df = ak.stock_lhb_jgmmtj_em(start_date=start_date, end_date=end_date)
            
            transactions = []
            for _, row in df.iterrows():
                transactions.append({
                    'stock_code': row['代码'],
                    'stock_name': row['名称'],
                    'close_price': float(row['收盘价']),
                    'change_percent': float(row['涨跌幅']),
                    'buy_inst_count': int(row['买方机构数']),
                    'sell_inst_count': int(row['卖方机构数']),
                    'buy_amount': self._convert_amount(row['机构买入总额']),
                    'sell_amount': self._convert_amount(row['机构卖出总额']),
                    'net_amount': self._convert_amount(row['机构买入净额']),
                    'total_amount': self._convert_amount(row['市场总成交额']),
                    'net_ratio': float(row['机构净买额占总成交额比']),
                    'turnover_rate': float(row['换手率']),
                    'floating_mv': float(row['流通市值']),
                    'list_reason': row['上榜原因'],
                    'trade_date': row['上榜日期']
                })
                
            summary = {
                'total_stocks': len(transactions),
                'total_buy_amount': sum(t['buy_amount'] for t in transactions),
                'total_sell_amount': sum(t['sell_amount'] for t in transactions),
                'total_net_amount': sum(t['net_amount'] for t in transactions),
                'avg_turnover_rate': sum(t['turnover_rate'] for t in transactions) / len(transactions)
            }
            
            return {
                'transactions': transactions,
                'summary': summary
            }
            
        except Exception as e:
            raise Exception(f"Failed to get institutional trading data: {str(e)}")

    def get_institutional_tracking(self, period: str = "近一月") -> Dict[str, Union[List[Dict], Dict]]:
        """
        获取机构席位追踪数据
        
        Args:
            period: 统计周期，可选值："近一月", "近三月", "近六月", "近一年"
            
        Returns:
            Dict: 包含以下内容：
                stocks: List[Dict] - 股票列表
                    - stock_code: str - 股票代码
                    - stock_name: str - 股票名称
                    - close_price: float - 收盘价
                    - change_percent: float - 涨跌幅(%)
                    - total_amount: float - 龙虎榜成交金额(亿元)
                    - list_count: int - 上榜次数
                    - buy_amount: float - 机构买入额(亿元)
                    - buy_count: int - 机构买入次数
                    - sell_amount: float - 机构卖出额(亿元)
                    - sell_count: int - 机构卖出次数
                    - net_amount: float - 机构净买额(亿元)
                    - returns: Dict - 各期间涨跌幅
                        - 1m: float - 近1个月涨跌幅(%)
                        - 3m: float - 近3个月涨跌幅(%)
                        - 6m: float - 近6个月涨跌幅(%)
                        - 1y: float - 近1年涨跌幅(%)
                statistics: Dict - 统计数据
                    - total_stocks: int - 统计股票数
                    - total_buy_amount: float - 总买入金额(亿元)
                    - total_sell_amount: float - 总卖出金额(亿元)
                    - total_net_amount: float - 总净买入金额(亿元)
                    - avg_list_count: float - 平均上榜次数
        """
        try:
            df = ak.stock_lhb_jgstatistic_em(symbol=period)
            
            stocks = []
            for _, row in df.iterrows():
                stocks.append({
                    'stock_code': row['代码'],
                    'stock_name': row['名称'],
                    'close_price': float(row['收盘价']),
                    'change_percent': float(row['涨跌幅']),
                    'total_amount': self._convert_amount(row['龙虎榜成交金额']),
                    'list_count': int(row['上榜次数']),
                    'buy_amount': self._convert_amount(row['机构买入额']),
                    'buy_count': int(row['机构买入次数']),
                    'sell_amount': self._convert_amount(row['机构卖出额']),
                    'sell_count': int(row['机构卖出次数']),
                    'net_amount': self._convert_amount(row['机构净买额']),
                    'returns': {
                        '1m': float(row['近1个月涨跌幅']),
                        '3m': float(row['近3个月涨跌幅']),
                        '6m': float(row['近6个月涨跌幅']),
                        '1y': float(row['近1年涨跌幅'])
                    }
                })
            
            statistics = {
                'total_stocks': len(stocks),
                'total_buy_amount': sum(s['buy_amount'] for s in stocks),
                'total_sell_amount': sum(s['sell_amount'] for s in stocks),
                'total_net_amount': sum(s['net_amount'] for s in stocks),
                'avg_list_count': sum(s['list_count'] for s in stocks) / len(stocks)
            }
            
            return {
                'stocks': stocks,
                'statistics': statistics
            }
            
        except Exception as e:
            raise Exception(f"Failed to get institutional tracking data: {str(e)}")

    def get_active_traders(self, start_date: str, end_date: str) -> Dict[str, Union[List[Dict], Dict]]:
        """
        获取每日活跃营业部数据
        
        Args:
            start_date: 起始日期，格式为'YYYYMMDD'
            end_date: 结束日期，格式为'YYYYMMDD'
            
        Returns:
            Dict: 包含以下内容：
                traders: List[Dict] - 营业部列表
                    - name: str - 营业部名称
                    - trade_date: str - 上榜日期
                    - buy_stock_count: int - 买入个股数
                    - sell_stock_count: int - 卖出个股数
                    - buy_amount: float - 买入总金额(亿元)
                    - sell_amount: float - 卖出总金额(亿元)
                    - net_amount: float - 总买卖净额(亿元)
                    - buy_stocks: List[str] - 买入股票列表
                summary: Dict - 统计数据
                    - total_traders: int - 活跃营业部数量
                    - total_buy_amount: float - 总买入金额(亿元)
                    - total_sell_amount: float - 总卖出金额(亿元)
                    - total_net_amount: float - 总净买入金额(亿元)
                    - avg_buy_stock_count: float - 平均买入个股数
        """
        try:
            df = ak.stock_lhb_hyyyb_em(start_date=start_date, end_date=end_date)
            
            traders = []
            for _, row in df.iterrows():
                traders.append({
                    'name': row['营业部名称'],
                    'trade_date': row['上榜日'],
                    'buy_stock_count': float(row['买入个股数']),
                    'sell_stock_count': float(row['卖出个股数']),
                    'buy_amount': self._convert_amount(row['买入总金额']),
                    'sell_amount': self._convert_amount(row['卖出总金额']),
                    'net_amount': self._convert_amount(row['总买卖净额']),
                    'buy_stocks': row['买入股票'].split() if isinstance(row['买入股票'], str) else []
                })
            
            summary = {
                'total_traders': len(traders),
                'total_buy_amount': sum(t['buy_amount'] for t in traders),
                'total_sell_amount': sum(t['sell_amount'] for t in traders),
                'total_net_amount': sum(t['net_amount'] for t in traders),
                'avg_buy_stock_count': sum(t['buy_stock_count'] for t in traders) / len(traders)
            }
            
            return {
                'traders': traders,
                'summary': summary
            }
            
        except Exception as e:
            raise Exception(f"Failed to get active traders data: {str(e)}")

    def get_trader_ranking(self, period: str = "近一月") -> Dict[str, Union[List[Dict], Dict]]:
        """
        获取营业部排行数据
        
        Args:
            period: 统计周期，可选值："近一月", "近三月", "近六月", "近一年"
            
        Returns:
            Dict: 包含以下内容：
                traders: List[Dict] - 营业部列表
                    - name: str - 营业部名称
                    - performance: Dict - 不同时间段表现
                        - 1d: Dict - 上榜后1天表现
                            - buy_count: int - 买入次数
                            - avg_return: float - 平均涨幅(%)
                            - win_rate: float - 上涨概率(%)
                        - 2d, 3d, 4d, 10d: 同上
                statistics: Dict - 统计数据
                    - total_traders: int - 统计营业部数量
                    - avg_win_rates: Dict - 各时间段平均胜率
                    - avg_returns: Dict - 各时间段平均收益率
        """
        try:
            df = ak.stock_lhb_yybph_em(symbol=period)
            
            traders = []
            for _, row in df.iterrows():
                traders.append({
                    'name': row['营业部名称'],
                    'performance': {
                        '1d': {
                            'buy_count': int(row['上榜后1天-买入次数']),
                            'avg_return': float(row['上榜后1天-平均涨幅']),
                            'win_rate': float(row['上榜后1天-上涨概率'])
                        },
                        '2d': {
                            'buy_count': int(row['上榜后2天-买入次数']),
                            'avg_return': float(row['上榜后2天-平均涨幅']),
                            'win_rate': float(row['上榜后2天-上涨概率'])
                        },
                        '3d': {
                            'buy_count': int(row['上榜后3天-买入次数']),
                            'avg_return': float(row['上榜后3天-平均涨幅']),
                            'win_rate': float(row['上榜后3天-上涨概率'])
                        },
                        '4d': {
                            'buy_count': int(row['上榜后4天-买入次数']),
                            'avg_return': float(row['上榜后4天-平均涨幅']),
                            'win_rate': float(row['上榜后4天-上涨概率'])
                        },
                        '10d': {
                            'buy_count': int(row['上榜后10天-买入次数']),
                            'avg_return': float(row['上榜后10天-平均涨幅']),
                            'win_rate': float(row['上榜后10天-上涨概率'])
                        }
                    }
                })
            
            # 计算统计数据
            periods = ['1d', '2d', '3d', '4d', '10d']
            avg_win_rates = {}
            avg_returns = {}
            
            for period in periods:
                valid_traders = [t for t in traders if t['performance'][period]['buy_count'] > 0]
                if valid_traders:
                    avg_win_rates[period] = sum(t['performance'][period]['win_rate'] for t in valid_traders) / len(valid_traders)
                    avg_returns[period] = sum(t['performance'][period]['avg_return'] for t in valid_traders) / len(valid_traders)
                else:
                    avg_win_rates[period] = 0
                    avg_returns[period] = 0
            
            statistics = {
                'total_traders': len(traders),
                'avg_win_rates': avg_win_rates,
                'avg_returns': avg_returns
            }
            
            return {
                'traders': traders,
                'statistics': statistics
            }
            
        except Exception as e:
            raise Exception(f"Failed to get trader ranking data: {str(e)}")

    def get_trader_statistics(self, period: str = "近一月") -> Dict[str, Union[List[Dict], Dict]]:
        """
        获取营业部统计数据
        
        Args:
            period: 统计周期，可选值："近一月", "近三月", "近六月", "近一年"
            
        Returns:
            Dict: 包含以下内容：
                traders: List[Dict] - 营业部列表
                    - name: str - 营业部名称
                    - total_amount: float - 龙虎榜成交金额(亿元)
                    - list_count: int - 上榜次数
                    - buy_amount: float - 买入额(亿元)
                    - buy_count: int - 买入次数
                    - sell_amount: float - 卖出额(亿元)
                    - sell_count: int - 卖出次数
                statistics: Dict - 统计数据
                    - total_traders: int - 统计营业部数量
                    - total_transaction_amount: float - 总成交金额(亿元)
                    - total_buy_amount: float - 总买入金额(亿元)
                    - total_sell_amount: float - 总卖出金额(亿元)
                    - avg_list_count: float - 平均上榜次数
                    
        Examples:
            >>> analyzer = StockTraderAnalyzer()
            >>> stats = analyzer.get_trader_statistics("近一月")
            >>> print(f"共有{stats['statistics']['total_traders']}个活跃营业部")
            >>> print(f"总成交金额: {stats['statistics']['total_transaction_amount']}亿元")
        """
        try:
            df = ak.stock_lhb_traderstatistic_em(symbol=period)
            
            traders = []
            for _, row in df.iterrows():
                traders.append({
                    'name': row['营业部名称'],
                    'total_amount': self._convert_amount(row['龙虎榜成交金额']),
                    'list_count': int(row['上榜次数']),
                    'buy_amount': self._convert_amount(row['买入额']),
                    'buy_count': int(row['买入次数']),
                    'sell_amount': self._convert_amount(row['卖出额']),
                    'sell_count': int(row['卖出次数'])
                })
            
            statistics = {
                'total_traders': len(traders),
                'total_transaction_amount': sum(t['total_amount'] for t in traders),
                'total_buy_amount': sum(t['buy_amount'] for t in traders),
                'total_sell_amount': sum(t['sell_amount'] for t in traders),
                'avg_list_count': sum(t['list_count'] for t in traders) / len(traders) if traders else 0
            }
            
            return {
                'traders': traders,
                'statistics': statistics
            }
            
        except Exception as e:
            raise Exception(f"Failed to get trader statistics: {str(e)}")

    def _convert_amount(self, value: float) -> float:
        """
        转换金额，统一单位为万元
        
        Args:
            value: 原始金额（元）
            
        Returns:
            float: 转换后的金额（万元）
        """
        return round(value / 10000, 4)
    
    def _process_stock_data(self, row: pd.Series) -> Dict:
        """
        处理A股大宗交易数据
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            Dict: 处理后的交易数据字典
        """
        return {
            'trade_date': row['交易日期'],
            'stock_code': row['证券代码'],
            'stock_name': row['证券简称'],
            'change_percent': float(row['涨跌幅']),
            'close_price': float(row['收盘价']),
            'trade_price': float(row['成交价']),
            'premium_discount': float(row['折溢率']),
            'volume': int(row['成交量']),
            'amount': self._convert_amount(float(row['成交额'])),
            'amount_to_float_mv': float(row['成交额/流通市值']),
            'buyer': row['买方营业部'],
            'seller': row['卖方营业部']
        }
    
    def _process_other_data(self, row: pd.Series) -> Dict:
        """
        处理B股、基金、债券大宗交易数据
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            Dict: 处理后的交易数据字典
        """
        return {
            'trade_date': row['交易日期'],
            'code': row['证券代码'],
            'name': row['证券简称'],
            'trade_price': float(row['成交价']),
            'volume': int(row['成交量']),
            'amount': self._convert_amount(float(row['成交额'])),
            'buyer': row['买方营业部'],
            'seller': row['卖方营业部']
        }

    def get_block_trades(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Union[str, float, int]]]:
        """
        获取大宗交易数据
        
        Args:
            symbol: 证券类型，可选值：'A股'、'B股'、'基金'、'债券'
            start_date: 开始日期，格式：'YYYYMMDD'
            end_date: 结束日期，格式：'YYYYMMDD'
            
        Returns:
            List[Dict]: 大宗交易记录列表，具体字段根据证券类型不同：
                A股：
                    - trade_date: str - 交易日期
                    - stock_code: str - 证券代码
                    - stock_name: str - 证券简称
                    - change_percent: float - 涨跌幅(%)
                    - close_price: float - 收盘价
                    - trade_price: float - 成交价
                    - premium_discount: float - 折溢率(%)
                    - volume: int - 成交量(股)
                    - amount: float - 成交额(万元)
                    - amount_to_float_mv: float - 成交额占流通市值比例(%)
                    - buyer: str - 买方营业部
                    - seller: str - 卖方营业部
                
                B股/基金/债券：
                    - trade_date: str - 交易日期
                    - code: str - 证券代码
                    - name: str - 证券简称
                    - trade_price: float - 成交价
                    - volume: int - 成交量(股)
                    - amount: float - 成交额(万元)
                    - buyer: str - 买方营业部
                    - seller: str - 卖方营业部
                    
        Examples:
            >>> analyzer = BlockTradeAnalyzer()
            >>> # 获取A股大宗交易数据
            >>> trades = analyzer.get_block_trades('A股', '20240417', '20240430')
            >>> print(f"共有{len(trades)}笔大宗交易")
            >>> # 查看第一笔交易
            >>> first_trade = trades[0]
            >>> print(f"交易日期：{first_trade['trade_date']}")
            >>> print(f"成交金额：{first_trade['amount']}万元")
        
        Raises:
            ValueError: 当symbol参数不在允许的范围内时
            Exception: 获取数据失败时
        """
        if symbol not in ['A股', 'B股', '基金', '债券']:
            raise ValueError("symbol must be one of 'A股', 'B股', '基金', '债券'")
            
        try:
            # 获取原始数据
            df = ak.stock_dzjy_mrmx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # 根据证券类型处理数据
            trades = []
            if symbol == 'A股':
                for _, row in df.iterrows():
                    trades.append(self._process_stock_data(row))
            else:
                for _, row in df.iterrows():
                    trades.append(self._process_other_data(row))
                    
            return trades
            
        except Exception as e:
            raise Exception(f"Failed to get block trade data: {str(e)}")
    
    def get_summary(self, trades: List[Dict]) -> Dict:
        """
        生成大宗交易数据统计摘要
        
        Args:
            trades: 大宗交易数据列表
            
        Returns:
            Dict: 统计摘要，包含：
                - total_trades: int - 交易总笔数
                - total_amount: float - 总成交金额(万元)
                - avg_amount: float - 平均每笔交易金额(万元)
                - total_volume: int - 总成交量(股)
                - institution_count: int - 机构专用交易笔数
                - premium_discount_avg: float - 平均溢价率(%)，仅A股数据包含
        """
        if not trades:
            return {
                'total_trades': 0,
                'total_amount': 0,
                'avg_amount': 0,
                'total_volume': 0,
                'institution_count': 0,
                'premium_discount_avg': 0
            }
            
        total_amount = sum(t['amount'] for t in trades)
        total_volume = sum(t['volume'] for t in trades)
        institution_count = sum(
            1 for t in trades 
            if t.get('buyer', '') == '机构专用' or t.get('seller', '') == '机构专用'
        )
        
        summary = {
            'total_trades': len(trades),
            'total_amount': round(total_amount, 2),
            'avg_amount': round(total_amount / len(trades), 2),
            'total_volume': total_volume,
            'institution_count': institution_count
        }
        
        # 如果是A股数据，添加溢价率统计
        if 'premium_discount' in trades[0]:
            premium_discount_avg = sum(
                t['premium_discount'] for t in trades
            ) / len(trades)
            summary['premium_discount_avg'] = round(premium_discount_avg, 2)
            
        return summary

    @cache.cache(ttl=60*60*24*7)
    def get_stock_profit_forecasts(self, symbol: str) -> List[Dict[str, Union[str, float, int]]]:
        """获取股票的盈利预测数据，包括每股收益预测、净利润预测、机构预测和详细指标预测。
        
        Args:
            symbol (str): 股票代码，如 "600519"
            
        Returns:
            List[Dict]: 包含以下4个元素的列表：
                1. eps_forecast: 每股收益预测数据
                    - year (str): 预测年度
                    - analyst_count (int): 预测机构数
                    - min_value (float): 最小预测值
                    - avg_value (float): 平均预测值
                    - max_value (float): 最大预测值
                    - industry_avg (float): 行业平均值
                    
                2. profit_forecast: 净利润预测数据
                    - year (str): 预测年度
                    - analyst_count (int): 预测机构数
                    - min_value (float): 最小预测值(亿元)
                    - avg_value (float): 平均预测值(亿元)
                    - max_value (float): 最大预测值(亿元)
                    - industry_avg (float): 行业平均值(亿元)
                    
                3. institution_forecasts: 机构预测明细数据
                    - institution (str): 机构名称
                    - analyst (str): 研究员
                    - eps_2022 (float): 2022年每股收益预测
                    - eps_2023 (float): 2023年每股收益预测
                    - eps_2024 (float): 2024年每股收益预测
                    - profit_2022 (float): 2022年净利润预测(亿元)
                    - profit_2023 (float): 2023年净利润预测(亿元)
                    - profit_2024 (float): 2024年净利润预测(亿元)
                    - report_date (str): 报告日期
                    
                4. detailed_indicators: 详细财务指标预测数据
                    - indicator (str): 预测指标名称
                    - actual_2019 (Union[str, float]): 2019年实际值
                    - actual_2020 (Union[str, float]): 2020年实际值
                    - actual_2021 (Union[str, float]): 2021年实际值
                    - forecast_2022 (Union[str, float]): 2022年预测均值
                    - forecast_2023 (Union[str, float]): 2023年预测均值
                    - forecast_2024 (Union[str, float]): 2024年预测均值
        
        Examples:
            >>> forecast = StockProfitForecast()
            >>> forecasts = forecast.get_stock_profit_forecasts("600519")
            >>> eps_forecast = forecasts[0]
            >>> print(f"2023年每股收益预测：{eps_forecast['avg_value']}")
            >>> institution_forecasts = forecasts[2]
            >>> print(f"最新机构预测：{institution_forecasts[0]['institution']}")
        
        Note:
            1. 所有金额单位均为亿元，增长率单位为百分比
            2. 部分历史数据可能为空或显示为'-'
            3. 预测数据仅供参考，不构成投资建议
        """
        try:
            # 获取每股收益预测
            eps_df = ak.stock_profit_forecast_ths(symbol=symbol, indicator="预测年报每股收益")
            eps_forecast = self._process_df_to_dict(eps_df)

            # 获取净利润预测
            profit_df = ak.stock_profit_forecast_ths(symbol=symbol, indicator="预测年报净利润")
            profit_forecast = self._process_df_to_dict(profit_df)

            # 获取机构预测
            institution_df = ak.stock_profit_forecast_ths(symbol=symbol, indicator="业绩预测详表-机构")
            institution_forecasts = self._clean_institution_forecasts(institution_df)

            # 获取详细指标预测
            indicators_df = ak.stock_profit_forecast_ths(symbol=symbol, indicator="业绩预测详表-详细指标预测")
            detailed_indicators = self._clean_detailed_indicators(indicators_df)

            return [eps_forecast, profit_forecast, institution_forecasts, detailed_indicators]

        except Exception as e:
            raise Exception(f"Failed to get profit forecast data: {str(e)}")

    def get_stock_comments_dataframe(self)->pd.DataFrame:
        """
        千股千评。返回DataFrame
        返回值：
            名称	类型	描述
            序号	int64	-
            代码	object	-
            名称	object	-
            最新价	float64	-
            涨跌幅	float64	-
            换手率	float64	注意单位: %
            市盈率	float64	-
            主力成本	float64	-
            机构参与度	float64	-
            综合得分	float64	-
            上升	int64	注意: 正负号
            目前排名	int64	-
            关注指数	float64	-
            交易日	float64	-
        """
        return ak.stock_comment_em()

    def get_main_business_description(self, symbol: str) -> str:
        """
        获取同花顺-主营介绍的数据，并返str
        
        输入参数:
            symbol (str): 股票代码
            
        返回值:
            一个描述性的字符串，包含以下信息的统计：
            - 股票代码
            - 主营业务
            - 产品类型
            - 产品名称
            - 经营范围
        """
        # 获取数据
        data = ak.stock_zyjs_ths(symbol=symbol)
        
        if data.empty:
            return f"未找到股票代码 {symbol} 的主营介绍数据。"
        
        row = data.iloc[0]
        description = (
            f"股票代码: {row['股票代码']}\n"
            f"主营业务: {row['主营业务']}\n"
            f"产品类型: {row['产品类型']}\n"
            f"产品名称: {row['产品名称']}\n"
            f"经营范围: {row['经营范围']}"
        )
        
        return description

    @cache.cache(ttl=60*60*24*7)
    def get_mainbussiness_more(self,symbol)->pd.DataFrame:
        """
        主营构成 返回DataFrame
        输入参数:
            symbol:str  股票代码
        返回值:
            名称	类型	描述
            股票代码	object	-
            报告日期	object	-
            分类类型	object	-
            主营构成	int64	-
            主营收入	float64	注意单位: 元
            收入比例	float64	-
            主营成本	float64	注意单位: 元
            成本比例	float64	-
            主营利润	float64	注意单位: 元
            利润比例	float64	-
            毛利率	float64	-
        """
        return ak.stock_zygc_em(symbol=symbol)

    def get_mainbussiness_mid(self,symbol:str)->pd.DataFrame:
        """
        主营构成.返回DataFrame
        输入参数:
            symbol:str  股票代码
        返回值:
            名称	类型	描述
            报告期	object	-
            内容	object	-
        """
        return ak.stock_zygc_ym(symbol=symbol)

    def get_manager_talk(self,symbol:str)->pd.DataFrame:
        """
        管理层讨论与分析.返回DataFrame
        输入参数:
            symbol:str  股票代码
        返回值:
            名称	类型	描述
            报告期	object	-
            内容	object	-
        """
        return ak.stock_mda_ym(symbol)

    @retry( wait=wait_exponential(multiplier=1, min=4, max=10))  
    def get_historical_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        日线数据 参数symbol: str, start_date: str, end_date: st  返回DataFrame
        返回值：Dict[symbol,list]
        list=名称	类型	描述
            日期	object	交易日
            股票代码	object	不带市场标识的股票代码
            开盘	float64	开盘价
            收盘	float64	收盘价
            最高	float64	最高价
            最低	float64	最低价
            成交量	int64	注意单位: 手
            成交额	float64	注意单位: 元
            振幅	float64	注意单位: %
            涨跌幅	float64	注意单位: %
            涨跌额	float64	注意单位: 元
            换手率	float64	注意单位: %
        """
        return ak.stock_zh_a_hist(symbol=symbol,period="daily", start_date=start_date, end_date=end_date)

    def format_stock_data(self,stock_data):
        formatted_output = []
        
        for code, data in stock_data.items():
            formatted_output.append(f"股票代码: {code}")
            formatted_output.append(data.replace(", ", "\n"))
            formatted_output.append("\n" + "="*50 + "\n")
        
        return "\n".join(formatted_output)

    def df_to_full_string(self,df: pd.DataFrame,index=False) -> str:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            return df.to_string(index=index)

    def df_columns_to_string(self,df: pd.DataFrame, columns: str, include_index: bool = False) -> str:
        # 将输入的列名字符串转换为列表
        column_list = columns.split(',')

        # 检查列名是否在 DataFrame 中
        for col in column_list:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' does not exist in DataFrame.")
        
        # 选择要输出的列，并转换为字符串
        if include_index:
            selected_df = df[column_list]
        else:
            selected_df = df[column_list].reset_index(drop=True)
        
        return selected_df.to_string(index=include_index)

    def generate_close_summary(self,df: pd.DataFrame) -> str:
        # 定义时间段和各时间段对应的数据量
        time_frames = {
            '5分钟': 5,
            '10分钟': 10,
            '15分钟': 15,
            '30分钟': 30,
            '60分钟': 60,
            '120分钟': 120,
            '240分钟': 240
        }

        output_lines = []

        for label, periods in time_frames.items():
            # 提取最后 periods 行的 close 数据
            close_data = df['close'].tail(periods).tolist()
            # 将数据转换为字符串，并用空格分隔
            close_data_str = " ".join(map(str, close_data))
            # 添加到输出结果
            output_lines.append(f"最后{label}close数据:\n{close_data_str}\n")

        return "\n".join(output_lines)

    def get_stock_account_statistics(self)->List[Dict]:
        """
        获取账户统计数据。

        返回：
        Dict[str, Any]: 包含股票基础数据的字典
        """
        stock_account_statistics_em_df = ak.stock_account_statistics_em()
        stock_account_statistics_em_df = stock_account_statistics_em_df.tail(10)
        return stock_account_statistics_em_df.to_dict(orient="records")

    def get_stock_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定股票的基础数据。

        参数：
        symbol (str): 股票代码

        返回：
        Dict[str, Any]: 包含股票基础数据的字典
        """
        # 使用 ak.stock_fundamentals_em 函数获取基础数据
        fundamentals = ak.stock_zh_a_spot_em()

        fundamentals = fundamentals[fundamentals["代码"]==symbol]

        # 将数据转换为字典
        fundamentals_dict = fundamentals.to_dict(orient="records")

        return fundamentals_dict

    def get_risk_warning_stocks(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有风险警示板股票的交易数据，使用线程安全的缓存机制。
        
        数据包含：
        - 基本信息：代码、名称
        - 交易数据：最新价、涨跌幅、成交量等
        - 市场指标：量比、换手率、市盈率、市净率等
        
        返回:
            Dict[str, Dict[str, Any]]: 以股票代码为键的字典，每个股票包含以下字段：
                - code (str): 股票代码
                - name (str): 股票名称
                - price (float): 最新价
                - change_percent (float): 涨跌幅(%)
                - change_amount (float): 涨跌额
                - volume (float): 成交量
                - amount (float): 成交额
                - amplitude (float): 振幅(%)
                - high (float): 最高价
                - low (float): 最低价
                - open (float): 开盘价
                - pre_close (float): 昨收价
                - volume_ratio (float): 量比
                - turnover_rate (float): 换手率(%)
                - pe_ratio (float): 市盈率(动态)
                - pb_ratio (float): 市净率
        """
        @thread_safe_cache.thread_safe_cached("risk_warning_stocks")
        def fetch_risk_warning_stocks() -> Dict[str, Dict[str, Any]]:
            try:
                df = ak.stock_zh_a_st_em()
                result = {}
                
                for _, row in df.iterrows():
                    result[row['代码']] = {
                        'code': row['代码'],
                        'name': row['名称'],
                        'price': row['最新价'],
                        'change_percent': row['涨跌幅'],
                        'change_amount': row['涨跌额'],
                        'volume': row['成交量'],
                        'amount': row['成交额'],
                        'amplitude': row['振幅'],
                        'high': row['最高'],
                        'low': row['最低'],
                        'open': row['今开'],
                        'pre_close': row['昨收'],
                        'volume_ratio': row['量比'],
                        'turnover_rate': row['换手率'],
                        'pe_ratio': row['市盈率-动态'],
                        'pb_ratio': row['市净率']
                    }
                
                return result
            except Exception as e:
                self.logger.error(f"获取风险警示股票数据时出错: {str(e)}")
                return {}

        return fetch_risk_warning_stocks()

    def get_stock_risk_warning_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定股票的风险警示信息。
        
        参数:
            symbol (str): 股票代码，如 "000001"

        返回:
            Dict[str, Any]: 包含股票风险警示信息的字典，若股票不在风险警示板则返回空字典
            返回字段同get_risk_warning_stocks函数的单个股票数据结构
        
        示例:
            >>> info = stock_provider.get_stock_risk_warning_info("300313")
            >>> if info:
            >>>     print(f"股票{info['name']}在风险警示板")
            >>>     print(f"最新价: {info['price']}, 涨跌幅: {info['change_percent']}%")
            >>> else:
            >>>     print("该股票不在风险警示板")
        """
        try:
            all_risk_stocks = self.get_risk_warning_stocks()
            return all_risk_stocks.get(symbol, {})
            
        except Exception as e:
            self.logger.error(f"获取股票{symbol}的风险警示信息时出错: {str(e)}")
            return {}

    def is_risk_warning_stock(self, symbol: str) -> bool:
        """
        判断指定股票是否为风险警示股票。
        
        参数:
            symbol (str): 股票代码，如 "000001"

        返回:
            bool: 如果是风险警示股票返回True，否则返回False
        """
        return bool(self.get_stock_risk_warning_info(symbol))

    def calculate_stock_correlations(self, symbols: List[str], days: int = 120) -> pd.DataFrame:
        """
        计算给定股票列表的相关性矩阵。

        该方法获取指定股票的历史数据，计算它们在给定时间段内的收盘价相关性，
        并返回相关性矩阵。

        参数：
        symbols (List[str]): 要分析的股票代码列表。
        days (int, 可选): 用于计算相关性的过去天数。默认为120天。

        返回：
        pd.DataFrame: 一个相关性矩阵，其中行和列都标记有股票代码。
                      每个单元格表示两个股票之间的相关系数。

        注意：
        - 相关性基于股票的收盘价计算。
        - 如果某个股票的数据不可用，它将被排除在相关性矩阵之外。
        - 该方法会打印出没有数据或数据获取错误的股票的消息。
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        close_prices = pd.DataFrame()

        for symbol in symbols:
            try:
                df = self.get_historical_daily_data(symbol, start_date, end_date)
                if not df.empty and '收盘' in df.columns:
                    close_prices[symbol] = df['收盘']
                else:
                    print(f"未找到股票 {symbol} 的数据或数据不完整")
            except Exception as e:
                print(f"获取股票 {symbol} 的数据时出错：{str(e)}")

        if close_prices.empty:
            print("没有足够的数据来计算相关性")
            return pd.DataFrame()

        # 确保所有股票的数据长度一致
        min_length = min(len(close_prices[col]) for col in close_prices.columns)
        close_prices = close_prices.tail(min_length)

        # 计算相关性矩阵
        correlation_matrix = close_prices.corr()

        return correlation_matrix

    @cache.cache(ttl=60*60*24,limiter=cd_limiter)
    @thread_safe_cache.thread_safe_cached(cache_key="stock_data")
    def fetch_stock_data(self) -> pd.DataFrame:
        """
        从API获取股票数据。
        
        Returns:
            pd.DataFrame: 股票数据DataFrame
        """
        import akshare as ak
        return ak.stock_info_a_code_name()
    
    def process_stock_data(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        处理股票数据DataFrame，转换为代码-名称映射字典。
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            Dict[str, str]: 代码-名称映射字典
        """
        return dict(zip(df["code"], df["name"]))

    def read_instructions(self, day: str = None) -> str:
        """
        Read and combine instructions from two files.
        
        Parameters:
            day (str, optional): Date in YYYYMMDD format. If None, current date will be used.
        
        Returns:
            str: Combined content from both instruction files.
        """
        import os
        from datetime import datetime

        # If day is not provided, use current date in YYYYMMDD format
        if day is None:
            day = datetime.now().strftime("%Y%m%d")

        # Initialize result string
        result = ""

        # Read base instructions file
        base_file = "./instructions/instructions.txt"
        if os.path.exists(base_file):
            try:
                with open(base_file, 'r', encoding='utf-8') as f:
                    result += f.read()
            except Exception as e:
                self.logger.error(f"Error reading base instructions file: {str(e)}")

        # Read day-specific instructions file
        day_file = f"./instructions/instruction_{day}.txt"
        if os.path.exists(day_file):
            try:
                # Add a newline if result is not empty
                if result:
                    result += "\n\n"
                with open(day_file, 'r', encoding='utf-8') as f:
                    result += f.read()
            except Exception as e:
                self.logger.error(f"Error reading day-specific instructions file: {str(e)}")

        return result

    @thread_safe_cache.thread_safe_cached(cache_key="stock_code_names")
    def get_code_name(self) -> Dict[str, str]:
        """
        获取股票代码和名称的映射字典。使用线程安全缓存机制并支持本地持久化。
        
        Returns:
            Dict[str, str]: 返回股票代码到名称的映射字典 {代码: 名称}
        """
        try:
            # 首先尝试从API获取最新数据
            df = self.fetch_stock_data()
            code_name_dict = self.process_stock_data(df)
            
            return code_name_dict
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch data from API: {str(e)}, trying to load from local cache...")
            
            # 如果API获取失败，尝试从本地缓存加载
            cached_data = load_pickle_cache(CodeNameCache.CACHE_FILE)
            if cached_data is not None:
                self.logger.info("Successfully loaded data from local cache")
                return cached_data
            
            # 如果本地缓存也失败，抛出异常
            raise RuntimeError("Failed to get stock code names from both API and local cache") from e

    def get_stock_code(self,stock_name:str)->str:
        """
        获取股票代码
        """
        cn =  self.get_code_name()
        for k,v in cn.items():
            if v == stock_name:
                return k
        return None

    def is_stock(self,symbol:str)->bool:
        """
        判断是否是股票
        """
        return symbol in self.get_code_name()

    def get_news_updates(self, symbols: List[str],since_time: datetime) -> Dict[str, List[Dict]]:
        """
        个股新闻更新.返回值Dict[symbol,str]
        返回值: Dict[symbol,list]
        list=名称	类型	描述
            关键词	object	-
            新闻标题	object	-
            新闻内容	object	-
            发布时间	object	-
            文章来源	object	-
            新闻链接	object	-
        """
        result = {}
        for symbol in symbols:
            news = ak.stock_news_em(symbol=symbol)
            news = news[news["发布时间"] > since_time]
            result[symbol] = news.to_dict(orient="list")

    def get_market_news_300(self) -> List[str]:
        """
        获取财联社最新300条新闻，并将其格式化为字符串列表。
        备注：这个函数一次获取的信息很多，无法让LLM一次处理，需要调用 summarizer_news(news,query) 来蒸馏提取内容

        返回值:
            List[str]: 每个元素是一个格式化的字符串，包含新闻的标题、内容、发布日期和发布时间。

        字符串格式:
            "标题: {标题}, 内容: {内容}, 发布日期: {发布日期}, 发布时间: {发布时间}"
        """
        # 获取新闻数据并转换为字典
        news_data = ak.stock_info_global_cls().to_dict(orient="list")
        
        # 提取新闻数据，并将其格式化为字符串列表
        formatted_news_list = [
            f"标题: {title}, 内容: {content}, 发布日期: {publish_date}, 发布时间: {publish_time}"
            for title, content, publish_date, publish_time in zip(
                news_data.get("标题", []),
                news_data.get("内容", []),
                news_data.get("发布日期", []),
                news_data.get("发布时间", [])
            )
        ]
        
        return formatted_news_list

    def get_market_news_300_update(self, since: Optional[datetime] = None) -> Tuple[List[str], Optional[datetime]]:
        """
        获取 get_market_news_300 的新闻更新,返回值Tuple[List[str], Optional[datetime]]
        """
        news_data = ak.stock_info_global_cls()
        
        if news_data.empty:
            return [], None
        
        data = news_data if since is None else news_data[news_data["发布时间"] > since.time()]
        
        if data.empty:
            return [], since  # 如果没有新数据，返回空列表和原始的since时间

        dict_data = data.to_dict(orient="list")
        formatted_news_list = [
            f"标题: {title}, 内容: {content}, 发布日期: {publish_date}, 发布时间: {publish_time}"
            for title, content, publish_date, publish_time in zip(
                dict_data.get("标题", []),
                dict_data.get("内容", []),
                dict_data.get("发布日期", []),
                dict_data.get("发布时间", [])
            )
        ]
        
        # 将最后一条新闻的日期和时间转换为datetime对象
        try:
            last_date = data.iloc[-1]["发布日期"]
            last_time = data.iloc[-1]["发布时间"]
            
            # 确保 last_date 是 date 对象，last_time 是 time 对象
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
            if isinstance(last_time, str):
                last_time = datetime.strptime(last_time, "%H:%M:%S").time()
            
            last_datetime = datetime.combine(last_date, last_time)
        except (IndexError, KeyError, ValueError) as e:
            last_datetime = datetime.now()

        return formatted_news_list, last_datetime

    def get_sector_fund_flow_rank(self, indicator: str = "今日", sector_type: str = "行业资金流", top_n: int = 10) -> str:
        """
        获取板块资金流排名数据,indicator: str = "今日", sector_type: str = "行业资金流", top_n: int = 10 ，返回流向字符串

        参数:
        indicator (str): 时间范围，可选 "今日", "5日", "10日"
        sector_type (str): 板块类型，可选 "行业资金流", "概念资金流", "地域资金流"
        top_n (int): 返回前n行数据，-1 表示返回所有数据

        返回:
        str: 格式化的板块资金流排名数据
        """
        try:
            if indicator == "今日":
                if 7<=datetime.now().hour < 9 or (datetime.now().hour == 9 and datetime.now().minute <= 30):
                    indicator = "5日"
            # 获取数据
            df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
            if df is None or df.empty or len(df.columns) == 0:
                df = ak.stock_sector_fund_flow_rank(indicator="5日", sector_type=sector_type)
            # 找到包含“涨跌幅”关键词的列名，并按照此列排序
            change_column = next((col for col in df.columns if '涨跌幅' in col), None)
            if change_column:
                df = df.sort_values(by=change_column, ascending=False)

            # 选择前 top_n 行，如果 top_n 为 -1，则选择所有行
            if top_n != -1:
                df = df.head(top_n)

            # 格式化输出
            result = f"板块资金流排名 ({sector_type} - {indicator}, Top {top_n if top_n != -1 else 'All'}):\n\n"
            for _, row in df.iterrows():
                result += f"{row['序号']}. {row['名称']} ({change_column}: {row[change_column]}%)\n"
                for col in df.columns:
                    if col not in ['序号', '名称', change_column]:
                        value = row[col]
                        # 判断数据类型来格式化输出
                        if isinstance(value, (float, int)):
                            result += f"   {col}: {value:.2f}\n"
                        else:
                            result += f"   {col}: {value}\n"
                result += "\n"

            return result
        except Exception as e:
            return f"获取板块资金流排名数据时出错: {str(e)}"

    def get_sector_fund_flow_rank_dict(self, indicator: str = "5日", sector_type: str = "行业资金流") -> Dict[str, Dict]:
        """
        获取板块资金流排名数据, indicator: str = "5日", sector_type: str = "行业资金流" ，返回流向字典
        如果结果已经缓存，则直接返回缓存的结果。

        参数:
        indicator (str): 时间范围，可选 "今日", "5日", "10日"
        sector_type (str): 板块类型，可选 "行业资金流", "概念资金流", "地域资金流"

        返回:
        Dict[str, Dict]: 以板块名称为键，行数据为值的字典
        """
        # 生成缓存的key
        cache_key = f"{indicator}_{sector_type}"

        # 如果缓存存在，则直接返回
        if cache_key in self.stock_sector_cache:
            #print(f"Returning cached data for key: {cache_key}")
            return self.stock_sector_cache[cache_key]

        try:
            # 获取数据
            df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
            if df is None or df.empty or len(df.columns) == 0:
                df = ak.stock_sector_fund_flow_rank(indicator="5日", sector_type=sector_type)

            # 找到包含“涨跌幅”关键词的列名，并按照此列排序
            change_column = next((col for col in df.columns if '涨跌幅' in col), None)
            if change_column:
                df = df.sort_values(by=change_column, ascending=False)

            # 将数据转换为字典格式
            result = {}
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                result[row['名称']] = row_dict

            # 将结果存入缓存
            self.stock_sector_cache[cache_key] = result

            return result
        except Exception as e:
            return {"error": f"获取板块资金流排名数据时出错: {str(e)}"}

    def get_latest_cyq_data(self, symbol: str, adjust: str = "") -> Dict:
        """
        获取筹码分布数据,参数 symbol: str, adjust: str = "" 返回值 Dict
        """
        # 获取筹码分布数据
        stock_cyq_em_df = ak.stock_cyq_em(symbol=symbol, adjust=adjust)
        
        # 获取最后一行数据
        latest_row = stock_cyq_em_df.iloc[-1]
        
        # 将最后一行数据转换为字典
        result = latest_row.to_dict()
        
        return result

    def get_market_profit_activity(self) -> Dict[str, Any]:
        """
        获取市场赚钱效应分析数据。
        
        该函数提供了对市场整体涨跌情况和投机氛围的分析，包括：
        1. 市场涨跌分布：上涨、下跌、平盘家数
        2. 涨跌停情况：涨停、跌停家数，含真实涨跌停和ST股涨跌停
        3. 市场活跃度：反映市场整体参与度
        
        返回:
            Dict[str, Any]: 包含以下字段的字典：
                - up_count (int): 上涨家数
                - down_count (int): 下跌家数
                - flat_count (int): 平盘家数
                - limit_up (dict): 涨停相关数据
                    - total (int): 涨停总数
                    - real (int): 真实涨停数（非一字无量涨停）
                    - st (int): ST股涨停数
                - limit_down (dict): 跌停相关数据
                    - total (int): 跌停总数
                    - real (int): 真实跌停数（非一字无量跌停）
                    - st (int): ST股跌停数
                - suspended_count (int): 停牌家数
                - activity_ratio (float): 市场活跃度(%)
                - date (str): 统计日期，格式为 "YYYY-MM-DD HH:MM:SS"
                - profit_ratio (float): 市场赚钱效应比例(上涨家数/总家数)
        """
        try:
            # 获取原始数据
            df = ak.stock_market_activity_legu()
            data = dict(zip(df['item'], df['value']))
            
            # 计算总家数（用于计算赚钱效应比例）
            total_stocks = data['上涨'] + data['下跌'] + data['平盘']
            
            # 构建返回字典
            result = {
                'up_count': int(data['上涨']),
                'down_count': int(data['下跌']),
                'flat_count': int(data['平盘']),
                'limit_up': {
                    'total': int(data['涨停']),
                    'real': int(data['真实涨停']),
                    'st': int(data['st st*涨停'])
                },
                'limit_down': {
                    'total': int(data['跌停']),
                    'real': int(data['真实跌停']),
                    'st': int(data['st st*跌停'])
                },
                'suspended_count': int(data['停牌']),
                'activity_ratio': float(data['活跃度'].rstrip('%')),
                'date': str(data['统计日期']),
                'profit_ratio': round(float(data['上涨']) / total_stocks * 100, 2)
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"获取市场赚钱效应数据时出错: {str(e)}")
            return {
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'limit_up': {'total': 0, 'real': 0, 'st': 0},
                'limit_down': {'total': 0, 'real': 0, 'st': 0},
                'suspended_count': 0,
                'activity_ratio': 0.0,
                'date': '',
                'profit_ratio': 0.0,
                'error': str(e)
            }

    def get_sector_fund_flow(self, symbol: str, indicator: str = "今日") -> Dict[str, Dict]:
        """
        获取行业个股资金流数据,参数 symbol: str, indicator: str = "今日" 返回值 Dict[str, Dict]
        """
        # 获取行业个股资金流数据
        stock_sector_fund_flow_summary_df = ak.stock_sector_fund_flow_summary(symbol=symbol, indicator=indicator)
        
        # 将DataFrame转换为字典
        result = {}
        for _, row in stock_sector_fund_flow_summary_df.iterrows():
            stock_code = row['代码']
            row_dict = row.to_dict()
            result[stock_code] = row_dict
        
        return result

    def get_stock_main_fund_flow(self, top_n: int = 10 , symbol: str = "全部股票") -> str:
        """
        获取主力资金流向排名数据

        参数:
        symbol (str): 股票类型，可选 "全部股票", "沪深A股", "沪市A股", "科创板", "深市A股", "创业板", "沪市B股", "深市B股"
        top_n (int): 返回前n行数据，-1 表示返回所有数据

        返回:
        str: 格式化的主力资金流向排名数据
        """
        try:
            # 获取数据
            df = ak.stock_main_fund_flow(symbol=symbol)

            # 根据今日主力净占比排序
            df = df.sort_values(by='今日排行榜-主力净占比', ascending=False)

            # 选择前 top_n 行，如果 top_n 为 -1，则选择所有行
            if top_n != -1:
                df = df.head(top_n)

            # 格式化输出
            result = f"主力资金流向排名 ({symbol}, Top {top_n if top_n != -1 else 'All'}):\n\n"
            for _, row in df.iterrows():
                result += f"{row['序号']}. {row['名称']} ({row['代码']}) - {row['所属板块']}\n"
                result += f"   最新价: {row['最新价']:.2f}\n"
                result += f"   今日排行榜: 主力净占比 {row['今日排行榜-主力净占比']:.2f}%, 排名 {row['今日排行榜-今日排名']}, 涨跌幅 {row['今日排行榜-今日涨跌']:.2f}%\n"
                result += f"   5日排行榜: 主力净占比 {row['5日排行榜-主力净占比']:.2f}%, 排名 {row['5日排行榜-5日排名']}, 涨跌幅 {row['5日排行榜-5日涨跌']:.2f}%\n"
                result += f"   10日排行榜: 主力净占比 {row['10日排行榜-主力净占比']:.2f}%, 排名 {row['10日排行榜-10日排名']}, 涨跌幅 {row['10日排行榜-10日涨跌']:.2f}%\n\n"

            return result
        except Exception as e:
            return f"获取主力资金流向排名数据时出错: {str(e)}"

    def get_stock_minute(self,symbol:str, period='1'):
        """
        个股分钟数据 参数symbol:str  返回值DataFrame
        输入参数：
            symbol:str 股票代码
            period:str 周期，默认为1，可选值：1,5,15,30,60
        返回值:
            名称	类型	描述
            day	object	-
            open	float64	-
            high	float64	-
            low	float64	-
            close	float64	-
            volume	float64	-
        """
        symbol = self.add_market_prefix(symbol)
        return ak.stock_zh_a_minute(symbol=symbol, period=period)

    def add_market_prefix(self,stock_code):
        if not stock_code.isdigit() or len(stock_code) != 6:
            return stock_code
        
        first_digit = int(stock_code[0])
        
        if first_digit == 6:
            return "sh" + stock_code
        elif first_digit == 0 or first_digit == 3:
            return "sz" + stock_code
        elif first_digit == 4 or first_digit == 8:
            return "bj" + stock_code
        else:
            return "Unknown market: " + stock_code

    def get_index_data(self, index_symbols: List[str],start_date:str,end_date:str) -> Dict[str, pd.DataFrame]:
        """
        获取指数数据,参数index_symbols: List[str]  返回值Dict[symbol,DataFrame]
        """
        result = {}
        for index in index_symbols:
            
            data = ak.index_zh_a_hist(symbol=index,period="daily",start_date=start_date,end_date=end_date)
            result[index] = data
        return result

    def find_index_codes(self,names: List[str]) -> Dict[str, str]:
        # 获取所有指数数据
        stock_zh_index_spot_sina_df = ak.stock_zh_index_spot_sina()
        
        # 创建一个字典来存储结果
        result = {}
        
        # 将DataFrame的'名称'和'代码'列转换为字典，以便快速查找
        name_code_dict = dict(zip(stock_zh_index_spot_sina_df['名称'], stock_zh_index_spot_sina_df['代码']))
        
        # 遍历输入的名称列表
        for name in names:
            if name in name_code_dict:
                result[name] = name_code_dict[name]
        
        return result

    def fetch_historical_index_data(self,symbols: List[str], days: int = 120) -> Dict[str, pd.DataFrame]:
        result = {}
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        for symbol in symbols:
            try:
                df = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    result[symbol] = df
                else:
                    print(f"No data found for symbol: {symbol}")
            except Exception as e:
                print(f"Error fetching data for symbol {symbol}: {str(e)}")
        
        return result

    def calculate_industry_correlations(self, names: List[str], days: int = 120) -> pd.DataFrame:
        """
        计算给定行业指数的相关性矩阵。

        该方法获取指定行业指数的历史数据，计算它们在给定时间段内的收盘价相关性，
        并返回相关性矩阵。

        参数：
        names (List[str]): 要分析的行业指数名称列表。
        days (int, 可选): 用于计算相关性的过去天数。默认为120天。

        返回：
        pd.DataFrame: 一个相关性矩阵，其中行和列都标记有行业指数名称。
                      每个单元格表示两个指数之间的相关系数。

        注意：
        - 相关性基于指数的收盘价计算。
        - 如果某个指数的数据不可用，它将被排除在相关性矩阵之外。
        - 该方法会打印出没有数据或数据获取错误的指数的消息。
        """
        # 获取指数代码
        index_codes = self.find_index_codes(names)
        
        # 获取历史数据
        historical_data = self.fetch_historical_index_data(list(index_codes.values()), days)
        
        # 准备一个DataFrame来存储所有指数的收盘价
        close_prices = pd.DataFrame()
        
        for name, code in index_codes.items():
            if code in historical_data:
                close_prices[name] = historical_data[code]['close']
        
        # 计算相关性矩阵
        correlation_matrix = close_prices.corr()
        
        return correlation_matrix

    def get_stock_news(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        获取个股新闻。

        参数:
            symbols: List[str]  股票代码列表

        返回值:
            Dict[str, List[Dict]]: 每个股票代码对应的新闻列表，每条新闻包含以下字段：
                - 新闻标题
                - 新闻内容
                - 发布时间
        """
        result = {}
        # 要保留的列
        keep_columns = ['新闻标题', '新闻内容', '发布时间']
        
        for symbol in symbols:
            news = ak.stock_news_em(symbol=symbol)
            # 只保留指定的列
            filtered_news = news[keep_columns]
            result[symbol] = filtered_news.to_dict(orient="list")
        
        return result

    def get_one_stock_news(self, symbol: str, num: int = 5, days: int = 7) -> List[Dict[str, str]]:
        """
        获取指定股票的最新新闻。

        参数:
        symbol (str): 股票代码，例如 "000001" 代表平安银行
        num (int): 需要获取的新闻数量，默认为5条
        days (int): 获取最近几天的新闻，默认为7天

        返回:
        List[Dict[str, str]]: 包含新闻信息的字典列表，每个字典包含以下键：
            - 'title': 新闻标题
            - 'content': 新闻内容摘要
            - 'datetime': 新闻发布时间
            - 'url': 新闻链接（如果有）

        异常:
        ValueError: 如果无法获取股票新闻
        """
        if "." in symbol:
            symbol = symbol.split(".")[0]
        try:
            # 计算起始日期
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 使用 akshare 获取股票新闻
            # 注意：这里假设 ak.stock_news_em 是获取新闻的正确函数，您可能需要根据实际情况调整
            df = ak.stock_news_em(symbol=symbol)

            # 过滤日期范围内的新闻
            df['datetime'] = pd.to_datetime(df['发布时间'])
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

            # 选择最新的 num 条新闻
            df = df.sort_values('datetime', ascending=False).head(num)

            # 构建结果列表
            news_list = []
            for _, row in df.iterrows():
                news_item = {
                    'title': row['新闻标题'],
                    'content': row['新闻内容'][:200] + '...',  # 取前200个字符作为摘要
                    'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'url': row['新闻链接'] if '新闻链接' in row else ''
                }
                news_list.append(news_item)

            return news_list

        except Exception as e:
            raise ValueError(f"无法获取股票 {symbol} 的新闻: {str(e)}")

    def get_latest_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定股票的最新综合数据。

        参数:
        symbol (str): 股票代码，例如 "000001" 代表平安银行

        返回:
        Dict[str, Any]: 包含股票最新数据的字典，包括以下键：
            - 'symbol': 股票代码
            - 'name': 股票名称
            - 'price': 最新价格
            - 'change': 涨跌额
            - 'change_percent': 涨跌幅（百分比）
            - 'open': 开盘价
            - 'high': 最高价
            - 'low': 最低价
            - 'volume': 成交量
            - 'amount': 成交金额
            - 'bid_price': 买一价
            - 'ask_price': 卖一价
            - 'bid_volume': 买一量
            - 'ask_volume': 卖一量
            - 'timestamp': 数据时间戳

        异常:
        ValueError: 如果无法获取股票数据
        """
        if "." in symbol:
            symbol = symbol.split(".")[0]
        try:
            # 使用 akshare 的 stock_bid_ask_em 函数获取最新行情数据
            df = ak.stock_bid_ask_em(symbol=symbol)
            
            # 提取需要的数据
            data = {
                'symbol': symbol,
                'name': self._get_stock_name(symbol),  # 这个方法需要另外实现
                'price': float(df.loc[df['item'] == '最新', 'value'].values[0]),
                'change': float(df.loc[df['item'] == '涨跌', 'value'].values[0]),
                'change_percent': float(df.loc[df['item'] == '涨幅', 'value'].values[0]),
                'open': float(df.loc[df['item'] == '今开', 'value'].values[0]),
                'high': float(df.loc[df['item'] == '最高', 'value'].values[0]),
                'low': float(df.loc[df['item'] == '最低', 'value'].values[0]),
                'volume': float(df.loc[df['item'] == '总手', 'value'].values[0]),
                'amount': float(df.loc[df['item'] == '金额', 'value'].values[0]),
                'bid_price': float(df.loc[df['item'] == 'buy_1', 'value'].values[0]),
                'ask_price': float(df.loc[df['item'] == 'sell_1', 'value'].values[0]),
                'bid_volume': float(df.loc[df['item'] == 'buy_1_vol', 'value'].values[0]),
                'ask_volume': float(df.loc[df['item'] == 'sell_1_vol', 'value'].values[0]),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            return data

        except Exception as e:
            raise ValueError(f"无法获取股票 {symbol} 的最新数据: {str(e)}")

    def _get_stock_name(self, symbol: str) -> str:
        """
        根据股票代码获取股票名称的辅助方法。
        """
        return self.get_code_name()[symbol]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_stock_info(self,symbol: str) -> str:
        """
        查询指定股票代码的个股信息，参数symbol: str  返回str。

        参数:
        symbol (str): 股票代码，例如 "603777"。

        返回:
        str: 个股信息的格式化字符串，包括总市值、流通市值、行业、上市时间、股票代码、股票简称、总股本和流通股本。
        """

        # 获取个股信息数据框
        stock_info_df = ak.stock_individual_info_em(symbol=symbol)

        # 将数据转换为可读的字符串格式
        stock_info_str = "\n".join([f"{row['item']}: {row['value']}" for _, row in stock_info_df.iterrows()])
        
        return stock_info_str
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_stock_info_dict(self, symbol: str) -> Dict[str, Any]:
        """
        查询指定股票代码的个股信息，参数symbol: str  返回Dict[str, Any]。

        参数:
        symbol (str): 股票代码，例如 "603777"。

        返回:
        Dict[str, Any]: 个股信息的字典，包括总市值、流通市值、行业、上市时间、股票代码、股票简称、总股本和流通股本。
        """

        # 获取个股信息数据框
        stock_info_df = ak.stock_individual_info_em(symbol=symbol)

        # 将数据转换为字典格式
        stock_info_dict = {row['item']: row['value'] for _, row in stock_info_df.iterrows()}
        
        # 添加股票代码到字典中
        stock_info_dict['股票代码'] = symbol

        return stock_info_dict

    def get_realtime_stock_data(self, symbol: str) -> str:
        """
        查询指定证券代码的最新行情数据，参数symbol:str 返回str。

        参数:
        symbol (str): 证券代码，可以是 A 股个股代码 例如 "600000"。

        返回:
        str: 最新行情数据的格式化字符串，包括代码、现价、涨幅、最高价、最低价、市盈率、成交量等信息。
        """

        # 获取实时行情数据
        stock_spot_df = ak.stock_zh_a_spot_em()
        stock_spot_df = stock_spot_df[stock_spot_df['代码'] == symbol]

        if stock_spot_df.empty:
            return f"未找到证券代码 {symbol} 的数据"

        # 定义需要显示的字段及其格式化方式
        fields = [
            ('代码', '{}'),
            ('名称', '{}'),
            ('最新价', '{:.2f}'),
            ('涨跌幅', '{:.2f}%'),
            ('涨跌额', '{:.2f}'),
            ('成交量', '{:.0f}手'),
            ('成交额', '{:.2f}元'),
            ('振幅', '{:.2f}%'),
            ('最高', '{:.2f}'),
            ('最低', '{:.2f}'),
            ('今开', '{:.2f}'),
            ('昨收', '{:.2f}'),
            ('量比', '{:.2f}'),
            ('换手率', '{:.2f}%'),
            ('市盈率-动态', '{:.2f}'),
            ('市净率', '{:.2f}'),
            ('总市值', '{:.2f}元'),
            ('流通市值', '{:.2f}元'),
            ('涨速', '{:.2f}'),
            ('5分钟涨跌', '{:.2f}%'),
            ('60日涨跌幅', '{:.2f}%'),
            ('年初至今涨跌幅', '{:.2f}%')
        ]

        # 格式化数据
        formatted_data = []
        for field, format_str in fields:
            if field in stock_spot_df.columns:
                value = stock_spot_df[field].values[0]
                if pd.notna(value):  # 检查是否为NaN
                    formatted_value = format_str.format(value)
                    formatted_data.append(f"{field}: {formatted_value}")
                else:
                    formatted_data.append(f"{field}: 无数据")
            else:
                formatted_data.append(f"{field}: 无数据")

        # 将格式化的数据转换为字符串
        stock_spot_str = "\n".join(formatted_data)
        
        return stock_spot_str

    @thread_safe_cache.thread_safe_cached("full_realtime_data",timeout=60)
    def get_full_realtime_data(self) -> dict[str, str]:
        """
        获取并格式化当前证券代码列表的实时行情数据。返回值Dict[symbol,str]

        函数将获取证券代码的最新行情数据，并将其结果转换为格式化的字符串，存储在字典中。

        返回:
        dict[str, str]: 每个证券代码及其对应的最新行情数据的格式化字符串，包含代码、现价、涨幅、最高价、最低价、市盈率、成交量等信息。
        """

        # 获取实时行情数据
        stock_spot_df = ak.stock_sz_a_spot_em()

        # 初始化一个字典来存储结果
        formatted_data = {}

        # 遍历实时数据的每一行
        for _, row in stock_spot_df.iterrows():
            symbol = row['代码']
            
            # 将数据转换为可读的字符串格式
            stock_spot_str = "\n".join([f"{item}: {value}" for item, value in row.items() if item != '代码'])

            # 存储在字典中
            formatted_data[symbol] = stock_spot_str

        return formatted_data

    def get_block_trade_details_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        all_data = self.get_block_trade_details()
        if all_data is None or len(all_data) == 0:
            return []
        return [data for data in all_data if data['证券代码'] == symbol]

    @cache.cache(ttl=60*30)
    def get_block_trade_details(self, symbol: str = 'A股', start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        获取大宗交易明细数据。

        参数:
            symbol (str): 交易品种,'A股','B股','基金','债券'之一。默认'A股'
            start_date (str): 开始日期,格式'YYYYMMDD',为None时使用最近交易日
            end_date (str): 结束日期,格式'YYYYMMDD',为None时使用最近交易日

        返回:
            List[Dict[str, Any]]: 包含大宗交易数据的字典列表,每个字典包含:
                - 序号: 交易编号
                - 交易日期: 成交时间 
                - 证券代码: 标的代码
                - 证券简称: 标的名称
                - 涨跌幅: 涨跌幅(%)
                - 收盘价: 当日收盘价
                - 成交价: 大宗交易价格
                - 折溢率: 成交价格相对收盘价折溢率 
                - 成交量: 成交股数 
                - 成交额: 成交金额(元)
                - 成交额/流通市值: 占流通股比例(%)
                - 买方营业部: 买方机构
                - 卖方营业部: 卖方机构
        """
        if not start_date:
            start_date = self.get_previous_trading_date()
        if not end_date:
            end_date = self.get_latest_trading_date()
            
        df = ak.stock_dzjy_mrmx(symbol=symbol, start_date=start_date, end_date=end_date)
        return df.to_dict(orient='records')
    
    @cache.cache(ttl=60*30)
    def get_block_trade_summary(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        获取大宗交易每日统计数据。

        参数:
            start_date (str): 开始日期,格式'YYYYMMDD',为None时使用最近交易日
            end_date (str): 结束日期,格式'YYYYMMDD',为None时使用最近交易日

        返回:
            List[Dict[str, Any]]: 大宗交易统计数据的字典列表,每个字典包含:
                - 序号: 统计编号
                - 交易日期: 统计日期
                - 证券代码: 标的代码
                - 证券简称: 标的名称
                - 涨跌幅: 涨跌幅(%)
                - 收盘价: 当日收盘价
                - 成交均价: 平均成交价格
                - 折溢率: 成交均价相对收盘价折溢率
                - 成交笔数: 大宗交易笔数
                - 成交总量: 总成交股数(万股)
                - 成交总额: 总成交金额(万元)
                - 成交总额/流通市值: 占流通股比例(%)
        """
        if not start_date:
            start_date = self.get_previous_trading_date()
        if not end_date:
            end_date = self.get_latest_trading_date()
            
        df = ak.stock_dzjy_mrtj(start_date=start_date, end_date=end_date)
        return df.to_dict(orient='records')

    def get_stock_announcements(self,symbols: List[str], date: str = None) -> Dict[str, List[str]]:
        """
        获取指定日期内指定股票代码的公告信息。参数symbols: List[str] 返回值Dict[symbol,List[str]]

        参数:
        symbols (List[str]): 股票代码列表。
        date (str, 可选): 查询的日期，格式为 "YYYY-MM-DD"。如果未指定，则使用最近的交易日期。

        返回:
        Dict[str, List[str]]: 一个字典，其中键是股票代码，值是该股票在指定日期内发布的公告列表。
        """
        result = {}
        if not date:
            date = self.get_latest_trading_date()
        try:
            df = ak.stock_gsrl_gsdt_em(date=date)
        except Exception as e:
            #self.logger.error(f"获取公告数据失败: {str(e)} {format_exc()}")
            return {}
        
        if df is None or df.empty:
            return {}
        for symbol in symbols:
            result[symbol] = []
            filtered_df = df[df['股票代码'] == symbol]
            for row in filtered_df.itertuples():
                result[symbol].append(row["具体事项"])
        return result

    def stock_info_global_ths(self):
        """
        同花顺财经 20条,返回DataFrame
        返回值：
            名称	类型	描述
            标题	object	-
            内容	object	-
            发布时间	object	-
            链接	object	-
        """
        return ak.stock_info_global_ths()

    def stock_info_global_futu(self):
        """
        富途财经 50条 ,返回DataFrame
        返回值：
            名称	类型	描述
            标题	object	-
            内容	object	-
            发布时间	object	-
            链接	object	-
        """
        return ak.stock_info_global_futu()

    def stock_info_global_sina(self):
        """
        新浪财经 20条 ,返回DataFrame
        返回值：
        名称	类型	描述
        时间	object	-
        内容	object	-
        """
        return ak.stock_info_global_sina()

    def stock_info_global_em(self):
        """
        东方财富 200条 ,返回DataFrame
        返回值：
            名称	类型	描述
            标题	object	-
            摘要	object	-
            发布时间	object	-
            链接	object	-
        """
        return ak.stock_info_global_em()

    def summarize_historical_data_dict(self, symbols: List[str], days: int = 180) -> Dict[str, Dict[str, Any]]:
        summary_dict = {}
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        for symbol in symbols:
            if symbol in self.historical_data_cache:
                df = self.historical_data_cache[symbol]
            else:
                df = self.get_historical_daily_data(symbol, start_date, end_date)
                self.historical_data_cache[symbol] = df
            
            if df.empty:
                summary_dict[symbol] = {"error": "未找到数据"}
                continue

            if len(df) < 14:
                summary_dict[symbol] = {"error": f"数据点不足，仅有 {len(df)} 个数据点，无法计算所有技术指标"}
                continue

            # 计算技术指标
            df['MA20'] = ta.trend.sma_indicator(df['收盘'], window=min(20, len(df)))
            df['MA50'] = ta.trend.sma_indicator(df['收盘'], window=min(50, len(df)))
            df['RSI'] = ta.momentum.rsi(df['收盘'], window=min(14, len(df)))
            macd = ta.trend.MACD(df['收盘'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(df['收盘'], window=min(20, len(df)), window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()

            if len(df) >= 14:
                df['ATR'] = ta.volatility.average_true_range(df['最高'], df['最低'], df['收盘'], window=14)
                stoch = ta.momentum.StochasticOscillator(df['最高'], df['最低'], df['收盘'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
                df['RSI_9'] = ta.momentum.rsi(df['收盘'], window=9)
                df['OBV'] = ta.volume.on_balance_volume(df['收盘'], df['成交量'])
                df['Momentum'] = ta.momentum.roc(df['收盘'], window=min(10, len(df)))
                df['ADL'] = ta.volume.acc_dist_index(df['最高'], df['最低'], df['收盘'], df['成交量'])
                df['Williams_R'] = ta.momentum.williams_r(high=df['最高'], low=df['最低'], close=df['收盘'], lbp=min(14, len(df)))

            # 构建结构化的摘要字典
            summary = {
                "股票代码": symbol,
                "当前价格": df['收盘'].iloc[-1],
                "最高收盘价": df['收盘'].max(),
                "最低收盘价": df['收盘'].min(),
                "平均成交量": df['成交量'].mean(),
                "平均成交额": df['成交额'].mean(),
                "技术指标": {}
            }

            # 添加技术指标
            indicators = ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'MA20', 'MA50', 
                          'ATR', 'Stoch_K', 'Stoch_D', 'RSI_9', 'OBV', 'Momentum', 'ADL', 'Williams_R']
            
            for indicator in indicators:
                if indicator in df.columns:
                    summary["技术指标"][indicator] = df[indicator].iloc[-1]

            summary_dict[symbol] = summary

        return summary_dict

    def summarize_historical_data(self, symbols: List[str],days: int = 180) -> dict:
        summary_dict = {}
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        for symbol in symbols:
            if symbol in self.historical_data_cache:
                df = self.historical_data_cache[symbol]
            else:
                df = self.get_historical_daily_data(symbol, start_date, end_date)
                self.historical_data_cache[symbol] = df
            
            if df.empty:
                summary_dict[symbol] = "未找到数据"
                continue

            # 检查数据点是否足够
            if len(df) < 14:
                summary_dict[symbol] = f"数据点不足，仅有 {len(df)} 个数据点，无法计算所有技术指标"
                continue

            # 保留原有的指标计算
            df['MA20'] = ta.trend.sma_indicator(df['收盘'], window=min(20, len(df)))
            df['MA50'] = ta.trend.sma_indicator(df['收盘'], window=min(50, len(df)))
            df['RSI'] = ta.momentum.rsi(df['收盘'], window=min(14, len(df)))
            macd = ta.trend.MACD(df['收盘'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(df['收盘'], window=min(20, len(df)), window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()

            # 新增指标，确保数据点足够
            if len(df) >= 14:
                df['ATR'] = ta.volatility.average_true_range(df['最高'], df['最低'], df['收盘'], window=14)
                
                stoch = ta.momentum.StochasticOscillator(df['最高'], df['最低'], df['收盘'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
                
                df['RSI_9'] = ta.momentum.rsi(df['收盘'], window=9)
                
                df['OBV'] = ta.volume.on_balance_volume(df['收盘'], df['成交量'])
                
                df['Momentum'] = ta.momentum.roc(df['收盘'], window=min(10, len(df)))
                
                df['ADL'] = ta.volume.acc_dist_index(df['最高'], df['最低'], df['收盘'], df['成交量'])
                
                df['Williams_R'] = ta.momentum.williams_r(high=df['最高'], low=df['最低'], close=df['收盘'], lbp=min(14, len(df)))

            # 获取数据统计（包括新增指标）
            latest_close = df['收盘'].iloc[-1]
            highest_close = df['收盘'].max()
            lowest_close = df['收盘'].min()
            avg_volume = df['成交量'].mean()
            latest_rsi = df['RSI'].iloc[-1] if 'RSI' in df else None
            latest_macd = df['MACD'].iloc[-1] if 'MACD' in df else None
            latest_macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df else None
            bb_upper = df['BB_upper'].iloc[-1] if 'BB_upper' in df else None
            bb_lower = df['BB_lower'].iloc[-1] if 'BB_lower' in df else None
            latest_atr = df['ATR'].iloc[-1] if 'ATR' in df else None
            latest_stoch_k = df['Stoch_K'].iloc[-1] if 'Stoch_K' in df else None
            latest_stoch_d = df['Stoch_D'].iloc[-1] if 'Stoch_D' in df else None
            latest_rsi_9 = df['RSI_9'].iloc[-1] if 'RSI_9' in df else None
            latest_obv = df['OBV'].iloc[-1] if 'OBV' in df else None
            latest_momentum = df['Momentum'].iloc[-1] if 'Momentum' in df else None
            latest_adl = df['ADL'].iloc[-1] if 'ADL' in df else None
            latest_williams_r = df['Williams_R'].iloc[-1] if 'Williams_R' in df else None

            # 生成描述性的字符串
            description = (
                f"股票代码: {symbol}\n"
                f"当前价格: {latest_close:.2f}\n"
                f"最高收盘价: {highest_close:.2f}\n"
                f"最低收盘价: {lowest_close:.2f}\n"
                f"平均成交量: {avg_volume:.0f}\n"
            )

            if latest_rsi is not None:
                description += f"最新RSI(14): {latest_rsi:.2f}\n"
            if latest_macd is not None and latest_macd_signal is not None:
                description += f"最新MACD: {latest_macd:.2f}\n"
                description += f"最新MACD信号线: {latest_macd_signal:.2f}\n"
            if bb_upper is not None and bb_lower is not None:
                description += f"布林带上轨: {bb_upper:.2f}\n"
                description += f"布林带下轨: {bb_lower:.2f}\n"
            if 'MA20' in df and 'MA50' in df:
                description += f"MA20: {df['MA20'].iloc[-1]:.2f}\n"
                description += f"MA50: {df['MA50'].iloc[-1]:.2f}\n"
            if latest_atr is not None:
                description += f"ATR(14): {latest_atr:.2f}\n"
            if latest_stoch_k is not None and latest_stoch_d is not None:
                description += f"随机振荡器K(14): {latest_stoch_k:.2f}\n"
                description += f"随机振荡器D(14): {latest_stoch_d:.2f}\n"
            if latest_rsi_9 is not None:
                description += f"RSI(9): {latest_rsi_9:.2f}\n"
            if latest_obv is not None:
                description += f"OBV: {latest_obv:.0f}\n"
            if latest_momentum is not None:
                description += f"价格动量(10): {latest_momentum:.2f}%\n"
            if latest_adl is not None:
                description += f"ADL: {latest_adl:.0f}\n"
            if latest_williams_r is not None:
                description += f"威廉指标(14): {latest_williams_r:.2f}"
            
            summary_dict[symbol] = description
        
        return summary_dict

    def summarize_historical_index_data(self, index_symbols: List[str]) -> dict:
        """
        汇总多个指数的历史数据， 参数symbols: List[str] 返回Dict[symbol,str]
        备注：上证指数：000001;上证50:000016;上证300：000300；中证1000：000852；中证500：000905；创业板指数：399006

        输入参数:
            index_symbols: List[str] 指数代码列表

        返回值:
            一个字典，键是指数代码，值是描述性的字符串。
        """
        summary_dict = {}
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

        for symbol in index_symbols:
            # 检查缓存
            if symbol in self.historical_data_cache:
                df = self.historical_data_cache[symbol]
            else:
                df = self.get_index_data([symbol], start_date, end_date)[symbol]
                self.historical_data_cache[symbol] = df
            
            if df.empty:
                summary_dict[symbol] = "未找到数据"
                continue

            # 获取数据统计
            latest_close = df['收盘'].iloc[-1]
            highest_close = df['收盘'].max()
            lowest_close = df['收盘'].min()
            avg_volume = df['成交量'].mean()
            std_dev = df['收盘'].std()
            median_close = df['收盘'].median()
            avg_close = df['收盘'].mean()
            return_rate = (df['收盘'].iloc[-1] - df['收盘'].iloc[0]) / df['收盘'].iloc[0] * 100

            # 生成描述性的字符串
            description = (
                f"指数代码: {symbol}\n"
                f"最新收盘价: {latest_close}\n"
                f"最近半年内最高收盘价: {highest_close}\n"
                f"最近半年内最低收盘价: {lowest_close}\n"
                f"最近半年平均成交量: {avg_volume}\n"
                f"收盘价标准差: {std_dev}\n"
                f"收盘价中位数: {median_close}\n"
                f"最近半年平均收盘价: {avg_close}\n"
                f"半年累计回报率: {return_rate:.2f}%"
            )
            
            summary_dict[symbol] = description
        
        return summary_dict

    def get_index_components(self, index_symbol: str) -> list:
        """
        获取指定指数的最新成分股代码列表。参数 index_symbol: str 返回 list

        参数:
            index_symbol (str): 指数代码，例如 "000300" 表示沪深300指数。

        返回:
            list: 包含指定指数所有成分股代码的列表。
        """
        try:
            df = ak.index_stock_cons(index_symbol)
            stock_codes = df["品种代码"].to_list()
            return stock_codes
        except Exception as e:
            print(f"获取成分股数据时出错: {e}")
            return []
        
    def get_function_docstring(self, function_name: str) -> str:
        """
        获取指定函数的__docstring__。

        参数:
            function_name (str): 函数名称（字符串形式）。

        返回:
            str: 该函数的__docstring__，如果函数不存在或没有__docstring__，返回提示信息。
        """
        function = getattr(self, function_name, None)
        if function is None:
            return f"函数 '{function_name}' 不存在。"
        
        docstring = function.__doc__
        if docstring:
            return docstring.strip()
        else:
            return f"函数 '{function_name}' 没有 __docstring__。"

    @cache.cache(ttl=60*60*4)
    def get_institutional_holdings(self, symbol: str) -> str:
        """
        获取指定股票的机构持股数据。使用线程安全的缓存机制，按季度缓存数据。

        参数:
            symbol (str): 股票代码，如 "000001"

        返回:
            str: 格式化的机构持股信息字符串，包含：
                - 证券代码和简称
                - 机构数及其变化
                - 持股比例及其增幅
                - 占流通股比例及其增幅
                如果没有数据返回"暂无数据"信息
        """
        # 计算当前年份和季度
        now = datetime.now()
        year = now.year
        month = now.month

        # 由于季报有延迟，实际可获得的最新季报要往前推
        if month in [1, 2, 3]:  # 1-3月
            year = year - 1  # 上一年
            quarter = "4"
        elif month in [4, 5, 6]:  # 4-6月
            quarter = "1"
        elif month in [7, 8, 9]:  # 7-9月
            quarter = "2"
        elif month in [10, 11, 12]:  # 10-12月
            quarter = "3"
        report_period = f"{year}{quarter}"
        
        # 使用年份和季度作为缓存键
        cache_key = f"institutional_holdings_{report_period}"

        @thread_safe_cache.thread_safe_cached(cache_key)
        def fetch_institutional_holdings() -> Dict[str, str]:
            """
            获取并缓存所有股票的机构持股数据。
            """
            result = {}
            try:
                df = ak.stock_institute_hold(symbol=report_period)
                if df.empty:
                    self.logger.warning(f"{report_period}期间没有机构持股数据")
                    return result

                # 处理每行数据
                for _, row in df.iterrows():
                    try:
                        code = str(row['证券代码'])  # 确保转换为字符串
                        formatted_row = (
                            f"证券代码: {row['证券代码']}, 证券简称: {row['证券简称']}, "
                            f"机构数: {row['机构数']}, 机构数变化: {row['机构数变化']}, "
                            f"持股比例: {row['持股比例']}%, 持股比例增幅: {row['持股比例增幅']}%, "
                            f"占流通股比例: {row['占流通股比例']}%, "
                            f"占流通股比例增幅: {row['占流通股比例增幅']}%"
                        )
                        result[code] = formatted_row
                    except KeyError as e:
                        self.logger.error(f"处理机构持股数据时出现键错误: {str(e)}")
                        continue
                    except Exception as e:
                        self.logger.error(f"处理机构持股数据时出现错误: {str(e)}")
                        continue

                return result
                
            except Exception as e:
                self.logger.error(f"获取{report_period}期间机构持股数据时出错: {str(e)}")
                return result

        try:
            # 获取所有数据
            holdings_data = fetch_institutional_holdings()
            
            # 检查数据是否为空
            if not holdings_data:
                return f"{report_period}期间暂无机构持股数据"
                
            # 返回特定股票的数据
            return holdings_data.get(symbol, f"{report_period}期间该股票暂无机构持股数据")
            
        except Exception as e:
            self.logger.error(f"获取股票{symbol}的机构持股数据时出错: {str(e)}")
            return "获取机构持股数据失败"

    def get_latest_institutional_quarter(self) -> str:
        """
        获取最新的机构持股数据季度。

        返回:
            str: 格式为 "YYYYQ" 的季度标识，如 "20241" 表示2024年第1季度
        """
        now = datetime.now()
        year = str(now.year)
        quarter = (now.month - 1) // 3 + 1
        return f"{year}{quarter}"
    
    def _get_file_content_ths(self , file: str = "ths.js") -> str:
        """
        获取 JS 文件的内容
        :param file:  JS 文件名
        :type file: str
        :return: 文件内容
        :rtype: str
        """
        from akshare.datasets import get_ths_js
        setting_file_path = get_ths_js(file)
        with open(setting_file_path, encoding="utf-8") as f:
            file_data = f.read()
        return file_data

    @cache.cache(ttl=60*60*1)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def stock_fund_flow_concept(self, symbol: str = "即时") -> pd.DataFrame:
        """
        同花顺-数据中心-资金流向-概念资金流
        https://data.10jqka.com.cn/funds/gnzjl/#refCountId=data_55f13c2c_254
        
        Args:
            symbol: choice of {"即时", "3日排行", "5日排行", "10日排行", "20日排行"}
        
        Returns:
            pandas.DataFrame: 概念资金流数据
        """
        import py_mini_racer
        import requests
        from akshare.utils.tqdm import get_tqdm
        from bs4 import BeautifulSoup
        from io import StringIO
        
        def get_v_code():
            js_code = py_mini_racer.MiniRacer()
            js_content = self._get_file_content_ths("ths.js")
            js_code.eval(js_content)
            return js_code.call("v")
        
        def get_headers():
            return {
                "Accept": "text/html, */*; q=0.01",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "hexin-v": get_v_code(),
                "Host": "data.10jqka.com.cn",
                "Pragma": "no-cache",
                "Referer": "http://data.10jqka.com.cn/funds/gnzjl/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
                "X-Requested-With": "XMLHttpRequest",
            }

        # 获取总页数
        initial_url = "http://data.10jqka.com.cn/funds/gnzjl/field/tradezdf/order/desc/ajax/1/free/1/"
        r = requests.get(initial_url, headers=get_headers())
        soup = BeautifulSoup(r.text, features="lxml")
        raw_page = soup.find(name="span", attrs={"class": "page_info"}).text
        page_num = raw_page.split("/")[1]

        # 根据symbol选择URL
        url_map = {
            "3日排行": "http://data.10jqka.com.cn/funds/gnzjl/board/3/field/tradezdf/order/desc/page/{}/ajax/1/free/1/",
            "5日排行": "http://data.10jqka.com.cn/funds/gnzjl/board/5/field/tradezdf/order/desc/page/{}/ajax/1/free/1/",
            "10日排行": "http://data.10jqka.com.cn/funds/gnzjl/board/10/field/tradezdf/order/desc/page/{}/ajax/1/free/1/",
            "20日排行": "http://data.10jqka.com.cn/funds/gnzjl/board/20/field/tradezdf/order/desc/page/{}/ajax/1/free/1/",
            "即时": "http://data.10jqka.com.cn/funds/gnzjl/field/tradezdf/order/desc/page/{}/ajax/1/free/1/"
        }
        url = url_map.get(symbol, url_map["即时"])

        # 获取数据
        big_df = pd.DataFrame()
        tqdm = get_tqdm()
        for page in tqdm(range(1, int(page_num) + 1), leave=False):
            r = requests.get(url.format(page), headers=get_headers())
            temp_df = pd.read_html(StringIO(r.text))[0]
            big_df = pd.concat(objs=[big_df, temp_df], ignore_index=True)

        # 处理数据
        del big_df["序号"]
        big_df.reset_index(inplace=True)
        big_df["index"] = range(1, len(big_df) + 1)

        # 根据实际列数动态设置列名
        if len(big_df.columns) == 11:  # 即时数据
            columns_map = {
                "index": "序号",
                "行业": "行业",
                "行业指数": "行业指数",
                "涨跌幅": "行业-涨跌幅",
                "流入资金(亿)": "流入资金",
                "流出资金(亿)": "流出资金",
                "净额(亿)": "净额",
                "公司家数": "公司家数",
                "领涨股": "领涨股",
                "涨跌幅.1": "领涨股-涨跌幅",
                "当前价(元)": "当前价"
            }
            big_df = big_df.rename(columns=columns_map)
            
            # 处理百分比数据
            percent_columns = ["行业-涨跌幅", "领涨股-涨跌幅"]
            for col in percent_columns:
                big_df[col] = big_df[col].str.strip("%").astype(float)
                
        elif len(big_df.columns) == 8:  # 其他时间段数据
            columns_map = {
                "index": "序号",
                "行业": "行业",
                "公司家数": "公司家数",
                "行业指数": "行业指数",
                "涨跌幅": "阶段涨跌幅",
                "流入资金(亿)": "流入资金",
                "流出资金(亿)": "流出资金",
                "净额(亿)": "净额"
            }
            big_df = big_df.rename(columns=columns_map)
            
            # 处理百分比数据
            if "阶段涨跌幅" in big_df.columns:
                big_df["阶段涨跌幅"] = big_df["阶段涨跌幅"].str.strip("%").astype(float)
        
        # 转换资金列为数值型
        money_columns = ["流入资金", "流出资金", "净额"]
        for col in money_columns:
            if col in big_df.columns:
                big_df[col] = pd.to_numeric(big_df[col], errors="coerce")
        
        return big_df

    @cache.cache(ttl=60*60*1)
    def get_stock_fund_flow(self, indicator: Literal[ "即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时") -> Dict[str, Dict]:
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

    @cache.cache(ttl=60*60*1)
    def get_top_flow_concept(self, indicator: Literal["即时", "3日排行", "5日排行", "10日排行", "20日排行"]="即时", topn: int=20) -> List[Dict]:
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
                logging.warning(f"无效的indicator值: {indicator}，使用默认顺序")
        
        last_exception = None
        
        # 按顺序尝试每个周期
        for period in retry_sequence:
            try:
                df = self.stock_fund_flow_concept(symbol=period)
                if not df.empty:
                    result = df.head(topn).to_dict(orient="records")
                    if result:  # 确保结果不为空
                        if period != indicator:
                            logging.info(f"使用备选周期 {period} 获取数据成功")
                        return result
            except Exception as e:
                last_exception = e
                logging.warning(f"获取 {period} 周期数据失败: {str(e)}")
                continue
        
        # 如果所有尝试都失败了
        error_msg = f"所有周期都获取失败，最后一次错误: {str(last_exception)}" if last_exception else "所有周期都获取失败"
        logging.error(error_msg)
        return []  # 返回空列表而不是抛出异常，保证服务的稳定性

    @cache.cache(ttl=60*60*1)
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

    def select_stock_by_query(self, query: str):
        """
        根据用户的自然语言查询来筛选股票数据。

        参数:
        query (str): 用户的自然语言查询，描述了股票筛选的条件。

        返回:
        dict: 一个字典，其中键是股票代码，值是该股票的其他信息字符串。

        抛出:
        ValueError: 如果无法从LLM响应中提取Python代码。
        Exception: 如果代码执行失败或结果格式不正确。

        示例:
        >>> select_stock_by_query("5分钟涨跌幅大于1%的股票")
        {'000001': '名称: 平安银行, 现价: 10.5, 涨跌幅: 1.2%, ...', ...}
        """
        df = ak.stock_zh_a_spot_em()
        df_summary = self.data_summarizer.get_data_summary(df)
        global_vars={}
        global_vars["df"]=df
        prompt = f"""
            需要处理的请求：
            {query}

            需要处理的变量名：
            df

            df的摘要如下：
            {df_summary}

            生成一段python代码，完成query的筛选要求
            要求：
            1. 代码用```python   ```包裹
            2. 请求应该跟df的数据过滤相关，如果不相关，返回 
            ```python
            result={{}}
            ```
            3. 对df过滤后，需要把过滤的行处理为Dict[str,str]，赋值给result
            4. 根据query的内容对df进行过滤，例如：
                - 查询："5分钟涨跌幅大于1%的股票"
                - 代码
                ```python
                df_filtered = df[df['5分钟涨跌']>1]
                result = {{}}
                for _, row in df_filtered.iterrows():
                    result[row['代码']] = ", ".join([f"{{col}}: {{row[col]}}" for col in df_filtered.columns if col != '代码'])
                ```
            5. 确保 result 是一个字典，键为股票代码，值为该股票的其他信息字符串
            6. 不要使用任何不在 df 中的列名
            7. 使用名字查询的时候，注意使用模糊查询的方法，避免名字不精确查询不到数据
        """
        new_prompt = prompt
        while True:
            response = self.llm_client.one_chat(new_prompt)
            try:
                code = self._extract_code(response)
                if not code:
                    raise ValueError("No Python code found in the response, 请提供python代码，并包裹在```python  ```之中")
                
                execute_result = self.code_runner.run(code,global_vars=global_vars)
                if execute_result["error"]:
                    raise execute_result["error"]
                if "result" not in execute_result["updated_vars"]:
                    raise Exception("代码执行完以后，没有检测到result变量，必须把结果保存在result变量之中")
                if not isinstance(execute_result["updated_vars"]["result"], dict):
                    raise Exception("result必须是字典格式，请修改代码，把结果保存于字典格式的dict")
                
                return execute_result["updated_vars"]["result"]
            except Exception as e:
                fix_prompt = f"""
                刚刚用下面的提示词
                {prompt}

                生成了下面的代码
                {code}

                发生了下面的错误：
                {str(e)}

                请帮我修正代码，代码要求不变，输出的代码包裹在```python  ```之中
                修正代码不用加任何解释
                """
                new_prompt = fix_prompt

    def _extract_code(self, response):
        """
        从LLM的响应中提取Python代码。

        参数:
        response (str): LLM的完整响应文本

        返回:
        str: 提取出的Python代码。如果没有找到代码，返回空字符串。
        """
        # 使用正则表达式查找被 ```python 和 ``` 包围的代码块
        code_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            # 返回找到的第一个代码块
            return matches[0].strip()
        else:
            # 如果没有找到代码块，可以选择抛出异常或返回空字符串
            # 这里选择返回空字符串
            return ""

    def get_stock_volatility(self, symbol: str, period: int = 30, annualize: bool = True) -> float:
        """
        计算指定股票的波动率。

        参数:
        symbol (str): 股票代码
        period (int): 计算波动率的天数，默认为30天
        annualize (bool): 是否年化波动率，默认为True

        返回:
        float: 计算得到的波动率
        """
        if "." in symbol:
            symbol = symbol.split(".")[0]
        # 获取历史收盘价数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=period)).strftime('%Y%m%d')
        historical_data = self.get_historical_daily_data(symbol, start_date, end_date)

        # 计算日收益率
        returns = historical_data['收盘'].pct_change().dropna()

        # 计算波动率（标准差）
        volatility = returns.std()

        # 如果需要年化，假设一年有250个交易日
        if annualize:
            volatility *= (250 ** 0.5)

        return volatility

    def get_latest_price(self, symbol: str) -> float:
        """
        获取指定股票的最新价格。

        参数:
        symbol (str): 股票代码，例如 "000001" 代表平安银行

        返回:
        float: 股票的最新价格

        异常:
        ValueError: 如果无法获取股票价格
        """
        try:
            # 使用 akshare 的 stock_bid_ask_em 函数获取行情数据
            if "." in symbol:
                symbol= symbol.split(".")[0]
            df = ak.stock_bid_ask_em(symbol=symbol)
            
            # 从返回的数据中提取最新价格
            # 根据数据示例，"最新" 对应的是索引为 20 的行
            latest_price = df.loc[df['item'] == '最新', 'value'].values[0]
            
            return float(latest_price)
        except Exception as e:
            raise ValueError(f"无法获取股票 {symbol} 的最新价格: {str(e)}")

    def get_concept_board_components(self, symbol: str = '车联网') -> dict:
        """
        获取指定概念板块的成分股。参数symbol: str = '车联网' 返回值 dict[symbol,str]

        参数:
            symbol (str): 概念板块名称，例如 "车联网"。可以通过调用 ak.stock_board_concept_name_em() 查看东方财富-概念板块的所有行业名称。

        返回:
            dict: 键为成分股代码，值为该成分股的详细信息，格式化为易于阅读的字符串。
        """
        df = ak.stock_board_concept_cons_em(symbol)

        # 处理数据，将每行数据转换为易于阅读的字符串
        result = {}
        for _, row in df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"最新价: {row['最新价']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"涨跌额: {row['涨跌额']}, "
                f"成交量: {row['成交量']}手, "
                f"成交额: {row['成交额']}, "
                f"振幅: {row['振幅']}%, "
                f"最高: {row['最高']}, "
                f"最低: {row['最低']}, "
                f"今开: {row['今开']}, "
                f"昨收: {row['昨收']}, "
                f"换手率: {row['换手率']}%, "
                f"市盈率-动态: {row['市盈率-动态']}, "
                f"市净率: {row['市净率']}"
            )
            result[row['代码']] = stock_info

        return result

    def select_stocks_by_concept_board_query(self, query: str) -> dict:
        """
        根据用户的自然语言查询来筛选概念板块中的股票数据。参数 query:str 返回dict[symbol,str]

        参数:
        query (str): 用户的自然语言查询，描述了概念板块筛选的条件。

        返回:
        dict: 一个字典，其中键是股票代码，值是该股票的其他信息字符串。

        抛出:
        ValueError: 如果无法从LLM响应中提取Python代码。
        Exception: 如果代码执行失败或结果格式不正确。

        示例:
        >>> select_stocks_by_concept_board_query("涨幅超过2%的板块")
        {'000001': '名称: 平安银行, 现价: 10.5, 涨跌幅: 1.2%, ...', ...}
        """
        # 获取所有概念板块数据
        df_concepts = ak.stock_board_concept_name_em()
        df_summary = self.data_summarizer.get_data_summary(df_concepts)
        global_vars ={"df_concepts":df_concepts}
        prompt = f"""
            需要处理的请求：
            {query}

            需要处理的变量名：
            df_concepts

            df_concepts的摘要如下：
            {df_summary}

            生成一段python代码，完成query的筛选要求
            要求：
            1. 代码用```python   ```包裹
            2. 请求应该跟df_concepts的数据过滤相关，如果不相关，返回 
            ```python
            result = []
            ```
            3. 对df_concepts过滤后，需要把符合条件的板块名提取出来，赋值给result
            4. 根据query的内容对df_concepts进行过滤，例如：
                - 查询："涨幅超过2%的板块"
                - 代码
                ```python
                df_filtered = df_concepts[df_concepts['涨跌幅']>2]
                result = df_filtered['板块名称'].tolist()
                ```
            5. 确保 result 是一个列表，其中包含符合条件的板块名称
            6. 不要使用任何不在 df_concepts 中的列名
            7. 示例代码（根据实际情况调整）：
            ```python
            import re
            keywords = ['科技', '电子', '信息', '通信', '互联网', '软件','人工智能','芯片']
            pattern = '|'.join(keywords)
            mask = df_concepts['板块名称'].str.contains(pattern, case=False, na=False)
            result = df_concepts[mask]['板块名称'].tolist()
            ```
            8. 如果没有完全匹配的结果，考虑返回部分匹配或相关的结果
            9. 添加注释解释你的匹配逻辑
        """
        new_prompt = prompt
        while True:
            response = self.llm_client.one_chat(new_prompt)
            try:
                code = self._extract_code(response)
                if not code:
                    raise ValueError("No Python code found in the response, 请提供python代码，并包裹在```python  ```之中")
                
                execute_result = self.code_runner.run(code,global_vars=global_vars)
                if execute_result["error"]:
                    raise execute_result["error"]
                if "result" not in execute_result["updated_vars"]:
                    raise Exception("代码执行完以后，没有检测到result变量，必须把结果保存在result变量之中")
                if not isinstance(execute_result["updated_vars"]["result"], list):
                    raise Exception("result必须是列表格式，请修改代码，确保返回的是板块名称的列表")
                
                # 获取成分股
                selected_boards = execute_result["updated_vars"]["result"]
                all_stocks = {}
                for board_name in selected_boards:
                    stocks = self.get_concept_board_components(board_name)
                    all_stocks.update(stocks)
                
                return all_stocks
            except Exception as e:
                fix_prompt = f"""
                刚刚用下面的提示词
                {prompt}

                生成了下面的代码
                {code}

                发生了下面的错误：
                {str(e)}

                请帮我修正代码，代码要求不变，输出的代码包裹在```python  ```之中
                修正代码不用加任何解释
                """
                new_prompt = fix_prompt

    def get_board_industry_components(self, symbol: str) -> dict:
        """
        获取指定行业板块的成分股。参数 symbol: str 返回 Dict[symbol,str]

        参数:
            symbol (str): 行业板块名称，例如 "小金属"。可以通过调用 ak.stock_board_industry_name_em() 查看东方财富-行业板块的所有行业名称。

        返回:
            dict: 键为成分股代码，值为该成分股的详细信息，格式化为易于阅读的字符串。
        """
        df = ak.stock_board_industry_cons_em(symbol)

        # 处理数据，将每行数据转换为易于阅读的字符串
        result = {}
        for _, row in df.iterrows():
            stock_info = (
                f"名称: {row['名称']}, "
                f"最新价: {row['最新价']}, "
                f"涨跌幅: {row['涨跌幅']}%, "
                f"涨跌额: {row['涨跌额']}, "
                f"成交量: {row['成交量']}手, "
                f"成交额: {row['成交额']}, "
                f"振幅: {row['振幅']}%, "
                f"最高: {row['最高']}, "
                f"最低: {row['最低']}, "
                f"今开: {row['今开']}, "
                f"昨收: {row['昨收']}, "
                f"换手率: {row['换手率']}%, "
                f"市盈率-动态: {row['市盈率-动态']}, "
                f"市净率: {row['市净率']}"
            )
            result[row['代码']] = stock_info

        return result

    def select_stocks_by_industry_board_query(self, query: str) -> dict:
        """
        根据用户的自然语言查询来筛选行业板块中的股票数据。参数 query: str 返回 Dict[symbol,str]

        参数:
        query (str): 用户的自然语言查询，描述了行业板块筛选的条件。

        返回:
        dict: 一个字典，其中键是股票代码，值是该股票的其他信息字符串。

        抛出:
        ValueError: 如果无法从LLM响应中提取Python代码。
        Exception: 如果代码执行失败或结果格式不正确。

        示例:
        >>> select_stocks_by_industry_board_query("涨幅超过2%的板块")
        {'000001': '名称: 平安银行, 现价: 10.5, 涨跌幅: 1.2%, ...', ...}
        """
        # 获取所有行业板块数据
        df_industries = ak.stock_board_industry_name_em()
        df_summary = self.data_summarizer.get_data_summary(df_industries)
        global_vars={"df_industries":df_industries}
        prompt = f"""
            需要处理的请求：
            {query}

            需要处理的变量名：
            df_industries

            df_industries的摘要如下：
            {df_summary}

            生成一段python代码，完成query的筛选要求
            要求：
            1. 代码用```python   ```包裹
            2. 请求应该跟df_industries的数据过滤相关，如果不相关，返回 
            ```python
            result = []
            ```
            3. 对df_industries过滤后，需要把符合条件的板块名提取出来，赋值给result
            4. 根据query的内容对df_industries进行过滤，考虑以下几点：
            - 使用更灵活的匹配方式，如模糊匹配或相关词匹配
            - 考虑同义词或相关词，例如"科技"可能与"电子"、"信息"、"通信"等相关
            - 可以使用正则表达式进行更复杂的匹配
            5. 确保 result 是一个列表，其中包含符合条件的板块名称
            6. 不要使用任何不在 df_industries 中的列名
            7. 示例代码（根据实际情况调整）：
            ```python
            import re
            keywords = ['科技', '电子', '信息', '通信', '互联网', '软件','人工智能','芯片']
            pattern = '|'.join(keywords)
            mask = df_industries['板块名称'].str.contains(pattern, case=False, na=False)
            result = df_industries[mask]['板块名称'].tolist()
            ```
            8. 如果没有完全匹配的结果，考虑返回部分匹配或相关的结果
            9. 添加注释解释你的匹配逻辑
        """
        new_prompt = prompt
        while True:
            response = self.llm_client.one_chat(new_prompt)
            try:
                code = self._extract_code(response)
                if not code:
                    raise ValueError("No Python code found in the response, 请提供python代码，并包裹在```python  ```之中")
                
                execute_result = self.code_runner.run(code,global_vars=global_vars)
                if execute_result["error"]:
                    raise execute_result["error"]
                if "result" not in execute_result["updated_vars"]:
                    raise Exception("代码执行完以后，没有检测到result变量，必须把结果保存在result变量之中")
                if not isinstance(execute_result["updated_vars"]["result"], list):
                    raise Exception("result必须是列表格式，请修改代码，确保返回的是板块名称的列表")
                
                # 获取成分股
                selected_boards = execute_result["updated_vars"]["result"]
                all_stocks = {}
                for board_name in selected_boards:
                    stocks = self.get_board_industry_components(board_name)
                    all_stocks.update(stocks)
                
                return all_stocks
            except Exception as e:
                fix_prompt = f"""
                刚刚用下面的提示词
                {prompt}

                生成了下面的代码
                {code}

                发生了下面的错误：
                {str(e)}

                请帮我修正代码，代码要求不变，输出的代码包裹在```python  ```之中
                修正代码不用加任何解释
                """
                new_prompt = fix_prompt

    def select_by_query(self, 
                        data_source: Union[pd.DataFrame, Callable[[], pd.DataFrame]], 
                        query: str, 
                        result_type: str = 'dict', 
                        key_column: str = None, 
                        value_columns: List[str] = None) -> Union[Dict[str, str], List[str]]:
        """
        根据用户的自然语言查询来筛选数据。

        参数:
        data_source (Union[pd.DataFrame, Callable[[], pd.DataFrame]]): 
            数据源，可以是一个DataFrame或者是返回DataFrame的无参数函数。
        query (str): 用户的自然语言查询，描述了筛选的条件。
        result_type (str): 结果类型，'dict' 或 'list'。默认为 'dict'。
        key_column (str): 如果result_type为'dict'，这个参数指定作为字典键的列名。
        value_columns (List[str]): 如果result_type为'dict'，这个参数指定作为字典值的列名列表。

        返回:
        Union[Dict[str, str], List[str]]: 
            如果result_type为'dict'，返回一个字典，其中键是key_column指定的列的值，
            值是value_columns指定的列的值组成的字符串。
            如果result_type为'list'，返回一个列表，包含符合条件的行的第一列值。

        抛出:
        ValueError: 如果无法从LLM响应中提取Python代码。
        Exception: 如果代码执行失败或结果格式不正确。

        示例:
        >>> df = pd.DataFrame({'code': ['000001', '000002'], 'name': ['平安银行', '万科A'], 'price': [10.5, 15.2]})
        >>> select_by_query(df, "价格大于12的股票", 'dict', 'code', ['name', 'price'])
        {'000002': '名称: 万科A, 价格: 15.2'}
        >>> select_by_query(df, "价格大于12的股票", 'list')
        ['000002']
        """
        if callable(data_source):
            df = data_source()
        else:
            df = data_source

        df_summary = self.data_summarizer.get_data_summary(df)
        global_vars={"df":df}
        prompt = f"""
            需要处理的请求：
            {query}

            需要处理的变量名：
            df

            df的摘要如下：
            {df_summary}

            生成一段python代码，完成query的筛选要求
            要求：
            1. 代码用```python   ```包裹
            2. 请求应该跟df的数据过滤相关，如果不相关，返回 
            ```python
            result = []
            ```
            3. 对df过滤后，需要把符合条件的行赋值给result
            4. 根据query的内容对df进行过滤，例如：
                - 查询："价格大于12的股票"
                - 代码
                ```python
                result = df[df['price'] > 12]
                ```
            5. 确保 result 是一个DataFrame，包含符合条件的所有行
            6. 不要使用任何不在 df 中的列名
            7. 使用名字查询的时候，注意使用模糊查询的方法，避免名字不精确查询不到数据
        """
        new_prompt = prompt
        while True:
            response = self.llm_client.one_chat(new_prompt)
            try:
                code = self._extract_code(response)
                if not code:
                    raise ValueError("No Python code found in the response, 请提供python代码，并包裹在```python  ```之中")
                
                execute_result = self.code_runner.run(code,global_vars=global_vars)
                if execute_result["error"]:
                    raise execute_result["error"]
                if "result" not in execute_result["updated_vars"]:
                    raise Exception("代码执行完以后，没有检测到result变量，必须把结果保存在result变量之中")
                if not isinstance(execute_result["updated_vars"]["result"], pd.DataFrame):
                    raise Exception("result必须是DataFrame格式，请修改代码，确保返回的是筛选后的DataFrame")
                
                filtered_df = execute_result["updated_vars"]["result"]
                
                if result_type == 'dict':
                    if not key_column or not value_columns:
                        raise ValueError("For dict result type, key_column and value_columns must be specified")
                    return {
                        row[key_column]: ", ".join([f"{col}: {row[col]}" for col in value_columns])
                        for _, row in filtered_df.iterrows()
                    }
                elif result_type == 'list':
                    return filtered_df.iloc[:, 0].tolist()
                else:
                    raise ValueError("Invalid result_type. Must be 'dict' or 'list'")
            
            except Exception as e:
                fix_prompt = f"""
                刚刚用下面的提示词
                {prompt}

                生成了下面的代码
                {code}

                发生了下面的错误：
                {str(e)}

                请帮我修正代码，代码要求不变，输出的代码包裹在```python  ```之中
                修正代码不用加任何解释
                """
                new_prompt = fix_prompt

    def select_by_stock_comments(self, query: str) -> dict:
        """
        根据用户的查询条件筛选千股千评数据。参数query:str 返回Dict[symbol,str]

        此函数使用 akshare 的 stock_comment_em 函数获取千股千评数据，
        然后使用 select_by_query 方法根据用户的查询条件进行筛选。

        参数:
        query (str): 用户的自然语言查询，描述了筛选千股千评数据的条件。

        返回:
        dict: 一个字典，其中键是股票代码，值是该股票的其他信息字符串。
              返回的信息包括：名称、最新价、涨跌幅、换手率、市盈率、主力成本、
              机构参与度、综合得分、排名变化、当前排名、关注指数和交易日。

        示例:
        >>> select_by_stock_comments("综合得分大于80的股票")
        {'000001': '名称: 平安银行, 最新价: 10.5, 涨跌幅: 1.2%, ...', ...}
        """
        # 获取千股千评数据
        df = ak.stock_comment_em()
        
        # 定义要包含在结果中的列
        value_columns = [
            "名称", "最新价", "涨跌幅", "换手率", "市盈率", "主力成本", 
            "机构参与度", "综合得分", "上升", "目前排名", "关注指数", "交易日"
        ]
        
        # 使用 select_by_query 方法进行筛选
        result = self.select_by_query(
            df, 
            query, 
            result_type='dict', 
            key_column="代码", 
            value_columns=value_columns
        )
        
        return result

    def remove_prefix(self,code: str) -> str:
        """移除股票代码的前缀"""
        return code.lstrip('SH').lstrip('SZ').lstrip('BJ')

    @cache.cache(ttl=60*60*7*24)
    def get_stock_hot_keyword(self, symbol: str) -> list:
        """获取股票的热门关键词，参数 symbol: str 返回值 dict"""
        symbol = self.add_market_prefix(symbol)
        try:
            df = ak.stock_hot_keyword_em(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error in get_stock_hot_keyword for stock {symbol}: {str(e)} {format_exc()}")
            return []
        df = df[["概念名称","热度"]]
        return df.to_dict(orient="records")

    def get_xueqiu_hot_follow(self, num: int = 100) -> dict:
        """获取雪球关注排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
        df = ak.stock_hot_follow_xq(symbol="最热门")
        result = {}
        for _, row in df.head(num).iterrows():
            code = self.remove_prefix(row['股票代码'])
            info = f"股票简称: {row['股票简称']}, 关注: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
            result[code] = info
        return result

    def get_xueqiu_hot_tweet(self, num: int = 100) -> dict:
        """获取雪球讨论排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
        df = ak.stock_hot_tweet_xq(symbol="最热门")
        result = {}
        for _, row in df.head(num).iterrows():
            code = self.remove_prefix(row['股票代码'])
            info = f"股票简称: {row['股票简称']}, 讨论: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
            result[code] = info
        return result

    def get_xueqiu_hot_deal(self, num: int = 100) -> dict:
        """获取雪球交易排行榜,参数num: int = 100，返回值Dict[symbol,str]"""
        df = ak.stock_hot_deal_xq(symbol="最热门")
        result = {}
        for _, row in df.head(num).iterrows():
            code = self.remove_prefix(row['股票代码'])
            info = f"股票简称: {row['股票简称']}, 交易: {row['关注']:.0f}, 最新价: {row['最新价']:.2f}"
            result[code] = info
        return result

    def get_wencai_hot_rank(self, num: int = 100) -> dict:
        """获取问财热门股票排名,参数num: int = 100，返回值Dict[symbol,str]"""
        date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")  # 获取昨天的日期
        df = ak.stock_hot_rank_wc(date=date)
        result = {}
        for _, row in df.head(num).iterrows():
            code = row['股票代码']
            info = f"股票简称: {row['股票简称']}, 现价: {row['现价']:.2f}, 涨跌幅: {row['涨跌幅']:.2f}%, 热度: {row['个股热度']:.0f}, 排名: {row['个股热度排名']}"
            result[code] = info
        return result

    def get_eastmoney_hot_rank(self, num: int = 100) -> dict:
        """获取东方财富人气榜-A股,参数num: int = 100，返回值Dict[symbol,str]"""
        df = ak.stock_hot_rank_em()
        result = {}
        for _, row in df.head(num).iterrows():
            code = self.remove_prefix(row['代码'])
            info = f"股票名称: {row['股票名称']}, 最新价: {row['最新价']:.2f}, 涨跌额: {row['涨跌额']:.2f}, 涨跌幅: {row['涨跌幅']:.2f}%"
            result[code] = info
        return result
    
    def rremove_prefix(self, code: str) -> str:
        """
        使用正则表达式移除股票代码的前缀（如 'SH'、'SZ'、'BJ'），不区分大小写。

        参数:
        code (str): 原始股票代码

        返回:
        str: 移除前缀后的股票代码
        """
        # 正则表达式模式：匹配开头的 SH, SZ, 或 BJ，不区分大小写
        pattern = r'^(sh|sz|bj)'
        
        # 使用 re.sub 来替换匹配的前缀为空字符串
        return re.sub(pattern, '', code, flags=re.IGNORECASE)

    def get_baidu_hotrank(self, hour=7, num=20) -> dict:
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
        hotlist = self.baidu_news_api.fetch_hotrank(day=date, hour=hour, rn=num)
        
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

    def get_combined_hot_stocks(self, num: int = 100) -> List[str]:
        """
        获取综合的热门股票列表，包括雪球讨论、雪球交易、问财热门、东方财富人气榜和百度热榜。

        参数:
        num (int): 从每个来源获取的股票数量，默认为100。

        返回:
        List[str]: 交叉合并后的去重股票代码列表。
        """
        symbol_dict = self.get_code_name()

        wencai = []

        # 获取各个来源的热门股票
        xueqiu_tweet = list(self.get_xueqiu_hot_tweet(num).keys())
        xueqiu_deal = list(self.get_xueqiu_hot_deal(num).keys())
        eastmoney = list(self.get_eastmoney_hot_rank(num).keys())
        baidu = list(self.get_baidu_hotrank(num=num).keys())
        try:
            wencai = list(self.get_wencai_hot_rank(num).keys())
        except Exception as e:
            pass
        
        # 将所有列表合并到一个列表中
        all_lists = [xueqiu_tweet, xueqiu_deal, eastmoney, baidu]
        if len(wencai)>0:
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
                    if stock not in seen and stock in symbol_dict:
                        seen.add(stock)
                        result.append(stock)

        return result

    def get_baidu_recommendation(self, hour=7, num=20) -> dict:
        """
        获取百度股票推荐列表。参数  hour=7, num=20 返回 Dict[symbol,str]

        参数:
        hour (int): 获取最近多少小时的数据，默认为7小时。
        num (int): 需要获取的推荐股票数量，默认为20。

        返回:
        dict: 键为股票代码，值为格式化的字符串，包含推荐股票的详细信息。
        """
        # 获取当前日期
        date = datetime.now().strftime("%Y%m%d")
        
        # 获取推荐股票列表
        rlist = self.baidu_news_api.fetch_recommendation_list(day=date, hour=hour, rn=num)
        
        # 格式化结果
        result = {}
        for rank, item in enumerate(rlist, 1):  # enumerate从1开始，给每个项目一个排名
            stock_info = (
                f"排名: {rank}\n"
                f"股票名称: {item['name']}\n"
                f"涨跌幅: {item['change']}\n"
                f"综合热度: {item['heat']}\n"
                f"所属板块: {item['sector_name']}\n"
                f"排名变化: {item['rank_change']}\n"
                f"是否连续上榜: {item['continued_ranking']}\n"
                f"----------------------"
            )
            result[item['code']] = stock_info
        
        return result

    def get_vote_baidu(self, symbol: str) -> str:
        """
        获取百度股市通的股票投票数据。参数 symbol:str 返回str

        参数:
        symbol (str): A股股票代码，例如 "000001"

        返回:
        str: 格式化的投票数据字符串
        """
        try:
            # 获取投票数据
            df = ak.stock_zh_vote_baidu(symbol=symbol, indicator="股票")
            
            # 格式化DataFrame为字符串
            result = f"股票代码 {symbol} 的投票数据：\n"
            for _, row in df.iterrows():
                result += (f"{row['周期']}：看涨 {row['看涨']}，看跌 {row['看跌']}，"
                        f"看涨比例 {row['看涨比例']}，看跌比例 {row['看跌比例']}\n")
            
            return result.strip()  # 移除末尾的换行符
        
        except Exception as e:
            return f"获取股票 {symbol} 的投票数据时发生错误: {str(e)}"

    def get_baidu_sentiment_rank(self, num=20) -> dict:
        """
        获取百度股票情绪排名。参数 num=20 返回 Dict[symbol,str]

        参数:
        num (int): 需要获取的股票数量，默认为20。

        返回:
        dict: 键为股票代码，值为格式化的字符串，包含股票的情绪排名信息。
        """
        # 获取情绪排名数据
        sentiment_list = self.baidu_news_api.fetch_sentiment_rank(rn=num)
        
        # 格式化结果
        result = {}
        for rank, item in enumerate(sentiment_list, 1):
            stock_info = (
                f"排名: {rank}\n"
                f"股票名称: {item['name']}\n"
                f"股票代码: {item['code']}\n"
                f"交易所: {item['exchange']}\n"
                f"市场: {item['market']}\n"
                f"所属板块: {item['plate']} ({item['plateCode']})\n"
                f"排名变化: {item['rankDiff']}\n"
                f"比率: {item['ratio']}\n"
                f"热度: {item['heat']}\n"
                f"利好新闻占比: {item['goodNewsPercent']}\n"
                f"中性新闻占比: {item['middleNewsPercent']}\n"
                f"利空新闻占比: {item['badNewsPercent']}\n"
                f"----------------------"
            )
            result[item['code']] = stock_info
        
        return result

    def get_baidu_analysis_rank(self, num=20) -> dict:
        """
        获取百度股票分析排名。参数 num=20 返回 Dict[symbol,str]

        参数:
        num (int): 需要获取的股票数量，默认为20。

        返回:
        dict: 键为股票代码，值为格式化的字符串，包含股票的分析排名信息。
        """
        # 获取分析排名数据
        analysis_list = self.baidu_news_api.fetch_analysis_rank(rn=num)
        
        # 格式化结果
        result = {}
        for rank, item in enumerate(analysis_list, 1):
            stock_info = (
                f"排名: {rank}\n"
                f"股票名称: {item['name']}\n"
                f"股票代码: {item['code']}\n"
                f"市场: {item['market']}\n"
                f"排名变化: {item['rank_change']}\n"
                f"综合得分: {item['synthesis_score']}\n"
                f"技术得分: {item['technology_score']}\n"
                f"资金得分: {item['capital_score']}\n"
                f"市场得分: {item['market_score']}\n"
                f"财务得分: {item['finance_score']}\n"
                f"所属板块: {item['sector']} ({item['sector_code']})\n"
                f"市场类型: {item['market_type']}\n"
                f"----------------------"
            )
            result[item['code']] = stock_info
        
        return result

    @cache.cache(ttl=60*60*4)
    def get_baidu_analysis_summary(self, symbol: str) -> str:
        """
        获取百度股票分析摘要。参数 symbol: str 返回 Dict[symbol,str]

        参数:
        symbol (str): 股票代码，例如 '000725'。

        返回:
        str: 格式化的字符串，包含股票的详细分析信息。
        """
        # 确定市场类型
        if symbol.startswith('HK'):
            market = 'hk'
        elif symbol.isalpha() or (symbol.isalnum() and not symbol.isdigit()):
            market = 'us'
        else:
            market = 'ab'  # 所有 A 股

        # 获取分析数据
        analysis_data = self.baidu_news_api.fetch_analysis(code=symbol, market=market)

        if analysis_data == "数据不可用":
            return "数据不可用"
        
        # 格式化结果
        formatted_analysis = (
            f"股票代码: {symbol}\n"
            f"市场: {'A股' if market == 'ab' else '港股' if market == 'hk' else '美股'}\n\n"
            f"{analysis_data}"
        )
        
        return formatted_analysis

    def get_baidu_stock_news(self, symbol: str, num: int = 20) -> List[str]:
        """
        获取指定股票的百度快讯新闻。 参数 symbol: str, num: int = 20 返回 Dict[symbol,str]

        参数:
        symbol (str): 股票代码，例如 '000725'。
        num (int): 需要获取的新闻数量，默认为20。

        返回:
        List[str]: 包含格式化新闻信息的字符串列表。
        """
        # 获取快讯新闻数据
        news_list = self.baidu_news_api.fetch_express_news(rn=num, code=symbol)
        
        # 格式化结果
        result = []
        for news_item in news_list:
            news_time = news_item['ptime']
            news_info = (
                f"发布时间: {news_time}\n"
                f"标题: {news_item['title']}\n"
                f"内容: {news_item['content']}\n"
                f"标签: {news_item['tag']}\n"
                f"来源: {news_item['provider']}\n"
                f"----------------------"
            )
            result.append(news_info)
        
        return result

    def get_baidu_market_news(self, num: int = 40) -> List[str]:
        """
        获取百度A股市场快讯新闻。参数 num: int = 40 返回 Dict[symbol,str]

        参数:
        num (int): 需要获取的新闻数量，默认为40。

        返回:
        List[str]: 包含格式化新闻信息的字符串列表。
        """
        # 获取快讯新闻数据
        news_list = self.baidu_news_api.fetch_express_news_v2(rn=num, pn=0, tag='A股')
        
        # 格式化结果
        result = []
        for news_item in news_list:
            news_time = news_item['ptime']
            news_info = (
                f"发布时间: {news_time}\n"
                f"标题: {news_item['title']}\n"
                f"内容: {news_item['content']}\n"
                f"标签: {news_item['tag']}\n"
                f"来源: {news_item['provider']}\n"
                f"----------------------"
            )
            result.append(news_info)
        
        return result

    def get_baidu_important_news(self, num: int = 200) -> List[str]:
        """
        获取重要市场新闻。参数 num: int = 200 返回 Dict[symbol,str]

        参数:
        num (int): 需要获取的新闻数量，默认为40。

        返回:
        List[str]: 包含格式化新闻信息的字符串列表。
        """
        # 获取重要新闻数据
        news_list = self.baidu_news_api.fetch_express_news_v2(rn=num, pn=0, tag='重要')
        
        # 格式化结果
        result = []
        for news_item in news_list:
            news_time = news_item['ptime']
            news_info = (
                f"发布时间: {news_time}\n"
                f"标题: {news_item['title']}\n"
                f"内容: {news_item['content']}\n"
                f"标签: {news_item['tag']}\n"
                f"来源: {news_item['provider']}\n"
                f"----------------------"
            )
            result.append(news_info)
        
        return result

    def summarizer_news(self, news_source: List[str], query: str = "总结市场热点,市场机会,市场风险", max_word: int = 500) -> str:
        def chunk_text(text_list: List[str], max_chars: int = 10000) -> List[str]:
            chunks = []
            current_chunk = ""
            for text in text_list:
                if len(current_chunk) + len(text) <= max_chars:
                    current_chunk += text + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = text + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        def summarize_chunk(chunk: str) -> str:
            prompt = f"请根据以下查询要求总结这段新闻内容：\n\n查询：{query}\n\n新闻内容：\n{chunk}\n\n总结："
            try:
                return self.llm_client.one_chat(prompt)
            except Exception as e:
                from core.utils.config_setting import Config
                config = Config()
                if config.has_key("deep_seek_api_key"):
                    from core.utils.lazy import lazy
                    module = lazy("core.llms.simple_deep_seek_client")
                    llm_client = module.SimpleDeepSeekClient()
                    return llm_client.one_chat(prompt)
                return chunk  # 如果出错，返回原始内容

        async def async_summarize():
            # 将新闻分成不超过10000字符的块
            news_chunks = chunk_text(news_source)
            
            # 使用QueryExecutor进行并发摘要
            executor = QueryExecutor(summarize_chunk)
            summaries = await executor.concurrent_query(news_chunks)
            
            # 如果摘要总长度已经小于max_word，直接返回
            if sum(len(s) for s in summaries) <= max_word:
                return " ".join(summaries)
            
            # 否则，继续进行摘要，直到总长度不超过max_word
            while sum(len(s) for s in summaries) > max_word:
                if len(summaries) == 1:
                    # 如果只剩一个摘要但仍然超过max_word，进行最后一次摘要
                    final_prompt = f"请将以下摘要进一步压缩到不超过{max_word}个字：\n\n{summaries[0]}"
                    return self.llm_client.one_chat(final_prompt)[:max_word]
                
                # 将现有的摘要分成3个一组进行进一步摘要
                combined_summaries = [summaries[i] + " " + summaries[i+1] if i + 1 < len(summaries) else summaries[i] for i in range(0, len(summaries), 3)]
                summaries = await executor.concurrent_query(combined_summaries)
            
            # 返回最终的摘要
            return " ".join(summaries)[:max_word]

        # 使用同步方式运行异步函数
        return asyncio.run(async_summarize())

    def summarizer_news_sync(self, news_source: list[str], query: str="总结市场热点,市场机会,市场风险", max_word: int = 500) -> str:
        """
        对给定的新闻列表进行摘要，根据指定的查询要求生成一个简洁的总结。 参数news_source: list[str], query: str="总结市场热点,市场机会,市场风险", max_word: int = 240 返回值str

        这个函数首先将新闻文本分成较小的块，然后对每个块进行摘要。如果摘要的总长度超过指定的最大字数，
        它会继续进行迭代摘要，直到得到一个不超过最大字数的最终摘要。

        参数:
        news_source (list[str]): 包含新闻文本的字符串列表。每个字符串应该是一条完整的新闻。
        query (str, 可选): 指定摘要的重点或方向。默认为"总结市场热点,市场机会,市场风险"。
        max_word (int, 可选): 最终摘要的最大字数。默认为240。

        返回:
        str: 不超过指定最大字数的新闻摘要。

        示例:
        >>> news = ["今日股市大涨，科技股领涨。", "央行宣布降息，刺激经济增长。", "新能源车企发布新品，股价应声上涨。"]
        >>> summary = stock_data_provider.summarizer_news(news, "分析今日股市表现", 100)
        >>> print(summary)
        """
        def chunk_text(text_list: list[str], max_chars: int = 10000) -> list[str]:
            chunks = []
            current_chunk = ""
            for text in text_list:
                if len(current_chunk) + len(text) <= max_chars:
                    current_chunk += text + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = text + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        def summarize_chunk(chunk: str, query: str) -> str:
            prompt = f"请根据以下查询要求总结这段新闻内容：\n\n查询：{query}\n\n新闻内容：\n{chunk}\n\n总结："
            try:
                return self.llm_client.one_chat(prompt)
            except Exception as e:
                from core.utils.config_setting import Config
                config=Config()
                if config.has_key("deep_seek_api_key"):
                    from core.utils.lazy import lazy
                    module=lazy("core.llms.simple_deep_seek_client")
                    llm_client=module.SimpleDeepSeekClient()
                    return llm_client.one_chat(prompt)
                return chunk  # 如果出错，返回原始内容

        # 将新闻分成不超过10000字符的块
        news_chunks = chunk_text(news_source)
        
        # 对每个块进行摘要
        summaries = [summarize_chunk(chunk, query) for chunk in news_chunks]
        
        # 如果摘要总长度已经小于max_word，直接返回
        if sum(len(s) for s in summaries) <= max_word:
            return " ".join(summaries)
        
        # 否则，继续进行摘要，直到总长度不超过max_word
        while sum(len(s) for s in summaries) > max_word:
            if len(summaries) == 1:
                # 如果只剩一个摘要但仍然超过max_word，进行最后一次摘要
                final_prompt = f"请将以下摘要进一步压缩到不超过{max_word}个字：\n\n{summaries[0]}"
                return self.llm_client.one_chat(final_prompt)[:max_word]
            
            # 将现有的摘要分成两两一组进行进一步摘要
            new_summaries = []
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    combined = summaries[i] + " " + summaries[i+1]
                    new_summary = summarize_chunk(combined, query)
                    new_summaries.append(new_summary)
                else:
                    new_summaries.append(summaries[i])
            
            summaries = new_summaries
        
        # 返回最终的摘要
        return " ".join(summaries)[:max_word]

    def replace_sensitive_subtitle(self,contents:list[str])->list[str]:
        sensitive_subtitle_pair=[{"近日":"最近"}]
        # 替换敏感词
        for pair in sensitive_subtitle_pair:
            for key, value in pair.items():
                contents = [content.replace(key, value) for content in contents]
        return contents

    def get_stock_qa_data_q(self, symbol: str) -> List[Dict[Any, Any]]:
        """
        获取指定股票的互动易问答数据

        Args:
            symbol (str): 股票代码，例如"002594"

        Returns:
            List[Dict]: 包含问答数据的列表，每个元素为一条问答记录的字典
            字典包含以下字段:
                - 股票代码: str
                - 公司简称: str
                - 行业: str
                - 行业代码: str
                - 问题: str
                - 提问者: str
                - 来源: str
                - 提问时间: datetime
                - 更新时间: datetime
                - 提问者编号: str
                - 问题编号: str
                - 回答ID: str
                - 回答内容: str
                - 回答者: str

        Examples:
            >>> crawler = StockQADataCrawler()
            >>> data = crawler.get_stock_qa_data("002594")
            >>> print(f"获取到 {len(data)} 条问答数据")
        """
        try:
            # 通过akshare获取数据
            df = ak.stock_irm_cninfo(symbol=symbol)
            
            # 确保数据不为空
            if df.empty:
                return []
                
            # 将DataFrame转换为List[Dict]格式
            records = df.to_dict('records')
            
            # 处理None值，确保数据的一致性
            for record in records:
                for key in record:
                    if pd.isna(record[key]):
                        record[key] = None
            
            return records
            
        except Exception as e:
            print(f"获取股票{symbol}的互动易数据时发生错误: {str(e)}")
            return []

    def get_answer_detail(self, question_id: str) -> List[Dict[Any, Any]]:
        """
        获取特定问题的回答详情

        Args:
            question_id (str): 问题ID，例如"1495108801386602496"
                             可通过 get_stock_qa_data 方法返回的提问者编号获取

        Returns:
            List[Dict]: 包含回答详情的列表，每个元素为一条回答记录的字典
            字典包含以下字段:
                - 股票代码: str
                - 公司简称: str
                - 问题: str
                - 回答内容: str
                - 提问者: str
                - 提问时间: datetime
                - 回答时间: datetime

        Examples:
            >>> crawler = StockQADataCrawler()
            >>> answer = crawler.get_answer_detail("1495108801386602496")
            >>> if answer:
            >>>     print(f"回答时间: {answer[0].get('回答时间')}")
        """
        try:
            # 通过akshare获取回答数据
            df = ak.stock_irm_ans_cninfo(symbol=question_id)
            
            # 确保数据不为空
            if df.empty:
                return []
            
            # 将DataFrame转换为List[Dict]格式
            records = df.to_dict('records')
            
            # 处理None值和时间格式
            for record in records:
                for key in record:
                    if pd.isna(record[key]):
                        record[key] = None
                    # 确保时间字段为datetime对象
                    elif key in ['提问时间', '回答时间'] and record[key] is not None:
                        if isinstance(record[key], str):
                            record[key] = pd.to_datetime(record[key])
            
            return records
            
        except Exception as e:
            print(f"获取问题{question_id}的回答详情时发生错误: {str(e)}")
            return []

    def get_sse_qa_data(self, symbol: str) -> List[Dict[Any, Any]]:
        """
        获取指定股票的上证e互动问答数据 (上交所)

        Args:
            symbol (str): 股票代码，例如"603119"

        Returns:
            List[Dict]: 包含问答数据的列表，每个元素为一条问答记录的字典
            字典包含以下字段:
                - 股票代码: str
                - 公司简称: str
                - 问题: str
                - 回答: str
                - 问题时间: str
                - 回答时间: str
                - 问题来源: str
                - 回答来源: str
                - 用户名: str

        Examples:
            >>> crawler = StockQADataCrawler()
            >>> data = crawler.get_sse_qa_data("603119")
            >>> print(f"获取到 {len(data)} 条e互动数据")
        """
        try:
            # 通过akshare获取上证e互动数据
            df = ak.stock_sns_sseinfo(symbol=symbol)
            
            # 确保数据不为空
            if df.empty:
                return []
                
            # 将DataFrame转换为List[Dict]格式
            records = df.to_dict('records')
            
            # 处理None值，确保数据的一致性
            for record in records:
                for key in record:
                    if pd.isna(record[key]):
                        record[key] = None
            
            return records
            
        except Exception as e:
            print(f"获取股票{symbol}的上证e互动数据时发生错误: {str(e)}")
            return []

    def extract_json_from_text(self, text: str, max_attempts: int = 4) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        从文本中提取JSON对象并返回字典或字典列表。允许多次尝试修复JSON。

        :param text: 包含JSON数据的字符串。
        :param max_attempts: 最大修复尝试次数，默认为3。
        :return: 解析后的JSON对象（字典或字典列表）。
        :raises JSONDecodeError: 如果在多次尝试后仍未能找到有效的JSON数据。
        """
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if not json_match:
            json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            json_match = re.search(r'\[[\s\S]*\]', text)
        
        if json_match:
            json_str = json_match.group(1) if '```json' in json_match.group() else json_match.group()
            json_str = json_str.replace("'", "\"")
            
            for attempt in range(max_attempts):
                try:
                    return json.loads(json_str, strict=False)
                except Exception as err:
                    if attempt == max_attempts - 1:
                        raise err
                    
                    fix_prompt = f"""
                    以下JSON字符串解析时发生错误：

                    ```json
                    {json_str}
                    ```

                    错误信息：{str(err)}

                    请帮我修复这个JSON。要求：

                    尽可能保持原始数据不变。
                    只修复导致解析错误的问题，不要改变有效的数据结构和值。
                    特别注意检查并修复以下常见问题：
                    缺少逗号
                    多余的逗号
                    未闭合的引号
                    未闭合的括号
                    双斜杠注释
                    内部嵌套的引号冲突（请将内层引号改为不同符号或使用转义字符）
                    如果有多个错误，请尝试一次性修复所有错误。
                    返回修复后的完整JSON，用```json  ```包裹。
                    不要添加任何解释，只返回修复后的JSON。
                    """
                    
                    response = self.llm_client.one_chat(fix_prompt)
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response  # 如果没有找到```json```包裹，就使用整个响应
                    json_str = json_str.replace("'", "\"")
        
        raise json.JSONDecodeError("No valid JSON found in the text", text, 0)

    def get_function_list(self):
        """
        返回可用函数列表的字符串。

        Returns:
            str: 可用函数列表，以逗号分隔。
        """
        function_list = [
            func_name for func_name in dir(self)
            if not func_name.startswith("__") and callable(getattr(self, func_name))
        ]
        return ", ".join(function_list)

    def get_available_functions(self):
        """
        返回可用函数名称的列表。

        Returns:
            list: 可用函数名称的列表。
        """
        function_list = [
            func_name for func_name in dir(self)
            if not func_name.startswith("__") and callable(getattr(self, func_name))
        ]
        return function_list

    def has_function(self, func_name):
        """
        判断函数是否可用。

        Args:
            func_name (str): 函数名称。

        Returns:
            bool: 如果函数可用，则返回 True，否则返回 False。
        """
        return (
            not func_name.startswith("__")
            and callable(getattr(self, func_name, None))
        )


    def get_self_description(self)->str:
        prompt="""
        适合用于选择股票范围的函数(只有get_code_name函数只返回名字和代码，其余函数均会返回数据，方便进行下一次筛选)：
            - select_stock_by_query                     使用最新行情数据，用自然语言进行筛选，返回值包含股票当前的数据信息
            - get_index_components                      输入指数代码，获得成分列表，返回值只有名字和代码
            - get_code_name                             全部股票代码的字典，返回值只有名字和代码
            - get_full_realtime_data                    获取全部股票目前的行情，返回Dict[Symnol,行情描述字符串]
            - get_concept_board_components              用自然语言进行概念板块选股，query只能针对板块的名字和板块的数据，返回值包含股票当前的数据信息
            - select_stocks_by_industry_board_query     用自然语言进行行业板块选股,query只能针对板块的名字和板块的数据，返回值包含股票当前的数据信息
            - get_rebound_stock_pool                    炸板股票池，返回值包含股票数据.返回Dict[Symbol,行情描述字符串]
            - get_new_stock_pool                        新股股票池，返回值包含股票数据.返回Dict[Symbol,行情描述字符串]
            - get_strong_stock_pool                     强势股票池，返回值包含股票数据.返回Dict[Symbol,行情描述字符串]
            - get_stock_comments_summary                千股千评数据.返回Dict[Symbol,行情描述字符串]。包括最新价、涨跌幅、换手率、市盈率、主力成本、机构参与度、综合得分、上升、目前排名、关注指数等数据
            - get_market_anomaly                        盘口异动，包括:'火箭发射', '快速反弹', '大笔买入', '封涨停板', '打开跌停板', '有大买盘', '竞价上涨', '高开5日线', '向上缺口', '60日新高', '60日大幅上涨', '加速下跌', '高台跳水', '大笔卖出', '封跌停板', '打开涨停板', '有大卖盘', '竞价下跌', '低开5日线', '向下缺口', '60日新低', '60日大幅下跌'
            - get_active_a_stock_stats                  活跃个股，查询周期：'近一月', '近三月', '近六月', '近一年'。，返回值包含股票数据
            - get_daily_lhb_details                     龙虎榜，返回值包含股票数据
            - get_institute_recommendations             机构推荐，包括：'最新投资评级', '上调评级股票', '下调评级股票', '股票综合评级', '首次评级股票', '目标涨幅排名', '机构关注度', '行业关注度', '投资评级选股'
            - get_investment_ratings                    最新投资评级
            - get_individual_stock_fund_flow_rank       资金流排名
            - select_by_stock_comments                  用自然语言查询千股千评论的数据，比如最受关注的10支股票，综合得分最高的10支股票
            - get_baidu_hotrank                         百度热门股票
            - get_baidu_recommendation                  百度推荐排名
            - get_baidu_sentiment_rank                  百度情绪指数排名
            - get_baidu_analysis_rank                   百度技术分析排名
            - get_top_holdings_by_market                持仓排名， "北向", "沪股通", "深股通"   "今日排行", "3日排行", "5日排行", "10日排行", "月排行", "季排行", "年排行" 
            - get_stock_report_fund_hold                获取机构持股报告数据。indicator="基金持仓" 返回dict[symbol,str]    
            - get_xueqiu_hot_follow                     获取雪球关注排行榜，参数num: int = 100，返回值Dict[symbol,str]
            - get_xueqiu_hot_tweet                      获取雪球讨论排行榜，参数num: int = 100，返回值Dict[symbol,str]
            - get_xueqiu_hot_deal                       获取雪球交易排行榜，参数num: int = 100，返回值Dict[symbol,str]
            - get_wencai_hot_rank                       获取问财热门股票排名，参数num: int = 100，返回值Dict[symbol,str]
            - get_eastmoney_hot_rank                    获取东方财富人气榜-A股，参数num: int = 100，返回值Dict[symbol,str]
        宏观经济
            - get_macro_economic_indicators             获取中国宏观经济数据的文字描述
            - get_global_economic_indicators            获取全球经济数据的文字描述
        用于获取市场整体信息的函数：
            - stock_market_desc                         市场平均市盈率等市场指标
            - get_current_buffett_index                 市场的巴菲特指数
            - get_sector_fund_flow_rank                 行业资金流向,返回值str
            - get_a_stock_pb_stats                      获取市场市净率统计
            - get_a_stock_pe_ratios                      获取市场市盈率统计
        用于财报日期的函数
            - get_latest_financial_report_date           上一个财报日
            - get_next_financial_report_date             下一个财报日
        用于交易日期的函数
            - get_latest_trading_date                    九点半之前，返回今天之前的交易日，九点半之后返回包括今天在内的最近交易日
            - get_previous_trading_date                  永远返回不包含今天的交易日
        用于获取个股信息的函数
            - get_main_business_description              主营业务，返回包含主营业务信息的字符串
            - get_stock_info                             个股信息，返回个股指标字符串
            - get_stock_a_indicators                     个股指标,返回包含指标信息的字符串
            - get_baidu_analysis_summary                  百度个股分析，返回包含分析信息的字符串
            - get_stock_news                              个股新闻,参数 symbol: str, num: int = 20 返回 List[str]
            - get_news_updates                            获取某个时间以后的个股新闻
            - get_vote_baidu                              获取百度股市通的股票投票数据。参数 symbol:str 返回str
            - get_stock_profit_forecast                   获取指定股票的盈利预测数据。symbol: str ,返回str 盈利预测字符串
            - get_esg_score                               获取指定股票的ESG评分数据。symbol: str ,返回str ESG评分字符串     
            - get_main_competitors                        获取主要竞争对手信息。symbol: str ,返回str 竞争对手信息字符串 
            - get_stock_big_deal                          获取大单数据,参数symbol: str ,返回str 大单数据字符串
            - get_recent_recommendations_summary          最近半年机构推荐汇总，参数symbol: str ,返回str 机构推荐汇总字符串
            - get_one_stock_news                         获取指定股票的最近新闻，参数symbol: str ,返回List[str] 新闻列表
            - get_realtime_stock_data                    获取指定股票的实时数据，参数symbol: str ,返回str 实时数据字符串
            - get_stock_announcements                    获取指定股票的公告，参数symbols: List[str] ,返回Dict[symbol,List[str]] 公告列表
            - calculate_stock_correlations               计算股票相关性，参数symbols: List[str] ,返回pd.DataFrame 相关性矩阵
            - get_stock_info_dict                        获取指定股票公司概况，参数symbol: str ,返回dict 公司概况
            - get_stock_report                           获取个股研报，参数symbol: str ,返回str 研报字符串
        用于代码查询的函数
            - search_index_code                         通过名称模糊查询指数代码
            - search_stock_code                         通过名称模糊查询股票代码
        用于查询财务数据
            - get_financial_analysis_summary            个股的财务分析指标
            - get_key_financial_indicators              关键财务指标
            - get_financial_forecast_summary            个股的财务预测指标.返回全部股票的财务预测dict
            - get_financial_report_summary              个股的财务报告摘要.一个字典，键是股票代码，值是描述性的字符串
        用于查询财务数据细节(非必要勿使用)
            - get_balance_sheet_summary                 资产负债表摘要
            - get_profit_statement_summary              利润表摘要
            - get_cash_flow_statement_summary           现金流量表摘要
            - get_stock_balance_sheet_by_report_em      资产负债表完整数据
        用于查询行情摘要
            - summarize_historical_data                 股票历史数据摘要
            - summarize_historical_index_data           指数历史信息摘要
        用于查询行情细节
            - get_historical_daily_data                 行情历史数据，参数symbol: str, start_date: str, end_date : str 返回DataFrame    
            - get_index_data                            指数行情数据，参数symbol: str, start_date: str, end_date: str  返回DataFrame
        查询新闻数据
            - get_baidu_market_news                     百度市场新闻，参数num=40,返回List[str]
            - get_baidu_stock_news                      百度个股新闻,参数 symbol: str, num: int = 20 返回 List[str]
            - get_market_news_300                       财联社新闻300条,返回List[str]
            - get_cctv_news                             新闻联播文字稿，参数天数，返回包含新闻数据的字典列表，每个字典包含date（日期）、title（标题）和content（内容）
            - get_baidu_important_news                  百度重要新闻，num=200,返回List[str]
        用于新闻数据处理
            - summarizer_news                           把新闻数据根据 query 的要求 总结出短摘要
        用于解析llm_client的json输出
            - extract_json_from_text                    从text中提取json,返回dict或者list
        行业数据
            - get_industry_pe_ratio                      获取行业市盈率，参数，symbol 行业，Dict[行业名称,str]
            - get_stock_sector                           获取股票所属行业，参数symbol，返回str
            - calculate_industry_correlations            计算行业相关性，参数names: List[str], days: int = 120 返回pd.DataFrame
            - get_stock_concept_fund_flow_top            获取概念板块资金流数据，参数num: int = 25，返回list
        大盘数据
            - get_latest_market_fund_flow                获取最新市场资金流数据，返回dict
            - get_stock_main_fund_flow                   获取主力资金流数据，参数top:int=20，返回str
        """
        return prompt
    
    