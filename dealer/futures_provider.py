
from datetime import datetime, timedelta
import hashlib
import json
import os
import pickle
import random
import re
import time
from typing import Any, Dict, List, Literal, Optional, Union
import akshare as ak
import pandas as pd
import requests
from core.llms._llm_api_client import LLMApiClient
from core.utils.single_ton import Singleton
from dealer.lazy import lazy
import logging
from bs4 import BeautifulSoup

rq = None
from core.utils.config_setting import Config
config = Config()
rq_user = config.get('rq_user')
rq_pwd = config.get('rq_pwd')
if rq_user and rq_pwd:
    rq = lazy("rqdatac")
    if rq:
        rq.init(rq_user, rq_pwd)

from core.tushare_doc.ts_code_matcher import StringMatcher
from tenacity import retry,stop_after_attempt,wait_fixed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

exchange_suffix_dict = {
    "中国金融期货交易所": ".CFX",
    "上海期货交易所": ".SHF",
    "上海国际能源交易中心": ".INE",
    "郑州商品交易所": ".ZCE",
    "大连商品交易所": ".DCE",
    "广州期货交易所": ".GFEX"
}
# 交易所数据转换为字典
futures_dict = {
    "IF": "沪深300股指期货",
    "TF": "5年期国债期货",
    "T": "10年期国债期货",
    "IC": "中证500股指期货",
    "IH": "上证50股指期货",
    "TS": "2年期国债期货",
    "IM": "中证1000股指期货",
    "TL": "30年期国债期货",
    "SC": "中质含硫原油",
    "NR": "20号胶",
    "LU": "低硫燃料油",
    "BC": "国际铜",
    "EC": "集运指数(欧线)",
    "RU": "橡胶",
    "CU": "沪铜",
    "AL": "沪铝",
    "FU": "燃油",
    "ZN": "沪锌",
    "AU": "沪金",
    "RB": "螺纹钢",
    "WR": "线材",
    "PB": "沪铅",
    "AG": "沪银",
    "BU": "沥青",
    "HC": "热卷",
    "SN": "锡",
    "NI": "镍",
    "SP": "纸浆",
    "SS": "不锈钢",
    "AO": "氧化铝",
    "BR": "丁二烯橡胶",
    "CF": "郑棉",
    "SR": "白糖",
    "TA": "PTA",
    "PM": "普麦",
    "FG": "玻璃",
    "RM": "菜粕",
    "RS": "菜籽",
    "OI": "菜油",
    "RI": "早籼",
    "WH": "强麦",
    "ZC": "郑煤",
    "JR": "粳稻",
    "LR": "晚籼",
    "SM": "锰硅",
    "SF": "硅铁",
    "MA": "甲醇",
    "CY": "棉纱",
    "AP": "苹果",
    "CJ": "红枣",
    "UR": "尿素",
    "SA": "纯碱",
    "PF": "短纤",
    "PK": "花生",
    "SH": "烧碱",
    "PX": "对二甲苯",
    "A": "豆一",
    "M": "豆粕",
    "C": "玉米",
    "B": "豆二",
    "Y": "豆油",
    "L": "塑料",
    "P": "棕榈",
    "V": "PVC",
    "J": "焦炭",
    "JM": "焦煤",
    "I": "铁矿石",
    "JD": "鸡蛋",
    "FB": "纤维板",
    "BB": "胶合板",
    "PP": "聚丙烯",
    "CS": "玉米淀粉",
    "EG": "乙二醇",
    "RR": "粳米",
    "EB": "苯乙烯",
    "PG": "液化石油气",
    "LH": "生猪",
    "SI": "工业硅",
    "LC": "碳酸锂"
}

# 创建一个字典，键为证券代码，值为交易所简称
exchange_dict = {
    "IF": "中国金融期货交易所",
    "TF": "中国金融期货交易所",
    "T": "中国金融期货交易所",
    "IC": "中国金融期货交易所",
    "IH": "中国金融期货交易所",
    "TS": "中国金融期货交易所",
    "IM": "中国金融期货交易所",
    "TL": "中国金融期货交易所",
    "SC": "上海国际能源交易中心",
    "NR": "上海国际能源交易中心",
    "LU": "上海国际能源交易中心",
    "BC": "上海国际能源交易中心",
    "EC": "上海国际能源交易中心",
    "RU": "上海期货交易所",
    "CU": "上海期货交易所",
    "AL": "上海期货交易所",
    "FU": "上海期货交易所",
    "ZN": "上海期货交易所",
    "AU": "上海期货交易所",
    "RB": "上海期货交易所",
    "WR": "上海期货交易所",
    "PB": "上海期货交易所",
    "AG": "上海期货交易所",
    "BU": "上海期货交易所",
    "HC": "上海期货交易所",
    "SN": "上海期货交易所",
    "NI": "上海期货交易所",
    "SP": "上海期货交易所",
    "SS": "上海期货交易所",
    "AO": "上海期货交易所",
    "BR": "上海期货交易所",
    "CF": "郑州商品交易所",
    "SR": "郑州商品交易所",
    "TA": "郑州商品交易所",
    "PM": "郑州商品交易所",
    "FG": "郑州商品交易所",
    "RM": "郑州商品交易所",
    "RS": "郑州商品交易所",
    "OI": "郑州商品交易所",
    "RI": "郑州商品交易所",
    "WH": "郑州商品交易所",
    "ZC": "郑州商品交易所",
    "JR": "郑州商品交易所",
    "LR": "郑州商品交易所",
    "SM": "郑州商品交易所",
    "SF": "郑州商品交易所",
    "MA": "郑州商品交易所",
    "CY": "郑州商品交易所",
    "AP": "郑州商品交易所",
    "CJ": "郑州商品交易所",
    "UR": "郑州商品交易所",
    "SA": "郑州商品交易所",
    "PF": "郑州商品交易所",
    "PK": "郑州商品交易所",
    "SH": "郑州商品交易所",
    "PX": "郑州商品交易所",
    "A": "大连商品交易所",
    "M": "大连商品交易所",
    "C": "大连商品交易所",
    "B": "大连商品交易所",
    "Y": "大连商品交易所",
    "L": "大连商品交易所",
    "P": "大连商品交易所",
    "V": "大连商品交易所",
    "J": "大连商品交易所",
    "JM": "大连商品交易所",
    "I": "大连商品交易所",
    "JD": "大连商品交易所",
    "FB": "大连商品交易所",
    "BB": "大连商品交易所",
    "PP": "大连商品交易所",
    "CS": "大连商品交易所",
    "EG": "大连商品交易所",
    "RR": "大连商品交易所",
    "EB": "大连商品交易所",
    "PG": "大连商品交易所",
    "LH": "大连商品交易所",
    "SI": "广州期货交易所",
    "LC": "广州期货交易所"
}


class MainContractGetter(StringMatcher, metaclass=Singleton):
    def __init__(self):
        df_path =  './json/main_contract_cache.pickle'
        index_cache = './json/main_contract_index_cache.pickle'
        df = pd.read_pickle(df_path)
        super().__init__(df, index_cache=index_cache, index_column='content', result_column='symbol')
    def __getitem__(self, query):
        return self.rapidfuzz_match(query)


class MainContractProvider:
    def __init__(self,llm_client:LLMApiClient = None) -> None:
        
        if llm_client is None:
            from core.llms.mini_max_pro import MiniMaxProClient
            self.llm_client = MiniMaxProClient()    
        else:
            self.llm_client = llm_client
        self.code_getter = MainContractGetter()
        self._spot_price_cache = {}
        self.ff_contracts = ["IF", "TF", "T", "IC", "IH", "TS", "IM", "TL"]
        self.futures_comm_info_df = None
        self.futures_fees_info_df = None
        self.futures_comm_info_cache_file = "./json/futures_comm_info.pickle"
    
    def get_bar_data(self, name: str, period: Literal['1', '5', '15', '30', '60', 'D'] = '1', date: Optional[str] = None):
        """
        获取期货合约的bar数据
        
        :param name: 合约名称
        :param period: 时间周期，默认为'1'（1分钟）
        :param date: 回测日期，格式为'YYYY-MM-DD'，如果不提供则使用当前日期
        :return: DataFrame包含bar数据
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        end_date = datetime.strptime(date, '%Y-%m-%d')
        if period == 'D':
            start_date = end_date - timedelta(days=365)  # 获取一年的日线数据
        else:
            start_date = end_date - timedelta(days=10)  # 获取5天的分钟数据
        
        frequency_map = {'1': '1m', '5': '5m', '15': '15m', '30': '30m', '60': '60m', 'D': '1d'}
        frequency = frequency_map[period]
        name = re.sub(r'[^a-zA-Z]', '', name)
        code = self.code_getter[name]
        if code.endswith('0'):
            code = code[:-1]
        
        df = self.get_rqbar(code, start_date, end_date, frequency)
        
        if period != 'D':
            df = df.reset_index()
            df = df.rename(columns={'datetime': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'open_interest': 'open_interest'})
        else:
            df = df.reset_index()
            #df['date'] = df['datetime'].dt.date
            df = df.rename(columns={'date': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'open_interest': 'open_interest'})
            #df = df.drop(columns=['datetime'])
        
        return df

    def get_main_contract(self):
        df = ak.futures_display_main_sina()
        df["content"] = df['symbol'] +'.'+df['exchange']+','+ df['name']
        return df   

    def make_main_chache(self):
        df = self.get_main_contract()
        df.to_pickle('./json/main_contract_cache.pickle')
        from core.tushare_doc.ts_code_matcher import StringMatcher
        matcher = StringMatcher(df, index_cache='./json/main_contract_index_cache.pickle', index_column='content', result_column='symbol')
    
    def get_shment_news(self, symbol: str = '全部'):
        return ak.futures_news_shmet(symbol=symbol)

    def generate_acs_token(self):
        current_time = int(time.time() * 1000)
        random_num = random.randint(1000000000000000, 9999999999999999)  # 16位随机数
        
        part1 = str(current_time)
        part2 = str(random_num)
        part3 = "1"
        
        token = f"{part1}_{part2}_{part3}"
        
        md5 = hashlib.md5()
        md5.update(token.encode('utf-8'))
        hash_value = md5.hexdigest()
        
        # 添加额外的随机字符串来增加长度
        extra_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=20))
        
        final_token = f"{token}_{hash_value}_{extra_chars}"
        
        return final_token   
    
    def get_futures_news(self, code: str = 'SC0', page_num: int = 0, page_size: int = 20) -> Optional[pd.DataFrame]:
        code = self.extract_contract_prefix(code)
        code = f"{code[:-1]}000" if code.endswith('0') else f"{code}000"
        url = 'https://finance.pae.baidu.com/vapi/getfuturesnews'

        headers_file_path = './json/baidu_headers.json'
        if os.path.exists(headers_file_path):
            with open(headers_file_path, 'r') as f:
                headers_data = json.load(f)
                headers = headers_data 
                cookies =  {
                    'cookie': headers_data['cookie']
                }
                del headers['cookie'] # remove cookie from headers 
        else:
            headers = {
                'accept': 'application/vnd.finance-web.v1+json',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'acs-token': self.generate_acs_token(),
                'origin': 'https://gushitong.baidu.com',
                'referer': 'https://gushitong.baidu.com/',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0',
                'sec-ch-ua': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-site',
            }

            cookies ={
                'BAIDUID': '564AD52829EF1290DDC1A20DCC14F220:FG=1',
                'BAIDUID_BFESS': '564AD52829EF1290DDC1A20DCC14F220:FG=1',
                'BIDUPSID': '564AD52829EF1290DDC1A20DCC14F220',
                'PSTM': '1714397940',
                'ZFY': '3ffAdSTQ3amiXQ393UWe0Uy1s70:BPIai4AGEBTM6yIQ:C',
                'H_PS_PSSID': '60275_60287_60297_60325',
                'BDUSS': 'X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma',
                'BDUSS_BFESS': 'X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma',
                'ab_sr': '1.0.1_MTFlOTZiMTRlYjEyYTliZGU4YWFkMWIzMzkxYjdlOTJjNWY1NDM1MzZkZDQ5NTlhOGQwZDE3NWJkZjJmY2NmY2RkYWVkZTcyYTVhNDRmNjg4OGEzYjMzZGUzYTczMzhhNWZhNjRiOWE2YTJjNWZmNzNhNTEwMWQwODYwZDZkNmUzMjg3Yjc0NGM5Y2M0MjViNDY5NzU4MWQzZDZjMzViMw=='
            }


        params = {
            'code': code,
            'pn': page_num,
            'rn': page_size,
            'finClientType': 'pc'
        }

        try:
            response = requests.get(url, headers=headers, params=params, cookies=cookies)
            response.raise_for_status() 

            data = response.json()

            if 'Result' in data and isinstance(data['Result'], list):
                df = pd.DataFrame(data['Result'])
                return df
            else:
                print("Unexpected data structure in the response")
                return None

        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def get_article_content(self, url: str) -> str:
        try:
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()
            
            # 尝试检测编码
            if response.encoding.lower() != 'utf-8':
                response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')

            # 尝试多种策略提取文章内容
            content = (
                self._extract_by_class(soup, 'article') or
                self._extract_by_class(soup, 'content') or
                self._extract_by_tag(soup, 'article') or
                self._extract_by_longest_text(soup)
            )

            if content:
                # 清理文本
                content = self._clean_text(content)
                return content
            else:
                return "未能提取到文章内容。"

        except requests.RequestException as e:
            return f"获取文章内容时发生错误: {str(e)}"

    def _extract_by_class(self, soup, class_name):
        element = soup.find(class_=class_name)
        return element.get_text() if element else None

    def _extract_by_tag(self, soup, tag_name):
        element = soup.find(tag_name)
        return element.get_text() if element else None

    def _extract_by_longest_text(self, soup):
        # 移除脚本和样式元素
        for script in soup(["script", "style"]):
            script.decompose()

        # 找到文本最长的段落
        paragraphs = soup.find_all('p')
        if paragraphs:
            longest_p = max(paragraphs, key=lambda p: len(p.get_text()))
            return longest_p.get_text()
        return None

    def _clean_text(self, text):
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除特殊字符和HTML实体
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        # 移除可能的广告文本（根据需要调整）
        text = re.sub(r'广告|推广|赞助', '', text)
        return text

    def get_akbar(self, symbol: str, frequency: str = '1m') -> Optional[pd.DataFrame]:
        """
        获取期货行情数据，首选 AKShare，如果失败则尝试备用数据源

        :param symbol: 期货合约代码
        :param frequency: 数据频率，可选 '1m', '5m', '15m', '30m', '60m', 'D'
        :return: 包含行情数据的 DataFrame，如果无法获取数据则返回 None
        """
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        
        def fetch_akshare_data():
            try:
                # Ensure the symbol ends with "0"
                if not symbol[-1].isdigit():
                    adjusted_symbol = f"{symbol}0"
                else:
                    adjusted_symbol = symbol

                if frequency == 'D':
                    df = ak.futures_zh_daily_sina(symbol=adjusted_symbol)
                    df['datetime'] = pd.to_datetime(df['date'])
                    df = df.set_index('datetime')
                else:
                    period_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '60m': '60'}
                    period = period_map.get(frequency, '1')
                    df = ak.futures_zh_minute_sina(symbol=adjusted_symbol, period=period)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime')

                df = df.rename(columns={'hold': 'open_interest'})
                return df[columns_to_keep]
            except Exception as e:
                logging.error(f"Error fetching data from AKShare for symbol {adjusted_symbol}, frequency {frequency}: {str(e)}")
                return None

        def fetch_alternative_data():
            # Implement an alternative data fetching method here
            # For example, you could use another API or a local data source
            logging.warning(f"Attempting to fetch data from alternative source for symbol {symbol}, frequency {frequency}")
            return None  # Replace with actual alternative data fetching logic

        # Try to fetch data from AKShare
        df = fetch_akshare_data()
        
        # If AKShare fails, try the alternative source
        if df is None:
            df = fetch_alternative_data()
        
        # If both sources fail, return None
        if df is None:
            logging.error(f"Failed to fetch data for symbol {symbol}, frequency {frequency} from all sources")
            return None

        # Ensure all necessary columns exist
        for col in columns_to_keep:
            if col not in df.columns:
                df[col] = None

        return df

    def get_rqbar(self, symbol: str, start_date: str, end_date: str, frequency: str = '1m', adjust_type: str = 'none'):
        if len(symbol) <= 4 and symbol.endswith('0'):
            symbol = symbol[:-1]
        
        return rq.futures.get_dominant_price(symbol, start_date, end_date, frequency, adjust_type=adjust_type)
    
    def get_trade_calendar(self, start,end) -> List[datetime.date]:
        return rq.get_trading_dates(start,end)
    
    def get_main_contract_from_code(self,code:str)->str:
        code = code[:-1] if code.endswith('0') else code
        codelist:pd.Series = rq.get_dominant(code)
        return codelist.iloc[0]

    def get_trading_product(self,security_code: str) -> str:
        # 转换为大写，并返回对应的交易品种
        return futures_dict.get(security_code.upper(), "未找到对应的交易品种")

    def get_exchange_name(self,security_code: str) -> str:
        # 转换输入的证券代码为大写，并查找对应的交易所简称
        return exchange_dict.get(security_code.upper(), "未找到对应的交易所")

    def get_futures_inventory_string(self, contract: str) -> str:
        # 提取合约前缀
        contract_prefix = self.extract_contract_prefix(contract)
        
        # 获取交易品种
        product = self.get_trading_product(contract_prefix)
        if product == "未找到对应的交易品种":
            return "未找到对应的交易品种"

        # 获取交易所
        exchange = self.get_exchange_name(contract_prefix)
        if exchange == "未找到对应的交易所":
            return "未找到对应的交易所"

        # 调用 Akshare 接口获取库存数据
        try:
            inventory_data = ak.futures_inventory_99(exchange=exchange, symbol=product)
        except Exception as e:
            return f"获取数据失败: {str(e)}"

        # 获取最近15天的数据
        recent_data = inventory_data.tail(15)

        # 将数据转换为字符串
        result_str = ""
        for index, row in recent_data.iterrows():
            result_str += f"日期: {row['日期']}, 库存: {row['库存']}, 增减: {row['增减']}\n"

        return result_str

    def get_futures_inventory_em_string(self, contract: str, days: int = 15) -> str:
        # 提取合约前缀
        contract_prefix = self.extract_contract_prefix(contract)
        
        # 获取交易品种
        product = self.get_trading_product(contract_prefix)
        if product == "未找到对应的交易品种":
            return "未找到对应的交易品种"

        # 调用 Akshare 接口获取库存数据
        try:
            inventory_data = ak.futures_inventory_em(symbol=product)
        except Exception as e:
            return f"获取数据失败: {str(e)}"

        # 获取最近指定天数的数据
        recent_data = inventory_data.tail(days)

        # 将数据转换为字符串
        result_str = ""
        for index, row in recent_data.iterrows():
            result_str += f"日期: {row['日期']}, 库存: {row['库存']}, 增减: {row['增减']}\n"

        return result_str

    def extract_contract_prefix(self, contract: str) -> str:
        match = re.match(r"([a-zA-Z]+)", contract)
        if match:
            return match.group(1).upper()
        else:
            return ""

    def get_futures_spot_price_with_cache(self, contract: str) -> str:
        # 提取合约前缀并转换为符号
        symbol = self.extract_contract_prefix(contract)
        
        # 如果缓存不为空，直接在缓存中查找
        if self._spot_price_cache:
            if symbol in self._spot_price_cache:
                return self._spot_price_cache[symbol]
            else:
                return "未找到对应的符号数据"
        
        # 获取今天的日期
        today = datetime.today()
        
        while True:
            date_str = today.strftime('%Y%m%d')
            try:
                # 调用 AKShare 接口获取基差数据
                futures_spot_price_df = ak.futures_spot_price(date_str)
                
                # 如果获取到数据，进行处理并缓存
                if not futures_spot_price_df.empty:
                    # 将数据转换为字典，key=symbol, value=行内容转成易阅读的字符串
                    for _, row in futures_spot_price_df.iterrows():
                        formatted_str = (
                            f"品种: {row['symbol']}, 现货价格: {row['spot_price']}, "
                            f"最近交割合约: {row['near_contract']}, 最近交割合约价格: {row['near_contract_price']}, "
                            f"主力合约: {row['dom_contract']}, 主力合约价格: {row['dom_contract_price']}, "
                            f"最近合约基差值: {row['near_basis']}, 主力合约基差值: {row['dom_basis']}, "
                            f"最近合约基差率: {row['near_basis_rate']}, 主力合约基差率: {row['dom_basis_rate']}, 日期: {row['date']}"
                        )
                        self._spot_price_cache[row['symbol']] = formatted_str

                    # 返回查询结果
                    return self._spot_price_cache.get(symbol, "未找到对应的符号数据")

            except Exception as e:
                # 如果未获取到数据，则日期减1天继续尝试
                today -= datetime.timedelta(days=1)
                continue

    def get_futures_hold_pos_sina(self, contract: str) -> str:
        # 获取今天的日期
        today = datetime.today().strftime('%Y%m%d')

        # 定义需要查询的类型
        types = ["成交量", "多单持仓", "空单持仓"]

        # 初始化结果存储
        statistics_summary = []
        foreign_firms = []

        # 定义外资期货公司关键词
        foreign_keywords = ["摩根", "瑞银", "乾坤"]

        for symbol in types:
            try:
                # 调用 AKShare 接口获取成交持仓数据
                futures_hold_pos_sina_df = ak.futures_hold_pos_sina(symbol=symbol, contract=contract, date=today)

                # 统计数据处理
                total_volume = futures_hold_pos_sina_df[symbol].sum()
                total_change = futures_hold_pos_sina_df["比上交易增减"].sum()
                statistics_summary.append(f"{symbol} 总持仓: {total_volume}, 总增减: {total_change}")

                # 检查并处理外资期货公司数据
                for _, row in futures_hold_pos_sina_df.iterrows():
                    for keyword in foreign_keywords:
                        if row['会员简称'].startswith(keyword):
                            foreign_firms.append(
                                f"{symbol} - 名次: {row['名次']}, 会员简称: {row['会员简称']}, "
                                f"{symbol}: {row[symbol]}, 比上交易增减: {row['比上交易增减']}"
                            )
                            break

            except Exception as e:
                statistics_summary.append(f"{symbol} - 获取数据失败: {str(e)}")

        # 汇总结果
        summary = "\n".join(statistics_summary)
        foreign_summary = "\n".join(foreign_firms)

        # 最终返回结果
        if foreign_summary:
            return f"合约: {contract}\n\n统计数据:\n{summary}\n\n外资期货公司详细数据:\n{foreign_summary}"
        else:
            return f"合约: {contract}\n\n统计数据:\n{summary}\n\n"

    def get_futures_zh_spot(self, contract: str) -> str:
        try:
            # 提取合约前缀并判断市场类型
            contract_prefix = self.extract_contract_prefix(contract)
            market = "FF" if contract_prefix in self.ff_contracts else "CF"
            
            # 调用 AKShare 接口获取实时行情数据
            futures_zh_spot_df = ak.futures_zh_spot(symbol=contract, market=market, adjust='0')
            
            if futures_zh_spot_df.empty:
                return f"未找到合约 {contract} 的数据。"

            # 将数据转换为易于阅读的字符串格式
            result = []
            for _, row in futures_zh_spot_df.iterrows():
                formatted_str = (
                    f"合约: {row['symbol']}, 时间: {row['time']}, 开盘价: {row['open']}, 最高价: {row['high']}, "
                    f"最低价: {row['low']}, 当前价格: {row['current_price']}, 买价: {row['bid_price']}, 卖价: {row['ask_price']}, "
                    f"买量: {row['buy_vol']}, 卖量: {row['sell_vol']}, 持仓量: {row['hold']}, 成交量: {row['volume']}, "
                    f"均价: {row['avg_price']}, 上个交易日收盘价: {row['last_close']}, 上个交易日结算价: {row['last_settle_price']}"
                )
                result.append(formatted_str)

            return "\n".join(result)
        
        except Exception as e:
            return f"查询合约 {contract} 时出错: {str(e)}"

    def get_futures_realtime_data(self, contract: str) -> pd.DataFrame:
        try:
            # 提取合约前缀并获取交易品种名称
            contract_prefix = self.extract_contract_prefix(contract)
            symbol = self.get_trading_product(contract_prefix)

            if symbol == "未找到对应的交易品种":
                print(f"未找到合约前缀 {contract_prefix} 对应的交易品种。")
                return pd.DataFrame()

            # 调用 AKShare 接口获取实时行情数据
            futures_zh_realtime_df = ak.futures_zh_realtime(symbol=symbol)
            return futures_zh_realtime_df
        
        except Exception as e:
            print(f"查询合约 {contract} 时出错: {str(e)}")
            return pd.DataFrame()  # 返回空的 DataFrame 以防错误处理

    def get_futures_fees_info(self,contract:str)->Optional[dict]:
        """
        获取合约手续费信息
        名称	类型	描述
        交易所	object	-
        合约代码	object	-
        合约名称	object	-
        品种代码	object	-
        品种名称	object	-
        合约乘数	int64	-
        最小跳动	float64	-
        开仓费率（按金额）	float64	-
        开仓费用（按手）	float64	-
        平仓费率（按金额）	float64	-
        平仓费用（按手）	float64	-
        平今费率（按金额）	float64	-
        平今费用（按手）	float64	-
        做多保证金率（按金额）	float64	-
        做多保证金（按手）	int64	-
        做空保证金率（按金额）	float64	-
        做空保证金（按手）	int64	-
        上日结算价	float64	-
        上日收盘价	float64	-
        最新价	float64	-
        成交量	int64	-
        持仓量	int64	-
        1手开仓费用	float64	-
        1手平仓费用	float64	-
        1手平今费用	float64	-
        做多1手保证金	float64	-
        做空1手保证金	float64	-
        1Tick平仓盈亏	float64	-
        2Tick平仓盈亏	float64	-
        1Tick平仓收益率	object	-
        2Tick平仓收益率	object	-
        1Tick平今盈亏	float64	-
        2Tick平今盈亏	float64	-
        1Tick平今收益率	object	-
        2Tick平今收益率	object	-
        更新时间	object	-
        """
        if self.futures_fees_info_df is None:
            self.futures_fees_info_df = ak.futures_fees_info()
        selected_row = self.futures_fees_info_df.loc[self.futures_fees_info_df['合约代码'] == contract]
        if selected_row.empty:
            return None
        return selected_row.iloc[0].to_dict()

    def get_futures_comm_info(self, contract: str) -> Optional[dict]:
        """
        获取合约基本信息，优先从API获取，如果失败则尝试从本地缓存获取
        """
        try:
            # 尝试从API获取数据
            self.futures_comm_info_df = ak.futures_comm_info(symbol="所有")
            # 如果成功获取数据，更新本地缓存
            self._update_futures_comm_info_cache()
        except Exception as e:
            # 如果API获取失败，尝试从本地缓存读取
            self.futures_comm_info_df = self._read_futures_comm_info_from_cache()

        if self.futures_comm_info_df is None:
            return None

        selected_row = self.futures_comm_info_df.loc[self.futures_comm_info_df['合约代码'] == contract]
        if selected_row.empty:
            return None
        return selected_row.iloc[0].to_dict()

    def _update_futures_comm_info_cache(self):
        """更新期货合约信息的本地缓存文件"""
        os.makedirs(os.path.dirname(self.futures_comm_info_cache_file), exist_ok=True)
        with open(self.futures_comm_info_cache_file, 'wb') as f:
            pickle.dump(self.futures_comm_info_df, f)

    def _read_futures_comm_info_from_cache(self) -> Optional[pd.DataFrame]:
        """从本地缓存文件读取期货合约信息数据"""
        if not os.path.exists(self.futures_comm_info_cache_file):

            return None
        try:
            with open(self.futures_comm_info_cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:

            return None
        
    def get_futures_minute_data(self, contract: str, period: str = "1") -> pd.DataFrame:
        try:
            # 提取合约前缀并获取交易品种名称
            contract_prefix = self.extract_contract_prefix(contract)
            symbol = self.get_trading_product(contract_prefix)

            if symbol == "未找到对应的交易品种":
                print(f"未找到合约前缀 {contract_prefix} 对应的交易品种。")
                return pd.DataFrame()

            # 调用 AKShare 接口获取分时行情数据
            futures_minute_df = ak.futures_zh_minute_sina(symbol=contract, period=period)
            return futures_minute_df
        
        except Exception as e:
            print(f"查询合约 {contract} 时出错: {str(e)}")
            return pd.DataFrame()  # 返回空的 DataFrame 以防错误处理

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def extract_json_from_text(self, text: str, max_attempts: int = 3) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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

    def get_self_description(self) -> str:
        description = """
        MainContractProvider 类函数功能简介：

        1. get_main_contract() -> pd.DataFrame
           功能：获取主要期货合约的列表
           输入：无
           输出：包含主要期货合约信息的DataFrame

        2. get_futures_news(code: str = 'SC0', page_num: int = 0, page_size: int = 20) -> Optional[pd.DataFrame]
           功能：获取指定期货合约的新闻数据
           输入：期货合约代码、页码、每页显示数量
           输出：包含新闻数据的DataFrame或None

        3. get_akbar(symbol: str, frequency: str = '1m') -> Optional[pd.DataFrame]
           功能：获取期货行情数据，首选 AKShare，如果失败则尝试备用数据源
           输入：期货合约代码、数据频率
           输出：包含行情数据的DataFrame或None

        4. extract_contract_prefix(contract: str) -> str
            功能：提取期货合约前缀
            输入：期货合约
            输出：合约前缀字符串

        5. get_trading_product(security_code: str) -> str
            功能：根据合约代码获取交易品种
            输入：合约代码
            输出：交易品种字符串

        6. get_exchange_name(security_code: str) -> str
            功能：根据合约代码获取交易所名称
            输入：合约代码
            输出：交易所名称字符串

        7. get_futures_inventory_string(contract: str) -> str
            功能：获取指定合约的期货库存数据，返回最近15天的库存数据
            输入：期货合约
            输出：库存数据的字符串

        8. get_futures_inventory_em_string(contract: str, days: int = 15) -> str
            功能：获取指定合约的期货库存数据，返回最近指定天数的库存数据
            输入：期货合约、天数
            输出：库存数据的字符串

        9. get_futures_spot_price_with_cache(contract: str) -> str
            功能：获取指定合约的现货价格和基差数据，并缓存查询结果
            输入：期货合约
            输出：现货价格和基差数据的字符串

        10. get_futures_hold_pos_sina(contract: str) -> str
            功能：获取指定合约的成交持仓数据，并返回统计数据和外资期货公司详细数据
            输入：期货合约
            输出：统计数据和外资期货公司详细数据的字符串

        11. get_futures_zh_spot(contract: str) -> str
            功能：获取指定合约的实时行情数据，并返回格式化后的字符串
            输入：期货合约
            输出：实时行情数据的字符串

        12. get_futures_realtime_data(contract: str) -> pd.DataFrame
            功能：获取指定合约的实时行情数据，并返回DataFrame
            输入：期货合约
            输出：实时行情数据的DataFrame

        13. get_futures_minute_data(contract: str, period: str = '1') -> pd.DataFrame
            功能：获取指定合约的分时行情数据，并返回DataFrame
            输入：期货合约、时间周期（默认为1分钟）
            输出：分时行情数据的DataFrame
        """
        return description.strip()

def curl_to_python_code(curl_command: str) -> str:
    # Extract URL
    url_match = re.search(r"curl '([^']+)'", curl_command)
    url = url_match.group(1) if url_match else ''

    # Extract headers
    headers = {}
    cookies = {}
    header_matches = re.findall(r"-H '([^:]+): ([^']+)'", curl_command)
    for key, value in header_matches:
        if key.lower() == 'cookie':
            cookies = {k.strip(): v.strip() for k, v in [cookie.split('=', 1) for cookie in value.split(';')]}
        else:
            headers[key] = value

    # Generate Python code
    code = f"""import requests
import pandas as pd
from typing import Optional

def get_futures_news(code: str = 'SC0', page_num: int = 0, page_size: int = 20) -> Optional[pd.DataFrame]:
    code = self.extract_contract_prefix(code)
    code = f"{{code[:-1]}}888" if code.endswith('0') else f"{{code}}888"
    url = 'https://finance.pae.baidu.com/vapi/getfuturesnews'
    
    headers = {headers}
    
    cookies = {cookies}
    
    params = {{
        'code': code,
        'pn': page_num,
        'rn': page_size,
        'finClientType': 'pc'
    }}
    
    try:
        response = requests.get(url, headers=headers, params=params, cookies=cookies)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Result' in data and isinstance(data['Result'], list):
            df = pd.DataFrame(data['Result'])
            return df
        else:
            print("Unexpected data structure in the response")
            return None
    
    except requests.RequestException as e:
        print(f"An error occurred: {{e}}")
        return None

# Usage example:
# df = get_futures_news('SC0')
# if df is not None:
#     print(df.head())
"""
    return code
