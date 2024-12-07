from pathlib import Path
import pandas as pd
from collections import defaultdict
import jieba
import re
from fuzzywuzzy import fuzz
import rapidfuzz.fuzz as rfuzz
import pickle
import os
import core.stock.tushare_provider as tp
from core.tushare_doc.ts_code_matcher import StringMatcher
from core.utils.single_ton import singleton

@singleton
class StockMatcher(StringMatcher):
    def __init__(self, index_cache='./output/stock_index_cache.pkl'):
        Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
        # 获取股票基础信息
        df = tp.fetch_stock_basic_info(list_status="L")
        # 添加空的enname列，以防将来需要
        df['enname'] = ''
        # 创建搜索文本列，合并所有要搜索的字段
        df['search_content'] = df.apply(lambda x: f"{x['symbol']} {x['name']} {x['cnspell']}", axis=1)
        
        super().__init__(df, index_cache, index_column='search_content', result_column='ts_code')
        self.df = df

    def __contains__(self, query):
        """
        重载in操作符，判断查询字符串是否存在于symbol或ts_code中
        
        参数:
        - query: 查询文本，通常是股票代码
        
        返回:
        - bool: 如果查询字符串存在于symbol或ts_code中返回True，否则返回False
        
        用法:
        "000001" in matcher -> True
        "600001.SH" in matcher -> True
        "不存在的代码" in matcher -> False
        """
        # 将查询字符串转换为小写以进行不区分大小写的比较
        query = str(query).lower()
        
        # 检查是否存在于symbol中
        symbols = self.df['symbol'].str.lower()
        if any(symbols == query):
            return True
            
        # 检查是否存在于ts_code中
        ts_codes = self.df['ts_code'].str.lower()
        if any(ts_codes == query):
            return True
            
        return False
    def __getitem__(self, query):
        """
        重载[]操作符，通过中括号访问直接返回股票代码
        
        参数:
        - query: 查询文本，可以是股票代码、名称、拼音等
        
        返回:
        - 匹配的股票代码(symbol)，如果没找到返回None
        
        用法:
        matcher["平安银行"] -> "000001"
        matcher["payh"] -> "000001"
        """
        return self.get_symbol(query)

    def get_symbol(self, query, threshold=80):
        """
        根据查询文本返回匹配的股票代码(symbol)
        
        参数:
        - query: 查询文本，可以是股票代码、名称、拼音等
        - threshold: 模糊匹配阈值，默认80
        
        返回:
        - 匹配的股票代码，如果没找到返回None
        """
        methods = [
            ('exact', 70),
            ('rapid', threshold),
            ('fuzzy', threshold-5),
            ('inverted', threshold-10)
        ]
        
        for method, thresh in methods:
            ts_code = None
            if method == 'exact':
                ts_code = self.exact_match(query)
            elif method == 'rapid':
                ts_code = self.rapidfuzz_match(query, thresh)
            elif method == 'fuzzy':
                ts_code = self.fuzzywuzzy_match(query, thresh)
            elif method == 'inverted':
                ts_code = self.inverted_index_match(query)
            
            if ts_code:
                match_row = self.df[self.df['ts_code'] == ts_code].iloc[0]
                return match_row['symbol']
        
        return None
    def get_tscode(self, query, threshold=80):
        """
        根据查询文本返回匹配的股票代码(ts_code)
        
        参数:
        - query: 查询文本，可以是股票代码、名称、拼音等
        - threshold: 模糊匹配阈值，默认80
        
        返回:
        - 匹配的股票代码，如果没找到返回None
        """
        methods = [
            ('exact', 70),
            ('rapid', threshold),
            ('fuzzy', threshold-5),
            ('inverted', threshold-10)
        ]
        
        for method, thresh in methods:
            ts_code = None
            if method == 'exact':
                ts_code = self.exact_match(query)
            elif method == 'rapid':
                ts_code = self.rapidfuzz_match(query, thresh)
            elif method == 'fuzzy':
                ts_code = self.fuzzywuzzy_match(query, thresh)
            elif method == 'inverted':
                ts_code = self.inverted_index_match(query)
            
            if ts_code:
                match_row = self.df[self.df['ts_code'] == ts_code].iloc[0]
                return match_row['ts_code']
        
        return None
    def get_name(self, query, threshold=80):
        """
        根据查询文本返回匹配的股票名称(name)
        
        参数:
        - query: 查询文本，可以是股票代码、名称、拼音等
        - threshold: 模糊匹配阈值，默认80
        
        返回:
        - 匹配的股票名称，如果没找到返回None
        """
        methods = [
            ('exact', 70),
            ('rapid', threshold),
            ('fuzzy', threshold-5),
            ('inverted', threshold-10)
        ]
        
        for method, thresh in methods:
            ts_code = None
            if method == 'exact':
                ts_code = self.exact_match(query)
            elif method == 'rapid':
                ts_code = self.rapidfuzz_match(query, thresh)
            elif method == 'fuzzy':
                ts_code = self.fuzzywuzzy_match(query, thresh)
            elif method == 'inverted':
                ts_code = self.inverted_index_match(query)
            
            if ts_code:
                match_row = self.df[self.df['ts_code'] == ts_code].iloc[0]
                return match_row['name']
        
        return None