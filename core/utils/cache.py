import os
import pickle
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd

def load_pickle_cache(cache_file: str) -> Optional[Dict[str, Any]]:
    """
    从pickle文件加载缓存数据。
    
    Args:
        cache_file: pickle文件的路径
        
    Returns:
        Optional[Dict[str, Any]]: 加载的缓存数据，如果加载失败返回None
    """
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if isinstance(cache_data, dict) and 'data' in cache_data and 'timestamp' in cache_data:
                    cache_age = datetime.now() - cache_data['timestamp']
                    if cache_age <= timedelta(days=5):  # 检查缓存是否在5天内
                        return cache_data['data']
    except Exception as e:
        logging.error(f"Error loading cache from {cache_file}: {str(e)}")
    return None

def save_pickle_cache(cache_file: str, data: Dict[str, Any]) -> None:
    """
    将数据保存到pickle缓存文件。
    
    Args:
        cache_file: pickle文件的路径
        data: 要保存的数据
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # 保存数据和时间戳
        cache_data = {
            'data': data,
            'timestamp': datetime.now()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logging.error(f"Error saving cache to {cache_file}: {str(e)}")

class CodeNameCache:
    CACHE_FILE = './json/code_name_cache.pickle'
    
    @staticmethod
    def fetch_stock_data() -> pd.DataFrame:
        """
        从API获取股票数据。
        
        Returns:
            pd.DataFrame: 股票数据DataFrame
        """
        import akshare as ak
        return ak.stock_info_a_code_name()
    
    @staticmethod
    def process_stock_data(df: pd.DataFrame) -> Dict[str, str]:
        """
        处理股票数据DataFrame，转换为代码-名称映射字典。
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            Dict[str, str]: 代码-名称映射字典
        """
        return dict(zip(df["code"], df["name"]))