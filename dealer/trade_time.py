# 创建交易时间字典
import re
from typing import Literal


trading_hours = {
    'CU': ['15:00', '01:00'],
    'CU-C/P': ['15:00', '01:00'],
    'AL': ['15:00', '01:00'],
    'AL-C/P': ['15:00', '01:00'],
    'PB': ['15:00', '01:00'],
    'ZN': ['15:00', '01:00'],
    'ZN-C/P': ['15:00', '01:00'],
    'SN': ['15:00', '01:00'],
    'NI': ['15:00', '01:00'],
    'SS': ['15:00', '01:00'],
    'AU': ['15:00', '02:30'],
    'AG': ['15:00', '02:30'],
    'AU-C/P': ['15:00', '02:30'],
    'RB': ['15:00', '23:00'],
    'HC': ['15:00', '23:00'],
    'BU': ['15:00', '23:00'],
    'RU': ['15:00', '23:00'],
    'FU': ['15:00', '23:00'],
    'SP': ['15:00', '23:00'],
    'RU-C/P': ['15:00', '23:00'],
    'WR': ['15:00', None],  # 无夜盘交易
    'A': ['15:00', '23:00'],
    'B': ['15:00', '23:00'],
    'M': ['15:00', '23:00'],
    'M-C/P': ['15:00', '23:00'],
    'Y': ['15:00', '23:00'],
    'P': ['15:00', '23:00'],
    'I': ['15:00', '23:00'],
    'J': ['15:00', '23:00'],
    'JM': ['15:00', '23:00'],
    'C': ['15:00', '23:00'],
    'CS': ['15:00', '23:00'],
    'L': ['15:00', '23:00'],
    'V': ['15:00', '23:00'],
    'PP': ['15:00', '23:00'],
    'EG': ['15:00', '23:00'],
    'C-C/P': ['15:00', '23:00'],
    'RR': ['15:00', '23:00'],
    'EB': ['15:00', '23:00'],
    'I-C/P': ['15:00', '23:00'],
    'PG': ['15:00', '23:00'],
    'PG-C/P': ['15:00', '23:00'],
    'L-C/P': ['15:00', '23:00'],
    'V-C/P': ['15:00', '23:00'],
    'PP-C/P': ['15:00', '23:00'],
    'P-C/P': ['15:00', '23:00'],
    'JD': ['15:00', None],  # 无夜盘交易
    'FB': ['15:00', None],  # 无夜盘交易
    'BB': ['15:00', None],  # 无夜盘交易
    'LH': ['15:00', None],  # 无夜盘交易
    'RM': ['15:00', '23:00'],
    'OI': ['15:00', '23:00'],
    'CF': ['15:00', '23:00'],
    'TA': ['15:00', '23:00'],
    'SR': ['15:00', '23:00'],
    'SR-C/P': ['15:00', '23:00'],
    'MA': ['15:00', '23:00'],
    'FG': ['15:00', '23:00'],
    'ZC': ['15:00', '23:00'],
    'CY': ['15:00', '23:00'],
    'CF-C/P': ['15:00', '23:00'],
    'SA': ['15:00', '23:00'],
    'TA-C/P': ['15:00', '23:00'],
    'MA-C/P': ['15:00', '23:00'],
    'RM-C/P': ['15:00', '23:00'],
    'ZC-C/P': ['15:00', '23:00'],
    'PF': ['15:00', '23:00'],
    'JR': ['15:00', None],  # 无夜盘交易
    'RS': ['15:00', None],  # 无夜盘交易
    'PM': ['15:00', None],  # 无夜盘交易
    'WH': ['15:00', None],  # 无夜盘交易
    'RI': ['15:00', None],  # 无夜盘交易
    'LR': ['15:00', None],  # 无夜盘交易
    'SF': ['15:00', None],  # 无夜盘交易
    'SM': ['15:00', None],  # 无夜盘交易
    'AP': ['15:00', None],  # 无夜盘交易
    'CJ': ['15:00', None],  # 无夜盘交易
    'UR': ['15:00', None],  # 无夜盘交易
    'PK': ['15:00', None],  # 无夜盘交易
    'PX': ['15:00', None],  # 无夜盘交易
    'SH': ['15:00', None],  # 无夜盘交易
    'IF': ['15:00', None],  # 无夜盘交易
    'IH': ['15:00', None],  # 无夜盘交易
    'IC': ['15:00', None],  # 无夜盘交易
    "IM": ['15:00', None],  # 无夜盘交易
    'IO-C/P': ['15:00', None],  # 无夜盘交易
    'TF': ['15:15', None],  # 无夜盘交易
    'T': ['15:15', None],  # 无夜盘交易
    'TS': ['15:15', None],  # 无夜盘交易
    'SC': ['15:00', '02:30'],
    'SC-C/P': ['15:00', '02:30'],
    'NR': ['15:00', '23:00'],
    'LU': ['15:00', '23:00'],
    'BC': ['15:00', '01:00'],
    'SI': ['15:00', None],  # 无夜盘交易
    'LC': ['15:00', None],  # 无夜盘交易
}

# 查询函数
def get_trading_end_time(code: str, session: Literal['day', 'night']='day') -> str:
    # 去掉数字开头的所有字符，并转换为大写
    cleaned_code = re.sub(r'\d.*', '', code).upper()
    
    if cleaned_code in trading_hours:
        if session == 'day':
            return trading_hours[cleaned_code][0]
        elif session == 'night':
            return trading_hours[cleaned_code][1]
        else:
            return "无效的交易时段。请选择 'day' 或 'night'"
    else:
        return "合约代码不存在"
    
trading_times = {
    "CU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "CU-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "AL": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "AL-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "PB": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "ZN": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "ZN-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "SN": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "NI": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "SS": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "AU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "02:30"}},
    "AG": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "02:30"}},
    "AU-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "02:30"}},
    "RB": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "HC": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "BU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "RU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "FU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "SP": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "RU-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "WR": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "SC": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "02:30"}},
    "SC-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "02:30"}},
    "NR": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "LU": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "BC": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "01:00"}},
    "SI": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "LC": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "A": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "B": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "M": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "M-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "Y": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "I": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "J": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "JM": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "C": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "CS": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "L": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "V": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "PP": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "EG": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "C-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "RR": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "EB": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "I-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "PG": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "PG-CP": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "L-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "V-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "PP-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "P-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "JD": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "FB": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "BB": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "LH": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "RM": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "OI": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "CF": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "TA": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "SR": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "SR-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "MA": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "FG": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "ZC": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "CY": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "CF-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "SA": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "TA-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "MA-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "RM-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "ZC-C/P": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "PF": {"day": {"start": "09:00", "end": "15:00"}, "night": {"start": "21:00", "end": "23:00"}},
    "JR": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "RS": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "PM": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "WH": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "RI": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "LR": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "SF": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "SM": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "AP": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "CJ": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "UR": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "PK": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "PX": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "SH": {"day": {"start": "09:00", "end": "15:00"}, "night": None},
    "IF": {"day": {"start": "09:30", "end": "15:00"}, "night": None},
    "IH": {"day": {"start": "09:30", "end": "15:00"}, "night": None},
    "IM": {"day": {"start": "09:30", "end": "15:00"}, "night": None},
    "IC": {"day": {"start": "09:30", "end": "15:00"}, "night": None},
    "IO-C/P": {"day": {"start": "09:30", "end": "15:00"}, "night": None},
    "TF": {"day": {"start": "09:15", "end": "15:15"}, "night": None},
    "T": {"day": {"start": "09:15", "end": "15:15"}, "night": None},
    "TS": {"day": {"start": "09:15", "end": "15:15"}, "night": None}
}


# 定义查询函数
def get_trading_time(symbol: str, time_period:Literal['day', 'night']='day') -> dict:
    """
    根据symbol和时间段（'day' 或 'night'）返回开盘和收盘时间的字典。
    
    :param symbol: 合约代码，如 'CU', 'SC'
    :param time_period: 查询时段，'day' 或 'night'
    :return: 包含start和end时间的字典，如果未找到则返回提示字典
    """
    time_info = trading_times.get(symbol.upper())
    
    if not time_info:
        return {"error": f"未找到合约代码 {symbol} 的交易时间。"}
    
    period_info = time_info.get(time_period.lower())
    
    if not period_info:
        return {"error": f"合约代码 {symbol} 在 {time_period} 时段无交易。"}
    
    return period_info