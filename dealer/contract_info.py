

import re
from .futures_provider import MainContractProvider

class ContractInfo:
    """
    合约信息，包括合约基本信息和合约手续费信息
    可以计算合约的保证金，手续费，开仓和平仓的盈亏，收益率等
    """
    def __init__(self, contract: str):
        self.contract = contract
        self.futures_provider = MainContractProvider()
        self.contract_info = self.futures_provider.get_futures_comm_info(contract)
        self.contract_fees_info = self.futures_provider.get_futures_fees_info(contract)

    @property
    def min_point_value(self):
        return self.contract_fees_info.get('最小跳动', 0)

    def calculate_margin(self, price: float, volume: int, is_long: bool = True) -> float:
        """
        计算保证金
        :param price: 合约价格
        :param volume: 合约数量
        :param is_long: 是否为做多，默认为True
        :return: 保证金金额
        """
        if is_long:
            margin_rate = self.contract_fees_info.get('做多保证金率（按金额）', 0)
            margin_per_lot = self.contract_fees_info.get('做多保证金（按手）', 0)
        else:
            margin_rate = self.contract_fees_info.get('做空保证金率（按金额）', 0)
            margin_per_lot = self.contract_fees_info.get('做空保证金（按手）', 0)

        contract_multiplier = self.contract_fees_info.get('合约乘数', 1)
        
        # 如果有按金额的保证金率，优先使用
        if margin_rate > 0:
            contract_value = price * contract_multiplier * volume
            margin = contract_value * margin_rate
        else:
            # 如果没有按金额的保证金率，使用按手的保证金
            margin = margin_per_lot * volume

        return margin
    
    def calculate_fee(self, price: float, volume: int, is_open: bool, is_close_today: bool = False) -> float:
        """
        计算手续费
        :param price: 合约价格
        :param volume: 合约数量
        :param is_open: 是否为开仓（True为开仓，False为平仓）
        :param is_close_today: 是否为平今（仅在平仓时有效）
        :return: 手续费金额
        """
        contract_multiplier = self.contract_fees_info.get('合约乘数', 1)

        if is_open:
            fee_rate = self.contract_fees_info.get('开仓费率（按金额）', 0)
        elif is_close_today:
            fee_rate = self.contract_fees_info.get('平今费率（按金额）', 0)
        else:
            fee_rate = self.contract_fees_info.get('平仓费率（按金额）', 0)

        # 判断手续费类型（按比例还是固定金额）
        if isinstance(fee_rate, float):
            # 按比例收取手续费
            fee = price * volume * contract_multiplier * fee_rate
        else:
            # 固定金额手续费
            fee = fee_rate * volume

        return fee

    def parse_fee_string(self, fee_string):
        if isinstance(fee_string, (int, float)):
            return float(fee_string)
        
        # 尝试直接转换为float
        try:
            return float(fee_string)
        except ValueError:
            pass
        
        # 查找包含数字的模式
        patterns = [
            r'(\d+\.?\d*)元/手',  # 匹配 "5.6元/手" 或 "5元/手"
            r'(\d+\.?\d*)%',  # 匹配 "0.01%"
            r'(\d+\.?\d*)/万',  # 匹配 "0.5/万"
            r'(\d+\.?\d*)'  # 匹配任何数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, fee_string)
            if match:
                value = float(match.group(1))
                if '%' in fee_string:
                    return value / 100  # 将百分比转换为小数
                elif '/万' in fee_string:
                    return value / 10000  # 将万分比转换为小数
                elif '元/手' in fee_string:
                    return value  # 返回固定金额
                else:
                    return value  # 返回解析出的数值
        
        # 如果没有找到匹配的数字，返回0
        return 0
    def calculate_fee1(self, price: float, volume: int, is_open: bool, is_close_today: bool = False) -> float:
        """
        计算手续费
        :param price: 合约价格
        :param volume: 合约数量
        :param is_open: 是否为开仓（True为开仓，False为平仓）
        :param is_close_today: 是否为平今（仅在平仓时有效）
        :return: 手续费金额
        """
        contract_multiplier = self.contract_fees_info.get('合约乘数', 1)

        if is_open:
            fee_rate = self.contract_info.get('手续费标准-开仓-万分之', 0)
            fee_fixed = float( self.parse_fee_string(self.contract_info.get('手续费标准-开仓-元', 0)))
        elif is_close_today:
            fee_rate = float(self.parse_fee_string(self.contract_info.get('手续费标准-平今-万分之', 0)))
            fee_fixed = float(self.parse_fee_string(self.contract_info.get('手续费标准-平今-元', 0)))
        else:
            fee_rate = float(self.parse_fee_string(self.contract_info.get('手续费标准-平昨-万分之', 0)))
            fee_fixed = float(self.parse_fee_string(self.contract_info.get('手续费标准-平昨-元', 0)))

        # 计算按比例的手续费
        fee_by_rate = price * volume * contract_multiplier * (fee_rate / 10000) if fee_rate else 0
        
        # 计算固定手续费
        fee_by_fixed = fee_fixed * volume if fee_fixed else 0

        # 返回较大的那个手续费
        return max(fee_by_rate, fee_by_fixed)
    def parse_fee_string1(self,fee_string):
        # 尝试直接转换为float
        try:
            return float(fee_string)
        except ValueError:
            pass
        
        # 查找包含数字的模式
        patterns = [
            r'(\d+\.?\d*)元',  # 匹配 "5.6元" 或 "5元"
            r'(\d+\.?\d*)/万分之',  # 匹配 "0.51/万分之"
            r'\((\d+\.?\d*)元\)',  # 匹配 "(5.6元)"
            r'(\d+\.?\d*)'  # 匹配任何数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, fee_string)
            if match:
                return float(match.group(1))
        
        # 如果没有找到匹配的数字，返回None
        return None
    def calculate_pnl(self, open_price: float, close_price: float, volume: int) -> float:
        """
        计算开仓平仓盈亏
        :param open_price: 开仓价格
        :param close_price: 平仓价格
        :param volume: 合约数量
        :return: 盈亏金额
        """
        contract_multiplier = self.contract_fees_info.get('合约乘数', 1)
        pnl = (close_price - open_price) * volume * contract_multiplier
        return pnl

    def calculate_roi(self, open_price: float, close_price: float, volume: int) -> float:
        """
        计算收益率
        :param open_price: 开仓价格
        :param close_price: 平仓价格
        :param volume: 合约数量
        :return: 收益率（百分比）
        """
        pnl = self.calculate_pnl(open_price, close_price, volume)
        initial_margin = self.calculate_margin(open_price, volume)
        roi = (pnl / initial_margin) * 100
        return roi
    
    @property
    def multiplier(self) -> float:
        return self.contract_fees_info.get('合约乘数', 1)
    
    @property
    def tick_value(self) -> float:
        return self.contract_fees_info.get('1Tick平仓盈亏', 0)
    
    @property
    def tick_size(self) -> float:
        return self.contract_fees_info.get('最小跳动', 0)