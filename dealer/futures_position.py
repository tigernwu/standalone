from enum import Enum
import os
from typing import Optional
from .contract_info import ContractInfo
from datetime import datetime   
from .direction import DirectType


class FuturesPosition:
    def __init__(self, contract_info: ContractInfo):
        self.contract_info = contract_info
        self.InstrumentID = contract_info.contract
        self.Direction = DirectType.Buy  # 默认为买入(多头)
        self.Position = 0
        self.TdPosition = 0
        self.YdPosition = 0
        self.OpenPrice = 0.0
        self.PositionCost = 0.0
        self.CloseProfit = 0.0
        self.PositionProfit = 0.0
        self.Commission = 0.0
        self.Margin = 0.0
        self._exit_price = 0.0
        self._entry_time = datetime.now()
        self._exit_time: Optional[datetime] = None
        self._is_closed = False


    def update_position(self, direction: DirectType, quantity: int, price: float):
        old_position = self.Position
        old_cost = self.OpenPrice * abs(old_position) * self.contract_info.multiplier

        if direction == self.Direction:
            # 增加现有仓位
            new_position = old_position + quantity if self.Direction == DirectType.Buy else old_position - quantity
            new_cost = old_cost + price * quantity * self.contract_info.multiplier
        else:
            # 减少现有仓位或改变仓位方向
            new_position = old_position - quantity if self.Direction == DirectType.Buy else old_position + quantity
            if abs(new_position) >= abs(old_position):
                # 完全平掉旧仓位，并开新仓位
                close_quantity = abs(old_position)
                open_quantity = abs(new_position)
                close_profit = (price - self.OpenPrice) * close_quantity if self.Direction == DirectType.Buy else (self.OpenPrice - price) * close_quantity
                close_profit *= self.contract_info.multiplier
                self.CloseProfit += close_profit
                new_cost = price * open_quantity * self.contract_info.multiplier
                self.Direction = direction
            else:
                # 仓位减少但方向未变
                cost_reduction_ratio = quantity / abs(old_position)
                new_cost = old_cost * (1 - cost_reduction_ratio)

        self.Position = new_position
        self.OpenPrice = round(new_cost / (abs(new_position) * self.contract_info.multiplier), 2) if new_position != 0 else 0.0

        # 更新今仓和昨仓
        if direction == self.Direction:
            self.TdPosition += quantity
        else:
            self.TdPosition = max(0, self.TdPosition - quantity)
            self.YdPosition = max(0, self.YdPosition - max(0, quantity - self.TdPosition))

        self.PositionCost = new_cost

        # 更新佣金和保证金
        self.update_commission(self.contract_info.calculate_fee(price, quantity, is_open=True))
        self.update_margin()

    def close_position(self, quantity: int, price: float) -> 'FuturesPosition':
        if quantity > abs(self.Position):
            quantity = abs(self.Position)

        # 创建一个新的FuturesPosition对象来表示关闭的仓位
        closed_position = FuturesPosition(self.contract_info)
        closed_position.Direction = self.Direction
        closed_position.Position = quantity
        closed_position.OpenPrice = self.OpenPrice
        closed_position._exit_price = price
        closed_position._exit_time = datetime.now()
        closed_position._is_closed = True

        # 计算平仓盈亏
        close_profit = (price - self.OpenPrice) * quantity if self.Direction == DirectType.Buy else (self.OpenPrice - price) * quantity
        close_profit *= self.contract_info.multiplier
        closed_position.CloseProfit = close_profit

        # 更新当前仓位
        self.CloseProfit += close_profit
        self.Position -= quantity if self.Direction == DirectType.Buy else -quantity
        self.TdPosition = max(0, self.TdPosition - quantity)
        self.YdPosition = max(0, self.YdPosition - max(0, quantity - self.TdPosition))

        if self.Position == 0:
            self.OpenPrice = 0.0
            self.PositionCost = 0.0
            self._is_closed = True
            self._exit_price = price
            self._exit_time = datetime.now()
        else:
            # 如果还有剩余仓位，更新持仓成本
            self.PositionCost = self.OpenPrice * abs(self.Position) * self.contract_info.multiplier

        # 更新佣金和保证金
        commission = self.contract_info.calculate_fee(price, quantity, is_open=False)
        self.update_commission(commission)
        closed_position.update_commission(commission)
        self.update_margin()
        closed_position.update_margin()

        return closed_position

    def calculate_profit(self, current_price: float) -> float:
        if self.Position == 0:
            return self.CloseProfit

        unrealized_profit = (current_price - self.OpenPrice) * self.Position if self.Direction == DirectType.Buy else (self.OpenPrice - current_price) * self.Position
        unrealized_profit *= self.contract_info.multiplier
        self.PositionProfit = unrealized_profit
        return unrealized_profit

    def calculate_close_profit(self) -> float:
        if self.Position == 0:
            return self.CloseProfit

        close_point = (self.exit_price - self.OpenPrice) * self.Position if self.Direction == DirectType.Buy else (self.OpenPrice - self.exit_price) * self.Position
        profit = close_point * self.contract_info.multiplier
        return profit

    def update_commission(self, commission: float):
        self.Commission += commission

    def update_margin(self):
        self.Margin = self.contract_info.calculate_margin(self.OpenPrice, abs(self.Position))

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def exit_price(self) -> float:
        return self._exit_price

    @property
    def exit_time(self) -> Optional[datetime]:
        return self._exit_time
    
    @property
    def entry_time(self) -> Optional[datetime]:
        return self._entry_time


    def __str__(self):
        return (f"FuturesPosition(InstrumentID={self.InstrumentID}, Direction={self.Direction}, "
                f"Position={self.Position}, TdPosition={self.TdPosition}, YdPosition={self.YdPosition}, "
                f"OpenPrice={self.OpenPrice:.2f}, PositionCost={self.PositionCost:.2f}, "
                f"CloseProfit={self.CloseProfit:.2f}, PositionProfit={self.PositionProfit:.2f}, "
                f"Commission={self.Commission:.2f}, Margin={self.Margin:.2f})"
                f", is_closed={self._is_closed}, exit_price={self._exit_price:.2f}, exit_time={self._exit_time}")