from enum import Enum
import json
import os
import pickle
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from .futures_position import FuturesPosition, DirectType, ContractInfo



class FuturesPositionManager:
    def __init__(self):
        self.open_positions: Dict[str, FuturesPosition] = {}
        self.closed_positions: Dict[str, List[FuturesPosition]] = {}
        self.profit_records: Dict[str, Dict[str, float]] = {}  
        self.stop_loss = None

    def _get_position_key(self, instrument_id: str, direction: DirectType):
        return f"{instrument_id}_{direction}"


    def set_stop_loss(self, stop_loss: float):
        self.stop_loss = stop_loss

    def check_stop_loss(self, instrument_id: str, current_price: float) -> bool:
        position = self.get_open_position(instrument_id)
        if position and position.Direction == DirectType.Buy and current_price <= self.stop_loss:
            return True
        return False

    def get_open_position(self, instrument_id: str, direction: Optional[DirectType] = None):
        if direction is None:
            # 如果没有指定方向,优先返回买入(多头)仓位
            key_buy = self._get_position_key(instrument_id, DirectType.Buy)
            if key_buy in self.open_positions:
                return self.open_positions[key_buy]
            key_sell = self._get_position_key(instrument_id, DirectType.Sell)
            if key_sell in self.open_positions:
                return self.open_positions[key_sell]
            # 如果既没有买入也没有卖出仓位，返回一个新的买入仓位
            return FuturesPosition(ContractInfo(instrument_id))
        else:
            key = self._get_position_key(instrument_id, direction)
            if key in self.open_positions:
                return self.open_positions[key]
            # 如果找不到指定方向的仓位，返回一个新的仓位
            return FuturesPosition(ContractInfo(instrument_id))

    def close_position(self, instrument_id: str, direction: DirectType, quantity: int, price: float):
        key = self._get_position_key(instrument_id, direction)
        if key not in self.open_positions:
            return None

        position = self.open_positions[key]
        closed_position = position.close_position(quantity, price)
        
        if position.Position == 0:
            del self.open_positions[key]
            if instrument_id not in self.closed_positions:
                self.closed_positions[instrument_id] = []
            self.closed_positions[instrument_id].append(closed_position)
        
        return closed_position

    def update_position_profits(self, instrument_id: str, current_price: float):
        for direction in [DirectType.Buy, DirectType.Sell]:
            key = self._get_position_key(instrument_id, direction)
            if key in self.open_positions:
                self.open_positions[key].calculate_profit(current_price)

    def get_average_price(self, instrument_id: str, is_long: bool) -> float:
        direction = DirectType.Buy if is_long else DirectType.Sell
        key = self._get_position_key(instrument_id, direction)
        position = self.open_positions.get(key)
        return position.OpenPrice if position and position.Position != 0 else 0.0


    def get_open_position1(self, instrument_id: str, direction: Optional[DirectType] = None):
        if direction is None:
            # 如果没有指定方向,优先返回买入(多头)仓位
            key_buy = self._get_position_key(instrument_id, DirectType.Buy)
            if key_buy in self.open_positions:
                return self.open_positions[key_buy]
            key_sell = self._get_position_key(instrument_id, DirectType.Sell)
            return self.open_positions.get(key_sell)
        else:
            key = self._get_position_key(instrument_id, direction)
            return self.open_positions.get(key)

    def calculate_total_profit(self) -> float:
        total_profit = 0.0
        for position in self.open_positions.values():
            total_profit += position.CloseProfit + position.PositionProfit
        for positions in self.closed_positions.values():
            for position in positions:
                total_profit += position.CloseProfit
        return total_profit

    def calculate_profit_by_instrument(self, instrument_id: str, bar: pd.Series) -> Dict[str, float]:
        if 'close' not in bar:
            raise ValueError("Bar data does not contain 'close' price")

        current_price = bar['close']
        realized_profit = 0.0
        unrealized_profit = 0.0
        total_commission = 0.0
        total_margin = 0.0

        # 处理开仓位置
        for direction in [DirectType.Buy, DirectType.Sell]:
            key = self._get_position_key(instrument_id, direction)
            if key in self.open_positions:
                position = self.open_positions[key]
                position.calculate_profit(current_price)
                unrealized_profit += position.PositionProfit
                realized_profit += position.CloseProfit
                total_commission += position.Commission
                
                # 直接使用 position.Margin 作为保证金占用
                total_margin += position.Margin

        # 处理已平仓位置
        if instrument_id in self.closed_positions:
            for closed_position in self.closed_positions[instrument_id]:
                realized_profit += closed_position.CloseProfit
                total_commission += closed_position.Commission

        total_profit = realized_profit + unrealized_profit - total_commission

        # 计算基于保证金的盈亏比例
        profit_margin_ratio = total_profit / total_margin if total_margin > 0 else 0

        # 更新或初始化profit_records
        if instrument_id not in self.profit_records:
            self.profit_records[instrument_id] = {
                "highest_unrealized_profit": unrealized_profit,
                "lowest_unrealized_profit": unrealized_profit,
                "highest_profit_margin_ratio": profit_margin_ratio,
                "lowest_profit_margin_ratio": profit_margin_ratio
            }
        else:
            self.profit_records[instrument_id]["highest_unrealized_profit"] = max(
                self.profit_records[instrument_id]["highest_unrealized_profit"],
                unrealized_profit
            )
            self.profit_records[instrument_id]["lowest_unrealized_profit"] = min(
                self.profit_records[instrument_id]["lowest_unrealized_profit"],
                unrealized_profit
            )
            self.profit_records[instrument_id]["highest_profit_margin_ratio"] = max(
                self.profit_records[instrument_id]["highest_profit_margin_ratio"],
                profit_margin_ratio
            )
            self.profit_records[instrument_id]["lowest_profit_margin_ratio"] = min(
                self.profit_records[instrument_id]["lowest_profit_margin_ratio"],
                profit_margin_ratio
            )

        return {
            "realized_profit": realized_profit,
            "unrealized_profit": unrealized_profit,
            "total_commission": total_commission,
            "total_profit": total_profit,
            "total_margin": total_margin,
            "profit_margin_ratio": profit_margin_ratio,
            "highest_unrealized_profit": self.profit_records[instrument_id]["highest_unrealized_profit"],
            "lowest_unrealized_profit": self.profit_records[instrument_id]["lowest_unrealized_profit"],
            "highest_profit_margin_ratio": self.profit_records[instrument_id]["highest_profit_margin_ratio"],
            "lowest_profit_margin_ratio": self.profit_records[instrument_id]["lowest_profit_margin_ratio"]
        }

    def reset_profit_records(self, instrument_id: str = None):
        if instrument_id:
            if instrument_id in self.profit_records:
                del self.profit_records[instrument_id]
        else:
            self.profit_records.clear()


    def save_to_file(self, filename):
        data = {
            'closed_positions': self.closed_positions
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_file(cls, filename):
        manager = cls()
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            manager.closed_positions = data.get('closed_positions', {})
        return manager


    def __str__(self):
        result = "FuturesPositionManager:\n"
        result += "  Open Positions:\n"
        for instrument_id, position in self.open_positions.items():
            result += f"    {instrument_id}: {position}\n"
        result += "  Closed Positions:\n"
        for instrument_id, positions in self.closed_positions.items():
            result += f"    {instrument_id}:\n"
            for position in positions:
                result += f"      {position}\n"
        return result
    
    def get_current_position(self) -> Dict[str, int]:
        return {instrument_id: position.Position for instrument_id, position in self.open_positions.items()}

    def get_position_details(self) -> str:
        return self.__str__()

    def open_long(self, instrument_id: str, quantity: int, price: float) -> FuturesPosition:
        return self.open_position(instrument_id, DirectType.Buy, quantity, price, datetime.now())

    def close_long(self, instrument_id: str, quantity: int, price: float) -> FuturesPosition:
        return self.close_position(instrument_id, DirectType.Buy, quantity, price)

    def open_short(self, instrument_id: str, quantity: int, price: float) -> FuturesPosition:
        return self.open_position(instrument_id, DirectType.Sell, quantity, price, datetime.now())

    def close_short(self, instrument_id: str, quantity: int, price: float) -> FuturesPosition:
        return self.close_position(instrument_id, DirectType.Sell, quantity, price)

    def get_position(self, instrument_id: str) -> FuturesPosition:
        return self.get_open_position(instrument_id)

    def add_position(self, direction: int, price: float, quantity: int, instrument_id: str):
        if direction == 1:  # 多头
            self.open_long(instrument_id, quantity, price)
        elif direction == -1:  # 空头
            self.open_short(instrument_id, quantity, price)
        else:
            raise ValueError("方向必须是 1 (多头) 或 -1 (空头)")

    def remove_position(self, direction: int, price: float, quantity: int, instrument_id: str):
        if direction == 1:  # 多头
            self.close_long(instrument_id, quantity, price)
        elif direction == -1:  # 空头
            self.close_short(instrument_id, quantity, price)
        else:
            raise ValueError("方向必须是 1 (多头) 或 -1 (空头)")

    def get_long_position(self, instrument_id: str) -> int:
        position = self.get_open_position(instrument_id, direction=DirectType.Buy)
        return position.Position if position and position.Direction == DirectType.Buy else 0

    def get_short_position(self, instrument_id: str) -> int:
        position = self.get_open_position(instrument_id, direction=DirectType.Sell)
        return abs(position.Position) if position and position.Direction == DirectType.Sell else 0

    def get_net_position(self, instrument_id: str) -> int:
        long_position = self.get_long_position(instrument_id)
        short_position = self.get_short_position(instrument_id)
        net_position = long_position - short_position
        return net_position