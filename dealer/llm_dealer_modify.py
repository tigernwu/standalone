





from datetime import datetime, timedelta, time as dt_time
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from ta import add_all_ta_features
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
import pandas as pd
import pytz
from dealer.futures_provider import MainContractProvider
from dealer.llm_dealer import TradePosition, TradePositionManager
from dealer.trade_time import get_trading_end_time
beijing_tz = pytz.timezone('Asia/Shanghai')


class LLMDealer:
    def __init__(self, llm_client, symbol: str,data_provider: MainContractProvider,trade_rules:str="" ,
                 max_daily_bars: int = 60, max_hourly_bars: int = 30, max_minute_bars: int = 240,
                 backtest_date: Optional[str] = None, compact_mode: bool = False,
                 max_position: int = 1):
        self._setup_logging()
        self.trade_rules = trade_rules
        self.symbol = symbol
        self.night_closing_time = self._get_night_closing_time()
        self.backtest_date = backtest_date
        self.data_provider = data_provider
        self.llm_client = llm_client
        self.last_news_time = None
        self.news_summary = ""
        self.is_backtest = backtest_date is not None
        self.max_daily_bars = max_daily_bars
        self.max_hourly_bars = max_hourly_bars
        self.max_minute_bars = max_minute_bars
        self.max_position = max_position
        self.compact_mode = compact_mode
        self.backtest_date = backtest_date or datetime.now().strftime('%Y-%m-%d')
        self.trade_history = []
        self.important_trade_plan = None
        
        self.today_minute_bars = pd.DataFrame()
        self.last_msg = ""
        self.current_date = None
        self.last_trade_date = None  # 添加这个属性

        self.position_manager = TradePositionManager()
        self.total_profit = 0

        self.trading_hours = [
            (dt_time(9, 0), dt_time(11, 30)),
            (dt_time(13, 0), dt_time(15, 0)),
            (dt_time(21, 0), dt_time(23, 59)),
            (dt_time(0, 0), dt_time(2, 30))
        ]
        logging.basicConfig(level=logging.DEBUG)
        self.timezone = pytz.timezone('Asia/Shanghai') 
        

        self.daily_history = self._initialize_history('D')
        self.hourly_history = self._initialize_history('60')  
        self.minute_history = self._initialize_history('1')

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create a handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(console_handler)

        # Ensure the output directory exists
        os.makedirs('./output', exist_ok=True)

        # Create a file handler
        file_handler = logging.FileHandler(f'./output/log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def update_position(self, positions: List[TradePosition]):
        """
        更新当前持仓信息
        
        :param positions: 持仓列表
        """
        self.position_manager.positions = positions
        self.logger.info(f"LLMDealer position updated for {self.symbol}: "
                    f"Net position: {self.position_manager.get_current_position()}")

    def _get_latest_news(self):
        if self.is_backtest:
            return pd.DataFrame()  # 回测模式下不读取新闻
        news_df = self.data_provider.get_futures_news(self.symbol, page_num=0, page_size=20)
        if news_df is not None and not news_df.empty:
            return news_df.sort_values('publish_time', ascending=False)
        return pd.DataFrame()

    def _summarize_news(self, news_df):
        if news_df.empty:
            return ""

        news_text = "\n".join(f"- {row['title']}" for _, row in news_df.iterrows())
        prompt = f"请将以下新闻整理成不超过200字的今日交易提示简报：\n\n{news_text}"
        
        summary = self.llm_client.one_chat(prompt)
        return summary[:200]  # Ensure the summary doesn't exceed 200 characters

    def _get_night_closing_time(self) -> Optional[dt_time]:
        night_end = get_trading_end_time(self.symbol, 'night')
        if isinstance(night_end, str) and ':' in night_end:
            hour, minute = map(int, night_end.split(':'))
            return dt_time(hour, minute)
        return None

    def _update_news(self, current_datetime):
        if self.is_backtest:
            return False  # 回测模式下不更新新闻

        try:
            news_df = self._get_latest_news()
            if news_df.empty:
                self.logger.info("No new news available")
                return False

            # 安全地解析新闻时间
            def safe_parse_time(time_str):
                try:
                    # 假设时间戳是毫秒级的
                    return pd.to_datetime(int(time_str) / 1000, unit='s', utc=True).tz_convert(beijing_tz)
                except ValueError:
                    # 如果失败，尝试直接解析字符串
                    try:
                        return pd.to_datetime(time_str).tz_localize(beijing_tz)
                    except:
                        self.logger.error(f"Failed to parse news time: {time_str}")
                        return None

            latest_news_time = safe_parse_time(news_df['publish_time'].iloc[0])
            
            if latest_news_time is None:
                self.logger.warning("Failed to parse latest news time, skipping news update")
                return False

            if self.last_news_time is None or latest_news_time > self.last_news_time:
                self.last_news_time = latest_news_time
                new_summary = self._summarize_news(news_df)
                if new_summary != self.news_summary:
                    self.news_summary = new_summary
                    self.logger.info(f"Updated news summary: {self.news_summary[:100]}...")
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Error in _update_news: {str(e)}", exc_info=True)
            return False
       
    def _is_trading_time(self, dt: datetime) -> bool:
        t = dt.time()
        for start, end in self.trading_hours:
            if start <= t <= end:
                return True
        return False

    def _filter_trading_data(self, df: pd.DataFrame) -> pd.DataFrame:
        def is_trading_time(dt):
            t = dt.time()
            return any(start <= t <= end for start, end in self.trading_hours)
        
        mask = df['datetime'].apply(is_trading_time)
        filtered_df = df[mask]
        
        self.logger.debug(f"Trading hours filter: {len(df)} -> {len(filtered_df)} rows")
        return filtered_df

    def _get_today_data(self, date: datetime.date) -> pd.DataFrame:
        self.logger.info(f"Fetching data for date: {date}")

        if self.is_backtest:
            today_data = self.data_provider.get_bar_data(self.symbol, '1', date.strftime('%Y-%m-%d'))
        else:
            today_data = self.data_provider.get_akbar(self.symbol, '1m')
            today_data = today_data[today_data.index.date == date]
            today_data = today_data.reset_index()

        self.logger.info(f"Raw data fetched: {len(today_data)} rows")

        if today_data.empty:
            self.logger.warning(f"No data returned from data provider for date {date}")
            return pd.DataFrame()

        # Ensure column names are consistent
        today_data = today_data.rename(columns={
            'open_interest': 'hold'
        })

        if self.is_backtest:
            filtered_data = today_data[today_data['trading_date'] == date]
        else:
            filtered_data = today_data[today_data['datetime'].dt.date == date]

        if filtered_data.empty:
            self.logger.warning(f"No data found for date {date} after filtering.")
        else:
            self.logger.info(f"Filtered data: {len(filtered_data)} rows")

        return filtered_data
    
    def _validate_and_prepare_data(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        original_len = len(df)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
        df = self._filter_trading_data(df)
        
        self.logger.info(f"Bars for date {date}: Original: {original_len}, After filtering: {len(df)}")
        
        if len(df) > self.max_minute_bars:
            self.logger.warning(f"Unusually high number of bars ({len(df)}) for date {date}. Trimming to {self.max_minute_bars} bars.")
            df = df.tail(self.max_minute_bars)
        
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据，处理空值和异常值"""
        # 将所有列转换为数值类型，非数值替换为 NaN
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 使用前向填充方法填充 NaN 值
        df = df.fillna(method='ffill')

        # 如果仍有 NaN 值（比如在数据开始处），则使用后向填充
        df = df.fillna(method='bfill')

        # 确保没有负值
        for col in numeric_columns:
            df[col] = df[col].clip(lower=0)

        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 5:
            return df
        
        try:
            # 创建DataFrame的副本
            df = df.copy()
            
            # 确保 'close' 列是数值类型
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # 检查是否有足够的有效数据
            if df['close'].notna().sum() < 5:
                return df

            df['sma_10'] = df['close'].rolling(window=min(10, len(df))).mean()
            df['ema_20'] = df['close'].ewm(span=min(20, len(df)), adjust=False).mean()
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=min(14, len(df))).rsi()
            
            macd = ta.trend.MACD(close=df['close'], window_slow=min(26, len(df)), window_fast=min(12, len(df)), window_sign=min(9, len(df)))
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bollinger = ta.volatility.BollingerBands(close=df['close'], window=min(20, len(df)), window_dev=2)
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            df['bollinger_low'] = bollinger.bollinger_lband()
            
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=min(14, len(df))).average_true_range()
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
        
        return df

    def _format_indicators(self, indicators: pd.Series) -> str:
        def format_value(value):
            if pd.isna(value):
                return 'N/A'
            elif isinstance(value, (int, float)):
                return f"{value:.2f}"
            return str(value)

        if self.compact_mode:
            return f"""
            SMA10: {format_value(indicators.get('sma_10'))}
            EMA20: {format_value(indicators.get('ema_20'))}
            RSI: {format_value(indicators.get('rsi'))}
            MACD: {format_value(indicators.get('macd'))}
            BB高: {format_value(indicators.get('bollinger_high'))}
            BB低: {format_value(indicators.get('bollinger_low'))}
            """
        else:
            return f"""
            10周期简单移动平均线 (SMA): {format_value(indicators.get('sma_10'))}
            20周期指数移动平均线 (EMA): {format_value(indicators.get('ema_20'))}
            相对强弱指标 (RSI): {format_value(indicators.get('rsi'))}
            MACD: {format_value(indicators.get('macd'))}
            MACD信号线: {format_value(indicators.get('macd_signal'))}
            平均真实范围 (ATR): {format_value(indicators.get('atr'))}
            布林带上轨: {format_value(indicators.get('bollinger_high'))}
            布林带中轨: {format_value(indicators.get('bollinger_mid'))}
            布林带下轨: {format_value(indicators.get('bollinger_low'))}
            """
        
    def _initialize_history(self, period: Literal['1', '5', '15', '30', '60', 'D']) -> pd.DataFrame:
        try:
            frequency_map = {'1': '1m', '5': '5m', '15': '15m', '30': '30m', '60': '60m', 'D': 'D'}
            frequency = frequency_map[period]
            
            if self.is_backtest:
                df = self.data_provider.get_bar_data(self.symbol, period, self.backtest_date)
            else:
                df = self.data_provider.get_akbar(self.symbol, frequency)
            
            if df is None or df.empty:
                self.logger.warning(f"No data available for period {period}")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
            
            df = df.reset_index()
            
            # Ensure 'datetime' column is datetime type
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Select and order the required columns
            columns_to_keep = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
            df = df[columns_to_keep]
            
            return self._limit_history(df, period)
        except Exception as e:
            self.logger.error(f"Error initializing history for period {period}: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])

    def _limit_history(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """根据时间周期限制历史数据的长度"""
        if df.empty:
            return df
        if period == 'D':
            return df.tail(self.max_daily_bars)
        elif period == '60':
            return df.tail(self.max_hourly_bars)
        else:
            return df.tail(self.max_minute_bars)

    def _update_histories(self, bar: pd.Series):
        """更新历史数据"""
        # 更新分钟数据
        self.minute_history = pd.concat([self.minute_history, bar.to_frame().T], ignore_index=True).tail(self.max_minute_bars)
        
        # 更新小时数据
        if bar['datetime'].minute == 0:
            self.hourly_history = pd.concat([self.hourly_history, bar.to_frame().T], ignore_index=True).tail(self.max_hourly_bars)
        
        # 更新日线数据
        if bar['datetime'].hour == 15 and bar['datetime'].minute == 0:
            daily_bar = bar.copy()
            daily_bar['datetime'] = daily_bar['datetime'].date()
            self.daily_history = pd.concat([self.daily_history, daily_bar.to_frame().T], ignore_index=True).tail(self.max_daily_bars)

    def _format_history(self) -> dict:
        """格式化历史数据，确保所有数据都被包含，并且格式一致"""
        
        def format_dataframe(df: pd.DataFrame, max_rows: int = None) -> str:
            if max_rows and len(df) > max_rows:
                df = df.tail(max_rows)  # 只保留最后 max_rows 行
            
            df_reset = df.reset_index(drop=True)
            formatted = df_reset.to_string(index=True, index_names=False, 
                                            formatters={
                                                'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, pd.Timestamp) else str(x),
                                                'open': '{:.2f}'.format,
                                                'high': '{:.2f}'.format,
                                                'low': '{:.2f}'.format,
                                                'close': '{:.2f}'.format
                                            })
            return formatted

        return {
            'daily': format_dataframe(self.daily_history, self.max_daily_bars),
            'hourly': format_dataframe(self.hourly_history, self.max_hourly_bars),
            'minute': format_dataframe(self.minute_history, self.max_minute_bars),
            'today_minute': format_dataframe(pd.DataFrame(self.today_minute_bars))
        }

    def _compress_history(self, df: pd.DataFrame, period: str) -> str:
        if df.empty:
            return "No data available"
        
        df = df.tail(self.max_daily_bars if period == 'D' else self.max_hourly_bars if period == 'H' else self.max_minute_bars)
        
        summary = []
        for _, row in df.iterrows():
            if self.compact_mode:
                summary.append(f"{row['datetime'].strftime('%Y-%m-%d %H:%M' if period != 'D' else '%Y-%m-%d')}: "
                               f"C:{row['close']:.2f} V:{row['volume']}")
            else:
                summary.append(f"{row['datetime'].strftime('%Y-%m-%d %H:%M' if period != 'D' else '%Y-%m-%d')}: "
                               f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']}")
        
        return "\n".join(summary)
    
    def _prepare_llm_input(self, bar: pd.Series, news: str) -> str:
        modules = {
            "role": self._get_role_description(),
            "rules": self._get_trading_rules(),
            "market_data": self._get_market_data(bar),
            "technical_indicators": self._get_technical_indicators(bar),
            "position": self._get_position_info(),
            "profit_loss": self._get_profit_loss_info(bar),
            "news": self._get_news_analysis(news),
            "trade_history": self._get_trade_history(),
            "multi_timeframe": self._get_multi_timeframe_analysis(),
            "market_sentiment": self._get_market_sentiment(),
            "instructions": self._get_trading_instructions()
        }

        # 随机打乱模块顺序，以减少知识遮蔽
        module_order = list(modules.keys())
        random.shuffle(module_order)

        prompt = "请仔细分析以下每个部分的信息，并根据所有提供的数据做出交易决策：\n\n"

        for module in module_order:
            prompt += f"[{module.upper()}]\n{modules[module]}\n\n"

        if self.important_trade_plan:
            prompt += f"""
            当前保留的重要交易计划：
            指令: {self.important_trade_plan['instruction']}
            计划: {self.important_trade_plan['plan']}
            计划制定时的价格: {self.important_trade_plan['price']}

            请仔细观察上述"当前保留的重要交易计划"。
            1. 如果当前价格已经达到交易计划预期的执行位置，您应该执行相应的交易指令。
            2. 如果当前市场状况与交易计划不再符合，请在 trade_plan 中以"更新："开头提供更新后的交易计划。
            3. 如果交易计划仍然适用，但需要小幅调整（如调整止损位），也请在 trade_plan 中以"更新："开头提供调整后的计划。

            请确保您的响应考虑了这些要求，特别是在决定是否执行现有计划或提供更新时。
            """
        prompt += """
        请提供详细的交易计划：
        1. 如果指令为"hold"，请预估下一个可能的入场时机和条件。
        2. 如果指令为"buy"或"short"，必须指定具体的止损价和目标价。
        3. 如果指令为"sell"或"cover"，请预估下一次可能的入场时机和条件。
        4. 交易计划应考虑当前市场状况、技术指标和潜在风险。
        """
        prompt += """
        根据以上所有信息，请提供以下格式的 JSON 输出：
        ```json
        {
            "trade_instruction": "buy/sell/short/cover/hold",
            "quantity": "数量或'all'",
            "next_message": "下一次需要关注的信息",
            "trade_reason": "交易理由",
            "trade_plan": "交易计划，包括止损和目标价"
        }
        ```
        注意：请确保考虑所有提供的信息，特别是市场数据、技术指标、当前持仓和盈亏情况。
        """
        return prompt

    def _get_role_description(self) -> str:
        return """
        你是一位经验丰富的期货交易员，专注于日内交易策略。你的职责是：
        1. 分析市场数据和技术指标
        2. 评估当前持仓和盈亏情况
        3. 考虑最新新闻和市场情绪
        4. 根据多时间周期分析做出明智的交易决策
        5. 严格遵守交易规则，特别是风险管理原则
        """

    def _get_trading_rules(self) -> str:
        rules = [
            "所有开仓必须在当天收盘前平仓，不留过夜仓位。",
            f"最大持仓量为 {self.max_position} 手。这是硬性限制，任何时候都不能超过。如果发现当前持仓超过最大持仓，请立即减仓。",
            "日内仓位需要在每天15:00之前平仓。",
            "交易时必须考虑当前市场趋势、多时间周期分析和市场情绪。",
            "始终使用动态止损来保护盈利并限制损失。",
            "在做出任何交易决定之前，必须评估风险回报比。",
            f"严格遵守最大持仓 {self.max_position} 手的限制。不得以任何理由超过此限制。"
        ]
        if self.trade_rules:
            rules.append(f"额外规则：{self.trade_rules}")
        return "交易规则：\n" + "\n".join(f"- {rule}" for rule in rules)

    def _get_market_data(self, bar: pd.Series) -> str:
        return f"""
        当前市场数据：
        - 时间：{bar['datetime'].strftime('%Y-%m-%d %H:%M')}
        - 开盘价：{bar['open']:.2f}
        - 最高价：{bar['high']:.2f}
        - 最低价：{bar['low']:.2f}
        - 收盘价：{bar['close']:.2f}
        - 成交量：{bar['volume']}
        - 持仓量：{bar.get('open_interest', bar.get('hold', 'N/A'))}
        """

    def _get_technical_indicators(self, bar: pd.Series) -> str:
        indicators = self._calculate_indicators(pd.DataFrame([bar]))
        return self._format_indicators(indicators.iloc[-1])

    def _get_position_info(self) -> str:
        current_position = self.position_manager.get_current_position()
        remaining_position = self.max_position - abs(current_position)
        return f"""
        当前持仓状态：
        - 净持仓：{current_position} 手
        - 持仓方向：{"多头" if current_position > 0 else "空头" if current_position < 0 else "空仓"}
        - 最大允许持仓：{self.max_position} 手
        - 剩余可用持仓：{remaining_position} 手
        - 警告：任何交易决策都必须确保总持仓不超过 {self.max_position} 手
        """

    def _get_profit_loss_info(self, bar: pd.Series) -> str:
        profits = self.position_manager.calculate_profits(bar['close'])
        return f"""
        盈亏情况：
        - 已实现盈亏：{profits['realized_profit']:.2f}
        - 浮动盈亏：{profits['unrealized_profit']:.2f}
        - 总盈亏：{profits['total_profit']:.2f}
        - 最高未实现盈利：{profits['highest_unrealized_profit']:.2f}
        - 最低未实现盈利：{profits['lowest_unrealized_profit']:.2f}
        """

    def _get_news_analysis(self, news: str) -> str:
        if not news:
            return "无最新相关新闻。"
        return f"""
        最新相关新闻：
        {news}

        新闻分析要点：
        1. 评估该新闻对市场短期、中期和长期的潜在影响。
        2. 考虑市场是否已经消化了这个信息。
        3. 分析这个新闻是否符合之前的市场预期，如有出入可能导致更大的市场反应。
        4. 思考这个消息如何影响市场情绪和交易者行为。
        """

    def _get_trade_history(self) -> str:
        recent_trades = self.trade_history[-5:]  # 获取最近5笔交易
        if not recent_trades:
            return "无近期交易记录。"
        
        history = "最近5笔交易记录：\n"
        for trade in recent_trades:
            history += f"- {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} {trade['action']} {trade['quantity']}手 @ {trade['price']:.2f}, 盈亏: {trade['profit_loss']:.2f}\n"
        return history

    def _get_multi_timeframe_analysis(self) -> str:
        return f"""
        多时间周期分析：
        - 5分钟RSI：{self._calculate_rsi(self.minute_history.resample('5T').last(), 14):.2f}
        - 15分钟RSI：{self._calculate_rsi(self.minute_history.resample('15T').last(), 14):.2f}
        - 1小时趋势：{self._identify_trend(self.hourly_history)}
        - 日线趋势：{self._identify_trend(self.daily_history)}
        """

    def _get_market_sentiment(self) -> str:
        sentiment = self._calculate_market_sentiment()
        return f"""
        市场情绪指标：
        - 总体情绪：{sentiment['overall']}
        - 成交量比率：{sentiment['volume_ratio']:.2f}
        - 开仓意愿：{sentiment['open_interest_change']}
        """

    def _get_trading_instructions(self) -> str:
        return """
        交易指示：
        1. 仔细评估所有提供的信息，包括市场数据、技术指标、持仓状况、盈亏情况、新闻和市场情绪。
        2. 确保任何交易决策都严格遵守之前列出的交易规则，特别是最大持仓限制。
        3. 如果决定交易，请提供明确的理由和详细的交易计划。
        4. 即使决定持仓不变，也请解释原因并提供市场观察建议。
        5. 交易指令格式：
        - 买入开仓：'buy 数量'
        - 卖出开仓：'short 数量'
        - 买入平仓：'cover 数量'
        - 卖出平仓：'sell 数量'
        - 保持现状：'hold'
        6. 数量可以是具体的数字，或者使用'all'表示最大可能数量。
        7. 请始终考虑风险管理，包括设置合理的止损和止盈水平。
        8. 重要提醒：任何交易决策都必须确保总持仓不超过最大允许持仓量。在接近最大持仓时要特别谨慎。
        """

    def _calculate_market_sentiment(self) -> Dict[str, Any]:
        df = self.today_minute_bars.tail(30)  # 取最近30根K线
        
        volume_ma = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        open_interest_change = df['hold'].diff().sum()
        volume_ratio = current_volume / volume_ma
        
        sentiment = {
            'volume_ratio': volume_ratio,
            'open_interest_change': "增加" if open_interest_change > 0 else "减少" if open_interest_change < 0 else "不变"
        }
        
        if volume_ratio > 1.5 and open_interest_change > 0:
            sentiment['overall'] = "积极"
        elif volume_ratio < 0.5 and open_interest_change < 0:
            sentiment['overall'] = "消极"
        else:
            sentiment['overall'] = "中性"
        
        return sentiment

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float:
        if df.empty or len(df) < period:
            return float('nan')

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _identify_trend(self, df: pd.DataFrame) -> str:
        # 使用简单的移动平均线判断趋势
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        
        last_row = df.iloc[-1]
        if last_row['SMA20'] > last_row['SMA50'] and last_row['close'] > last_row['SMA20']:
            return "上涨"
        elif last_row['SMA20'] < last_row['SMA50'] and last_row['close'] < last_row['SMA20']:
            return "下跌"
        else:
            return "震荡"

    def _calculate_slippage(self, current_price: float) -> float:
        # 这里我们使用一个简单的固定比例滑点，您可以根据实际情况调整
        return current_price * 0.0005  # 0.05% 的滑点

    def _get_dynamic_stop_loss_info(self) -> str:
        current_position = self.position_manager.get_current_position()
        if current_position == 0:
            return "当前无持仓，无需设置止损"
        
        atr = self.today_minute_bars['atr'].iloc[-1]
        current_price = self.today_minute_bars['close'].iloc[-1]
        
        if current_position > 0:
            stop_loss = current_price - 2 * atr
            return f"建议动态止损价: {stop_loss:.2f} (当前价格 - 2 * ATR)"
        else:
            stop_loss = current_price + 2 * atr
            return f"建议动态止损价: {stop_loss:.2f} (当前价格 + 2 * ATR)"

    def _summarize_trade_history(self) -> str:
        if not self.trade_history:
            return "无交易历史"

        recent_trades = self.trade_history[-10:]  # 获取最近10笔交易
        summary = []
        total_profit_loss = 0
        win_count = 0
        loss_count = 0

        for i, trade in enumerate(recent_trades, 1):
            profit_loss = trade['profit_loss'] - (self.trade_history[i-2]['profit_loss'] if i > 1 else 0)
            if profit_loss > 0:
                win_count += 1
            elif profit_loss < 0:
                loss_count += 1
            total_profit_loss += profit_loss

            summary.append(f"交易{i}: {trade['action']} {trade['quantity']}手 @ {trade['price']:.2f}, "
                        f"盈亏: {profit_loss:.2f}, 原因: {trade['reason']}")

        win_rate = win_count / len(recent_trades) if recent_trades else 0
        avg_profit_loss = total_profit_loss / len(recent_trades) if recent_trades else 0

        summary_text = "\n".join(summary)
        stats = f"\n胜率: {win_rate:.2f}, 平均盈亏: {avg_profit_loss:.2f}, 总盈亏: {total_profit_loss:.2f}"

        return summary_text + stats

    def _format_trade_history(self) -> str:
        if not self.trade_history:
            return "无交易历史"

        recent_trades = self.trade_history[-10:]  # 获取最近10笔交易
        formatted_trades = []

        for i, trade in enumerate(recent_trades, 1):
            profit_loss = trade['profit_loss'] - (self.trade_history[i-2]['profit_loss'] if i > 1 else 0)
            formatted_trades.append(
                f"交易{i}: {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                f"{trade['action']} {trade['quantity']}手 @ {trade['price']:.2f}, "
                f"盈亏: {profit_loss:.2f}, 原因: {trade['reason']}"
            )

        return "\n".join(formatted_trades)

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float:
        if df.empty or len(df) < period:
            return float('nan')

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _format_history(self) -> dict:
        """格式化历史数据"""
        return {
            'daily': self.daily_history.to_string(index=False) if not self.daily_history.empty else "No daily data available",
            'hourly': self.hourly_history.to_string(index=False) if not self.hourly_history.empty else "No hourly data available",
        }

    def _parse_llm_output(self, llm_response: str) -> Tuple[str, Union[int, str], str, str, str]:
        try:
            # 提取 JSON 内容
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in the response")
            
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            # 解析 LLM 的输出
            trade_instruction = data.get('trade_instruction', 'hold').lower()
            next_msg = data.get('next_message', '')
            trade_reason = data.get('trade_reason', '')
            trade_plan = data.get('trade_plan', '')
            
            # 解析交易指令和数量
            instruction_parts = trade_instruction.split()
            action = instruction_parts[0]
            quantity = instruction_parts[1] if len(instruction_parts) > 1 else '1'
            
            if action not in ['buy', 'sell', 'short', 'cover', 'hold']:
                self.logger.warning(f"Invalid trade instruction: {action}. Defaulting to 'hold'.")
                return "hold", 1, next_msg, "", trade_plan
            
            if quantity.lower() == 'all':
                quantity = 'all'
            else:
                try:
                    quantity = int(quantity)
                except ValueError:
                    quantity = 1  # 默认数量为1
            
            # 检查是否有更新的交易计划
            if trade_plan.startswith("更新："):
                updated_plan = trade_plan[3:].strip()  # 移除 "更新：" 前缀
                self.important_trade_plan = {
                    'instruction': action,
                    'plan': updated_plan,
                    'price': float(data.get('current_price', 0))
                }
                self.logger.info(f"更新交易计划: {updated_plan}")
            
            # 不再自动执行 important_trade_plan，而是依赖 LLM 的决策
            return action, quantity, next_msg, trade_reason, trade_plan

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return "hold", 1, "", "JSON 解析错误", ""
        except Exception as e:
            self.logger.error(f"Error parsing LLM output: {e}")
            return "hold", 1, "", "解析错误", ""

    def _execute_trade(self, trade_instruction: str, quantity: Union[int, str], bar: pd.Series, trade_reason: str, trade_plan: str):
        current_datetime = bar['datetime']
        current_date = current_datetime.date()
        current_price = bar['close']

        self.logger.info(f"尝试执行交易: 指令={trade_instruction}, 数量={quantity}, 价格={current_price}")
        self.logger.info(f"交易理由: {trade_reason}")
        self.logger.info(f"交易计划: {trade_plan}")

        if self.last_trade_date != current_date:
            self._close_all_positions(current_price, current_datetime)
            self.last_trade_date = current_date

        if trade_instruction.lower() in ['buy', 'sell', 'short', 'cover']:
            self.important_trade_plan = {
                'instruction': trade_instruction,
                'plan': trade_plan,
                'price': current_price
            }
        elif trade_instruction.lower() == 'hold':
            # 根据特定规则更新保留的交易计划
            if self.important_trade_plan:
                # 这里可以添加更新逻辑，例如调整止损或目标价
                pass

        if trade_instruction.lower() == 'hold':
            self.logger.info("保持当前仓位，不执行交易。")
            return

        action = trade_instruction.lower()
        current_position = self.position_manager.get_current_position()

        self.logger.info(f"当前仓位: {current_position}, 最大仓位: {self.max_position}")

        # 计算滑点
        slippage = self._calculate_slippage(current_price)

        execution_price = current_price  # Default execution price
        actual_quantity = 0

        if action == "buy":
            max_buy = self.max_position - current_position
            qty = max_buy if quantity == 'all' else min(int(quantity), max_buy)
            if qty > 0:
                execution_price = current_price + slippage
                self.logger.info(f"尝试买入 {qty} 手，执行价格：{execution_price:.2f}")
                self.position_manager.open_position(execution_price, qty, True, current_datetime, trade_plan)
                actual_quantity = qty
            else:
                self.logger.warning(f"买入订单未执行: 已达到最大仓位或无效数量. 最大可买入: {max_buy}, 尝试买入: {quantity}")
        elif action == "sell":
            qty = current_position if quantity == 'all' else min(int(quantity), current_position)
            if qty > 0:
                execution_price = current_price - slippage
                self.logger.info(f"尝试卖出 {qty} 手，执行价格：{execution_price:.2f}")
                self.position_manager.close_positions(execution_price, qty, True, current_datetime)
                actual_quantity = qty
            else:
                self.logger.warning(f"卖出订单未执行: 无多头仓位或无效数量. 当前仓位: {current_position}, 尝试卖出: {quantity}")
        elif action == "short":
            max_short = self.max_position + current_position
            qty = max_short if quantity == 'all' else min(int(quantity), max_short)
            if qty > 0:
                execution_price = current_price - slippage
                self.logger.info(f"尝试做空 {qty} 手，执行价格：{execution_price:.2f}")
                self.position_manager.open_position(execution_price, qty, False, current_datetime, trade_plan)
                actual_quantity = qty
            else:
                self.logger.warning(f"做空订单未执行: 已达到最大仓位或无效数量. 最大可做空: {max_short}, 尝试做空: {quantity}")
        elif action == "cover":
            max_cover = abs(current_position) if current_position < 0 else 0
            qty = max_cover if quantity == 'all' else min(int(quantity), max_cover)
            if qty > 0:
                execution_price = current_price + slippage
                self.logger.info(f"尝试买入平空 {qty} 手，执行价格：{execution_price:.2f}")
                self.position_manager.close_positions(execution_price, qty, False, current_datetime)
                actual_quantity = qty
            else:
                self.logger.warning(f"平空订单未执行: 无空头仓位或无效数量. 当前仓位: {current_position}, 尝试平仓: {quantity}")
        else:
            self.logger.error(f"未知的交易动作: {action}")
            return

        self._force_close_if_needed(current_datetime, current_price)

        profits = self.position_manager.calculate_profits(current_price)
        self.total_profit = profits['total_profit']

        self.logger.info(f"执行交易后的仓位: {self.position_manager.get_current_position()}")
        self.logger.info(f"当前总盈亏: {self.total_profit:.2f}")
        self.logger.info(self.position_manager.get_position_details())

        # 添加交易记录
        if actual_quantity > 0:
            trade_record = {
                'timestamp': current_datetime,
                'action': action,
                'quantity': actual_quantity,
                'price': execution_price,
                'reason': trade_reason,
                'profit_loss': self.total_profit
            }
            self.trade_history.append(trade_record)

    def _close_all_positions(self, current_price: float, current_datetime: pd.Timestamp):
        self.position_manager.close_positions(current_price, float('inf'), True, current_datetime)
        self.position_manager.close_positions(current_price, float('inf'), False, current_datetime)

    def _force_close_if_needed(self, current_datetime: pd.Timestamp, current_price: float):
        day_closing_time = dt_time(14, 55)
        night_session_start = dt_time(21, 0)

        current_time = current_datetime.time()
        is_day_session = current_time < night_session_start and current_time >= dt_time(9, 0)
        is_night_session = current_time >= night_session_start or current_time < dt_time(9, 0)

        if is_day_session and current_time >= day_closing_time:
            self._close_all_positions(current_price, current_datetime)
            self.logger.info("日盘强制平仓")
        elif is_night_session and self.night_closing_time:
            # Check if it's within 5 minutes of the night closing time
            closing_window_start = (datetime.combine(datetime.min, self.night_closing_time) - timedelta(minutes=5)).time()
            if closing_window_start <= current_time <= self.night_closing_time:
                self._close_all_positions(current_price, current_datetime)
                self.logger.info("夜盘强制平仓")
            else:
                self.logger.info(f"夜盘交易，当前仓位：{self.position_manager.get_current_position()}")
        elif is_night_session and not self.night_closing_time:
            self.logger.info(f"夜盘交易（无强制平仓时间），当前仓位：{self.position_manager.get_current_position()}")

    def _get_today_bar_index(self, timestamp: pd.Timestamp) -> int:
        """
        Calculate the current bar index based on today's minute bars
        """
        if self.today_minute_bars.empty:
            return 0
        
        try:
            # Convert 'datetime' column to datetime type if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.today_minute_bars['datetime']):
                self.today_minute_bars['datetime'] = pd.to_datetime(self.today_minute_bars['datetime'], utc=True)
            
            # Now check if datetimes are timezone-aware
            if self.today_minute_bars['datetime'].dt.tz is None:
                # If not timezone-aware, assume they're in local time and make them timezone-aware
                self.today_minute_bars['datetime'] = self.today_minute_bars['datetime'].dt.tz_localize('Asia/Shanghai')
            
            # Convert to UTC
            self.today_minute_bars['datetime'] = self.today_minute_bars['datetime'].dt.tz_convert('UTC')
            
            # Ensure the input timestamp is in UTC
            utc_timestamp = timestamp.tz_convert('UTC')
            
            today_bars = self.today_minute_bars[self.today_minute_bars['datetime'].dt.date == utc_timestamp.date()]
            return len(today_bars)
        except Exception as e:
            self.logger.error(f"Error in _get_today_bar_index: {str(e)}", exc_info=True)
            return 0

    def _is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """
        判断给定时间是否在交易时间内
        """
        # 定义交易时间段（根据实际情况调整）
        trading_sessions = [
            ((9, 0), (11, 30)),   # 上午交易时段
            ((13, 0), (15, 0)),   # 下午交易时段
            ((21, 0), (23, 59)),  # 夜盘交易时段开始
            ((0, 0), (2, 30))     # 夜盘交易时段结束（跨天）
        ]
        
        time = timestamp.time()
        for start, end in trading_sessions:
            if start <= (time.hour, time.minute) <= end:
                return True
        return False

    def _log_bar_info(self, bar: Union[pd.Series, dict], news: str, trade_instruction: str,trade_reason, trade_plan):
        try:
            # Ensure the output directory exists
            os.makedirs('./output', exist_ok=True)

            # Create or get the file handler for the current date
            current_date = datetime.now().strftime('%Y%m%d')
            file_handler = next((h for h in self.logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'{current_date}.log')), None)
            
            if not file_handler:
                # If the file handler for the current date doesn't exist, create a new one
                file_path = f'./output/log{current_date}.log'
                file_handler = logging.FileHandler(file_path)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(file_handler)

                # Remove old file handlers
                for handler in self.logger.handlers[:]:
                    if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(f'{current_date}.log'):
                        self.logger.removeHandler(handler)
                        handler.close()

            # Convert bar to pd.Series if it's a dict
            if isinstance(bar, dict):
                bar = pd.Series(bar)

            # Format the log message
            log_msg = f"""
            时间: {pd.to_datetime(bar['datetime'])}, Bar Index: {self._get_today_bar_index(pd.to_datetime(bar['datetime']))}
            价格: 开 {bar['open']:.2f}, 高 {bar['high']:.2f}, 低 {bar['low']:.2f}, 收 {bar['close']:.2f}
            成交量: {bar['volume']}, 持仓量: {bar.get('open_interest', bar.get('hold', 'N/A'))}
            新闻: {news[:200] + '...' if news else '无新闻数据'}
            交易指令: {trade_instruction}
            交易理由: {trade_reason}
            交易计划: {trade_plan}
            当前持仓: {self.position_manager.get_current_position()}
            盈亏情况:
            {self.position_manager.calculate_profits(bar['close'])}
            {self.position_manager.get_position_details()}
            """

            # Log to file (DEBUG level includes all information)
            self.logger.debug(log_msg)

            # Log to console only if there's a trade instruction (excluding 'hold')
            if trade_instruction.lower() != 'hold':
                console_msg = f"时间: {pd.to_datetime(bar['datetime'])}, 价格: {bar['close']:.2f}, 交易指令: {trade_instruction}, 交易理由: {trade_reason[:50]}..., 当前持仓: {self.position_manager.get_current_position()}"
                self.logger.info(console_msg)

        except Exception as e:
            self.logger.error(f"Error in _log_bar_info: {str(e)}", exc_info=True)

    def parse_timestamp(self, timestamp):
        """解析时间戳"""
        self.logger.debug(f"Attempting to parse timestamp: {timestamp}")
        try:
            # 尝试多种可能的时间戳格式
            if isinstance(timestamp, (int, float)):
                # 如果时间戳是以毫秒为单位
                if timestamp > 1e12:
                    # 检查时间戳是否超出合理范围（2262年之后）
                    if timestamp > 9999999999999:
                        self.logger.warning(f"Abnormally large timestamp detected: {timestamp}")
                        # 尝试将其解释为纳秒级时间戳
                        try:
                            return pd.Timestamp(timestamp, unit='ns').tz_localize(beijing_tz)
                        except Exception:
                            self.logger.error(f"Failed to parse abnormally large timestamp: {timestamp}")
                            return datetime.now(beijing_tz)
                    return datetime.fromtimestamp(timestamp / 1000, tz=beijing_tz)
                # 如果时间戳是以秒为单位
                else:
                    return datetime.fromtimestamp(timestamp, tz=beijing_tz)
            elif isinstance(timestamp, str):
                # 尝试使用dateutil解析字符串格式的时间戳
                import dateutil.parser as parser
                return parser.parse(timestamp).astimezone(beijing_tz)
            elif isinstance(timestamp, pd.Timestamp):
                return timestamp.tz_localize(beijing_tz) if timestamp.tz is None else timestamp.tz_convert(beijing_tz)
            elif isinstance(timestamp, datetime):
                return timestamp.astimezone(beijing_tz) if timestamp.tzinfo else beijing_tz.localize(timestamp)
            else:
                raise ValueError(f"Unexpected timestamp type: {type(timestamp)}")
        except Exception as e:
            self.logger.error(f"Error parsing timestamp {timestamp}: {str(e)}")
            # 如果所有方法都失败，返回当前时间作为后备选项
            self.logger.warning("Using current time as fallback")
            return datetime.now(beijing_tz)

    def process_bar(self, bar: pd.Series, news: str = "") -> Tuple[str, Union[int, str], str]:
        try:
            # 确保使用正确的时间戳键
            time_key = 'time' if 'time' in bar else 'datetime'
            bar['datetime'] = self.parse_timestamp(bar[time_key])
            bar_date = bar['datetime'].date()

            if self.current_date != bar_date:
                self.current_date = bar_date
                self.today_minute_bars = self._get_today_data(bar_date)
                # Ensure the datetime column is in datetime format and timezone-aware
                self.today_minute_bars['datetime'] = pd.to_datetime(self.today_minute_bars['datetime'], utc=True)
                
                # If the datetimes are not timezone-aware, assume they're in local time and make them timezone-aware
                if self.today_minute_bars['datetime'].dt.tz is None:
                    self.today_minute_bars['datetime'] = self.today_minute_bars['datetime'].dt.tz_localize('Asia/Shanghai')
                
                # Convert to UTC
                self.today_minute_bars['datetime'] = self.today_minute_bars['datetime'].dt.tz_convert('UTC')
                self.last_trade_date = bar_date
                
                if not self.is_backtest:
                    self.last_news_time = None
                    self.news_summary = ""

            if not self._is_trading_time(bar['datetime']):
                return "hold", 0, ""

            self.today_minute_bars = pd.concat([self.today_minute_bars, bar.to_frame().T], ignore_index=True)

            news_updated = False
            if not self.is_backtest:
                news_updated = self._update_news(bar['datetime'])

            # 检查是否触发止损
            if self.position_manager.check_stop_loss(bar['close']):
                current_position = self.position_manager.get_current_position()
                if current_position > 0:
                    trade_instruction = "sell"
                else:
                    trade_instruction = "cover"
                quantity = abs(current_position)
                trade_reason = "触发动态止损"
                trade_plan = "平仓并重新评估市场情况"
            else:
                # 原有的交易逻辑
                llm_input = self._prepare_llm_input(bar, self.news_summary if (not self.is_backtest and (news_updated or len(self.today_minute_bars) == 1)) else "")
                llm_response = self.llm_client.one_chat(llm_input)
                trade_instruction, quantity, next_msg, trade_reason, trade_plan = self._parse_llm_output(llm_response)

            self._execute_trade(trade_instruction, quantity, bar, trade_reason, trade_plan)
            self._log_bar_info(bar, self.news_summary if news_updated else "", f"{trade_instruction} {quantity}", trade_reason, trade_plan)
            self.last_msg = next_msg
            return trade_instruction, quantity, next_msg, trade_reason, trade_plan
        except Exception as e:
            self.logger.error(f"Error processing bar: {str(e)}", exc_info=True)
            self.logger.error(f"Problematic bar data: {bar}")
            return "hold", 0, "", "处理错误", "无交易计划"
