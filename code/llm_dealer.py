

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BarData:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class CompressedBarData:
    timestamp: int
    close: float

class HistoricalData:
    def __init__(self, max_daily_bars: int, max_hourly_bars: int, max_minute_bars: int):
        self.daily_bars: List[BarData] = []
        self.hourly_bars: List[BarData] = []
        self.minute_bars: List[CompressedBarData] = []
        self.today_minute_bars: List[BarData] = []
        
        self.max_daily_bars = max_daily_bars
        self.max_hourly_bars = max_hourly_bars
        self.max_minute_bars = max_minute_bars

    def update(self, new_bar: BarData):
        # 实现更新逻辑
        pass

    def get_formatted_history(self) -> Dict[str, List]:
        # 实现格式化历史数据的逻辑
        pass


class LLMDealer:
    def __init__(self, llm_client, symbol, max_history_days=30, max_hourly_bars=24, max_minute_bars=60):
        """
        初始化 LLMDealer 类
        
        :param llm_client: LLM API 客户端
        :param symbol: 交易的合约符号
        :param max_history_days: 保留的最大日线历史数据天数
        :param max_hourly_bars: 保留的最大小时线历史数据条数
        :param max_minute_bars: 保留的最大分钟线历史数据条数
        """
        self.llm_client = llm_client
        self.symbol = symbol
        self.max_history_days = max_history_days
        self.max_hourly_bars = max_hourly_bars
        self.max_minute_bars = max_minute_bars
        
        self.daily_history = []
        self.hourly_history = []
        self.minute_history = []
        self.today_minute_bars = []
        self.last_msg = ""
        self.position = 0  # 当前持仓量，正数表示多头，负数表示空头
    
    def on_bar(self, bar: BarData, news: str) -> Tuple[str, str]:
        """
        处理每分钟的 bar 数据和新闻

        :param bar: 最新的分钟 bar 数据
        :param news: 最近 1 分钟的新闻内容
        :return: 交易指令和下一次 on_bar 需要的消息
        """
        # 更新历史数据
        self.historical_data.update(bar)
        
        # 准备 LLM 输入
        llm_input = self._prepare_llm_input(bar, news)
        
        # 调用 LLM API
        llm_response = self.llm_client.one_chat(llm_input)
        
        # 解析 LLM 输出
        trade_instruction, next_msg = self._parse_llm_output(llm_response)
        
        # 执行交易指令
        if trade_instruction:
            self._execute_trade(trade_instruction)
        
        # 更新 last_msg 为下一次使用
        self.last_msg = next_msg
        
        # 计算当前的 bar index
        current_time = pd.Timestamp(bar.timestamp)
        market_open = pd.Timestamp(current_time.date()).replace(hour=9, minute=30)
        today_bar_index = (current_time - market_open).total_seconds() // 60
        
        # 记录日志
        self._log_bar_info(bar, news, trade_instruction, today_bar_index)
        
        return trade_instruction, next_msg

    def _prepare_llm_input(self, bar: BarData, news: str) -> str:
        """准备发送给 LLM 的输入数据"""
        formatted_history = self.historical_data.get_formatted_history()
        
        input_template = f"""
        上一次的消息: {self.last_msg}
        当前 bar index: {self._get_today_bar_index(bar.timestamp)}
        
        日线历史:
        {formatted_history['daily']}
        
        小时线历史:
        {formatted_history['hourly']}
        
        分钟线历史:
        {formatted_history['minute']}
        
        今日分钟线:
        {formatted_history['today_minute']}
        
        当前 bar 数据:
        时间: {bar.timestamp}
        开盘: {bar.open}
        最高: {bar.high}
        最低: {bar.low}
        收盘: {bar.close}
        成交量: {bar.volume}
        
        最新新闻:
        {news}
        
        当前持仓: {self.position}
        
        请根据以上信息，给出交易指令（buy/sell/short/cover）或不交易（hold），并提供下一次需要的消息。
        """
        return input_template

    def _parse_llm_output(self, llm_response: str) -> Tuple[str, str]:
        """解析 LLM 的输出"""
        # 使用正则表达式或其他方法解析 LLM 的响应
        # 提取交易指令和下一次需要的消息
        # 这里需要根据实际的 LLM 输出格式进行调整
        trade_instruction_match = re.search(r"交易指令：(buy|sell|short|cover|hold)", llm_response)
        next_msg_match = re.search(r"下一次消息：(.+)", llm_response, re.DOTALL)
        
        trade_instruction = trade_instruction_match.group(1) if trade_instruction_match else "hold"
        next_msg = next_msg_match.group(1) if next_msg_match else ""
        
        return trade_instruction, next_msg

    def _execute_trade(self, trade_instruction: str):
        """执行交易指令"""
        if trade_instruction == "buy":
            self.position += 1
        elif trade_instruction == "sell":
            self.position -= 1
        elif trade_instruction == "short":
            self.position -= 1
        elif trade_instruction == "cover":
            self.position += 1

    def _get_today_bar_index(self, timestamp: int) -> int:
        """计算当前的 bar index"""
        current_time = pd.Timestamp(timestamp)
        market_open = pd.Timestamp(current_time.date()).replace(hour=9, minute=30)
        return int((current_time - market_open).total_seconds() // 60)

    def _log_bar_info(self, bar: BarData, news: str, trade_instruction: str, today_bar_index: int):
        """记录每个 bar 的信息"""
        log_msg = f"""
        时间: {bar.timestamp}, Bar Index: {today_bar_index}
        价格: 开 {bar.open}, 高 {bar.high}, 低 {bar.low}, 收 {bar.close}
        成交量: {bar.volume}
        新闻: {news[:100]}...  # 截取前100个字符
        交易指令: {trade_instruction}
        当前持仓: {self.position}
        """
        logging.info(log_msg)