import logging
import os
from datetime import datetime

class SingletonLogger:
    _instance = None

    @classmethod
    def get_logger(cls, name=None, level=logging.DEBUG, log_file=None, backtester_mode=False):
        if cls._instance is None:
            cls._instance = cls._setup_logging(name, level, log_file, backtester_mode)
        return cls._instance

    @staticmethod
    def _setup_logging(name, level, log_file, backtester_mode):
        logger_name = name or __name__
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # 如果logger已经有处理器，就不再添加新的处理器
        if not logger.handlers:
            # 创建格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # 创建控制台输出处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # 添加控制台处理器到logger
            logger.addHandler(console_handler)

            # 设置默认日志文件路径
            if log_file is None:
                log_dir = "./output/logs"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{logger_name}")

            # 确保日志文件目录存在
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # 创建文件处理器
            date_str = datetime.now().strftime("%Y%m%d")
            if backtester_mode:
                file_handler = logging.FileHandler(f'{log_file}_backtest_{date_str}.log', encoding='utf-8')
            else:
                file_handler = logging.FileHandler(f'{log_file}_{date_str}.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            # 添加文件处理器到logger
            logger.addHandler(file_handler)

        return logger