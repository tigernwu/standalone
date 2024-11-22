from threading import Lock, Event
from typing import Dict, Any, Callable, Optional, TypeVar, ParamSpec, Tuple
from functools import wraps
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)
import time

T = TypeVar('T')  # 返回值类型
P = ParamSpec('P')  # 参数类型

class ThreadSafeCache:
    """
    线程安全的缓存装饰器类，用于管理对象的缓存状态和访问
    """
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expiry_time)
        self._locks: Dict[str, Lock] = {}
        self._initialization_events: Dict[str, Event] = {}
        self._initialization_flags: Dict[str, bool] = {}
        self._logger = logging.getLogger(__name__)

    def get_lock(self, cache_key: str) -> Lock:
        """获取指定缓存键的锁，如果不存在则创建"""
        if cache_key not in self._locks:
            self._locks[cache_key] = Lock()
        return self._locks[cache_key]

    def get_event(self, cache_key: str) -> Event:
        """获取指定缓存键的事件，如果不存在则创建"""
        if cache_key not in self._initialization_events:
            self._initialization_events[cache_key] = Event()
        return self._initialization_events[cache_key]

    def is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效（存在且未过期）"""
        if cache_key not in self._cache:
            return False
        _, expiry_time = self._cache[cache_key]
        # 如果expiry_time为None，表示永不过期
        return expiry_time is None or time.time() < expiry_time

    def thread_safe_cached(self, cache_key: str, timeout: Optional[float] = None, retry_config: Optional[dict] = None):
        """
        线程安全的缓存装饰器
        
        Args:
            cache_key (str): 缓存键名
            timeout (float, optional): 缓存有效期（秒），None表示永不过期
            retry_config (dict, optional): 重试配置参数
        """
        default_retry_config = {
            'stop': stop_after_attempt(3),
            'wait': wait_exponential(multiplier=1, min=4, max=10),
            'before_sleep': before_sleep_log(self._logger, logging.WARNING)
        }
        retry_config = retry_config or default_retry_config

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            @retry(**retry_config)
            def fetch_data(*args, **kwargs) -> T:
                """实际获取数据的函数，带有重试机制"""
                return func(*args, **kwargs)

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                lock = self.get_lock(cache_key)
                event = self.get_event(cache_key)
                
                # 检查缓存是否有效
                if self.is_cache_valid(cache_key):
                    return self._cache[cache_key][0]  # 返回缓存值

                should_initialize = False

                # 获取锁来检查和更新初始化状态
                with lock:
                    # 再次检查缓存（可能在获取锁的过程中被其他线程初始化）
                    if self.is_cache_valid(cache_key):
                        return self._cache[cache_key][0]
                    
                    # 检查是否有其他线程正在初始化
                    if not self._initialization_flags.get(cache_key, False):
                        self._initialization_flags[cache_key] = True
                        event.clear()
                        should_initialize = True

                # 如果其他线程在初始化，等待完成
                if not should_initialize:
                    self._logger.debug(f"Waiting for {cache_key} cache initialization...")
                    if not event.wait(timeout=120):  
                        self._logger.error(f"Cache initialization timeout for {cache_key}")
                        raise TimeoutError(f"Cache initialization timeout for {cache_key}")
                    
                    # 检查缓存是否成功初始化
                    if self.is_cache_valid(cache_key):
                        return self._cache[cache_key][0]
                    raise RuntimeError(f"Cache initialization failed for {cache_key}")

                # 执行初始化
                try:
                    result = fetch_data(*args, **kwargs)
                    
                    with lock:
                        # 计算过期时间
                        expiry_time = None if timeout is None else time.time() + timeout
                        self._cache[cache_key] = (result, expiry_time)
                        self._initialization_flags[cache_key] = False
                        event.set()
                    
                    return result
                    
                except Exception as e:
                    with lock:
                        if cache_key in self._cache:
                            del self._cache[cache_key]
                        self._initialization_flags[cache_key] = False
                        event.set()
                    
                    self._logger.error(f"Failed to initialize {cache_key} cache: {str(e)}")
                    raise

            def clear_cache():
                """清除指定键的缓存"""
                with self.get_lock(cache_key):
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                    self._initialization_flags[cache_key] = False
                    self.get_event(cache_key).clear()

            # 添加清除缓存的方法
            wrapper.clear_cache = clear_cache
            return wrapper

        return decorator