import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Any

import duckdb
import hashlib
import pickle
import time
from functools import wraps
from threading import Lock


from .log import logger


class RateLimiter:
    def __init__(self, name: str = 'def', rate: float = 0.0):
        """
        :param rate: 每秒允许的调用次数，默认值为 0.0
        """
        self.name = name
        self.rate: float = rate
        self.lock = threading.Lock()
        self.last_called = 0.0

    def __str__(self):
        return f"RateLimiter(name={self.name}, rate={self.rate}, lock={self.lock}, last_called={self.last_called})"




class DBCache:
    def __init__(self, db_path='./data/cache.db', default_ttl=7 * 24 * 60 * 60):
        self.default_ttl = default_ttl
        self.lock = Lock()  # 添加锁以保护数据库访问
        dir_path = os.path.dirname(db_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.conn = duckdb.connect(db_path)
        # 修改表结构，使用expiry而不是timestamp
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                expiry TIMESTAMP
            )
        ''')

    def cache(self, ttl=None, limiter: RateLimiter = None):
        """
        serial_locker 串行锁
        """
        ttl = ttl if ttl is not None else self.default_ttl

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal ttl
                key_args = args
                # 判断是否为实例方法
                if func.__code__:
                    if func.__code__.co_varnames:
                        if len(func.__code__.co_varnames) > 0:
                            if 'self' == func.__code__.co_varnames[0]:
                                # 实例方法，排除 self
                                key_args = args[1:]  # 排除 self

                # 生成唯一的缓存键
                key = f"{func.__name__}:{key_args}:{kwargs}"
                key_hash = key #hashlib.sha256(key.encode()).hexdigest()
                v = self.get(key_hash)
                if v is not None:
                    return v
                logger.debug(f"Cache miss: {key_hash}")
                # 若缓存不存在或已过期，计算结果并缓存
                # 限速锁，增加串行等待时间
                if limiter:
                    with limiter.lock:
                        if limiter.rate > 0:
                            current_time = time.time()
                            elapsed = current_time - limiter.last_called
                            wait_time = max(float(0), (1 / limiter.rate) - elapsed)
                            if wait_time > 0:
                                logger.debug(f'limit wait: {wait_time}, by {limiter}')
                                time.sleep(wait_time)  # 等待直到可以调用
                            # 等待时间有可能很长，再次尝试获取
                            v = self.get(key_hash)
                            if v is not None:
                                logger.debug(f"Cache hit key:{key} by limit wait.")
                                return v
                            result = func(*args, **kwargs)
                            limiter.last_called = time.time()  # 更新上次调用时间
                        else:
                            result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # 计算过期时间并存储
                current_time = time.time()
                expiry_time = current_time + ttl if ttl is not None else float('inf')


                # 将结果缓存到数据库
                with self.lock:  # 确保数据库访问是安全的
                    self.conn.execute(
                        'INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, TO_TIMESTAMP(?))',
                        (key_hash, pickle.dumps(result), expiry_time)
                    )
                logger.info(f"Cached hit: {key_hash}, expiry_time:{datetime.fromtimestamp(expiry_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, {expiry_time}")
                return result

            return wrapper

        return decorator

    def clear_expired(self):
        """清理已过期的缓存项"""
        current_time = time.time()
        with self.lock:  # 确保数据库访问是安全的
            self.conn.execute('DELETE FROM cache WHERE expiry < TO_TIMESTAMP(?)', (current_time,))

    def get(self, key) -> Optional[Any]:
        """直接获取缓存值"""
        # 获取当前时间
        current_time = int(time.time())

        # 打印为可读的 UTC 时间格式
        logger.debug(
            f"Query key: {key}, {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, {current_time}")  # 显示到毫秒                # 检查缓存是否存在且未过期
        with self.lock:  # 确保数据库访问是安全的
            cached_data = self.conn.execute(
                'SELECT value, expiry, key FROM cache WHERE key = ?',
                (key,)
            ).fetchone()
        if cached_data:
            logger.debug(f"Find -> {cached_data}, timestamp: {cached_data[1].timestamp() if cached_data else None}")
            if cached_data[1].timestamp() > current_time:
                logger.debug(f"Cache hit key:{key}")
                return pickle.loads(cached_data[0])

        return None

    def print_all(self):
        with self.lock:  # 确保数据库访问是安全的
            list = self.conn.execute('SELECT key,expiry FROM cache').fetchall()
            for r in list:
                logger.info(f'Cache hit: {r[0]}, expiry: {r[1].strftime("%Y-%m-%d %H:%M:%S")}')

    def set(self, key, value, ttl=None):
        """设置缓存值"""
        ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = time.time() + ttl if ttl is not None else float('inf')
        with self.lock:  # 确保数据库访问是安全的
            self.conn.execute(
                'INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, TO_TIMESTAMP(?))',
                (key, pickle.dumps(value), expiry_time)
            )


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(threadName)s] %(levelname)-5s %(module)s:%(lineno)-4d - %(message)s')
    # 创建 DuckDBCache 实例，设定默认 ttl 为 10 秒
    cache = DBCache()


    class Dog:
        def __init__(self):
            self.x = 1

        @cache.cache()
        def ww(self):
            return self.x


    cache.print_all()

    lock = RateLimiter('test', rate=1)

    # 示例函数，使用缓存装饰器
    @cache.cache(limiter=lock)  # 使用默认 ttl
    def expensive_calculation(x, y):
        logger.info(f"Calculating expensive for {x} and {y}")
        return x * y + x - y

    logger.info(Dog().ww())
    logger.info(expensive_calculation(1, 2))  # 缓存行为测试

    #定义线程池中的工作函数，接受参数
    def thread_test(x, y):
        logger.info(expensive_calculation(x, y))  # 缓存行为测试
        time.sleep(5)
        logger.info(expensive_calculation(x, y))  # 缓存未过期
        time.sleep(6)
        logger.info(expensive_calculation(x, y))  # 缓存过期重新计算

    def direct_get_set_test():
        logger.info("测试 get/set:")
        cache.set("test_key", "test_value", ttl=5)
        logger.info(cache.get("test_key"))  # 应该输出: test_value
        time.sleep(6)
        logger.info(cache.get("test_key"))  # 应该输出: None


    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 使用线程池执行任务
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        # 提交任务，使用不同的参数
        params = [(2, 3), (3, 4), (5, 6), (7, 8), (10, 12),  (13, 14),  (15, 16),  (17, 18),  (19, 20),  (21, 22)]
        for x, y in params:  # 启动三个线程执行 thread_test 函数
            futures.append(executor.submit(thread_test, x, y))

        # 等待所有线程完成并处理结果
        for future in as_completed(futures):
            try:
                future.result()  # 处理结果（可以选择性打印或处理）
            except Exception as e:
                logger.error(f"Thread generated an exception: {e}")