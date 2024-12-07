import os
import hashlib
import threading
import functools
import pickle
import queue
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from datetime import datetime, timedelta
from contextlib import contextmanager
import duckdb
from ratelimit import limits, sleep_and_retry
from cachetools import TTLCache
import weakref

T = TypeVar('T')

class FunctionCache(Generic[T]):
    _instances: Dict[str, 'weakref.ref[FunctionCache]'] = {}
    _instances_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, db_path: str = "./data/function_cache.db", cache_ttl: int = 3600) -> 'FunctionCache':
        db_path = os.path.abspath(os.path.expanduser(db_path))
        
        with cls._instances_lock:
            instance_ref = cls._instances.get(db_path)
            instance = instance_ref() if instance_ref else None
            
            if instance is None:
                instance = super().__new__(cls)
                instance._initialize(db_path, cache_ttl)
                cls._instances[db_path] = weakref.ref(instance, 
                    lambda _: cls._instances.pop(db_path, None))
            return instance
    
    def __new__(cls, db_path: str = "./data/function_cache.db", cache_ttl: int = 3600) -> 'FunctionCache':
        return cls.get_instance(db_path, cache_ttl)
    
    def _initialize(self, db_path: str, cache_ttl: int) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._default_ttl = cache_ttl
        self._cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self._connection_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 启动后台线程
        self._writer_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self._cleaner_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._writer_thread.start()
        self._cleaner_thread.start()
    
    def _init_database(self) -> None:
        """初始化数据库表"""
        try:
            with self._get_connection(timeout=5) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS function_cache (
                        function_name VARCHAR,
                        args_hash VARCHAR,
                        result BLOB,
                        created_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        PRIMARY KEY (function_name, args_hash)
                    )
                """)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database: {e}")

    def _compute_args_hash(self, *args, **kwargs) -> str:
        """计算参数的哈希值"""
        if args and not kwargs:
            args_to_hash = args[0] if len(args) == 1 else args
        elif kwargs and not args:
            args_to_hash = kwargs
        else:
            args_to_hash = (args, kwargs) if args else kwargs
        
        try:
            args_str = pickle.dumps(args_to_hash)
            return hashlib.sha256(args_str).hexdigest()
        except Exception:
            backup_str = str(args) + str(kwargs)
            return hashlib.sha256(backup_str.encode()).hexdigest()
        
    @contextmanager
    def _get_connection(self, timeout: float = 1.0):
        """获取数据库连接的上下文管理器，带超时机制"""
        start_time = datetime.now()
        while True:
            if not hasattr(self._local, 'connection'):
                if self._connection_lock.acquire(timeout=timeout):
                    try:
                        if not hasattr(self._local, 'connection'):
                            self._local.connection = duckdb.connect(self.db_path)
                    finally:
                        self._connection_lock.release()
            
            if hasattr(self._local, 'connection'):
                break
                
            if (datetime.now() - start_time).total_seconds() > timeout:
                raise TimeoutError("Unable to acquire database connection")
            
            threading.Event().wait(0.1)  # 短暂等待后重试
        
        try:
            yield self._local.connection
        except Exception as e:
            try:
                if hasattr(self._local, 'connection'):
                    self._local.connection.close()
                    del self._local.connection
            except Exception:
                pass
            raise

    def _safe_cache_operation(self, operation: Callable, timeout: float = 1.0):
        """安全地执行缓存操作，带超时机制"""
        if self._cache_lock.acquire(timeout=timeout):
            try:
                return operation()
            finally:
                self._cache_lock.release()
        raise TimeoutError("Unable to acquire cache lock")

    def cache(self, ttl: Optional[int] = None, qps: Optional[int] = None, period: int = 1) -> Callable:
        actual_ttl = ttl if ttl is not None else self._default_ttl

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            rate_limited_func = func
            if qps is not None:
                rate_limited_func = sleep_and_retry(limits(calls=qps, period=period)(func))
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if self._stop_event.is_set() or self._is_shutdown:
                    raise RuntimeError("Cache instance has been shut down")
                    
                function_name = func.__name__
                args_hash = self._compute_args_hash(*args, **kwargs)

                # 检查内存缓存
                try:
                    def get_cache():
                        return self._cache.get(args_hash)
                    result = self._safe_cache_operation(get_cache)
                    if result is not None:
                        return result
                except TimeoutError:
                    # 如果无法获取缓存锁，继续检查数据库
                    pass

                # 检查数据库缓存
                try:
                    with self._get_connection() as conn:
                        db_result = conn.execute("""
                            SELECT result
                            FROM function_cache
                            WHERE function_name = ?
                            AND args_hash = ?
                            AND expires_at > ?
                        """, (function_name, args_hash, datetime.now())).fetchone()

                        if db_result:
                            result = self._deserialize_result(db_result[0])
                            try:
                                def update_cache():
                                    self._cache[args_hash] = result
                                self._safe_cache_operation(update_cache)
                            except TimeoutError:
                                pass  # 即使更新内存缓存失败也返回结果
                            return result
                except Exception:
                    # 如果数据库访问失败，继续执行原始函数
                    pass

                # 执行原始函数
                result = rate_limited_func(*args, **kwargs)

                # 更新缓存
                try:
                    def update_cache():
                        self._cache[args_hash] = result
                    self._safe_cache_operation(update_cache)
                except TimeoutError:
                    pass  # 继续尝试更新数据库缓存

                try:
                    self._write_queue.put_nowait((function_name, args_hash, result, actual_ttl))
                except queue.Full:
                    pass  # 队列满时丢弃更新操作

                return result
            
            return wrapper
        
        return decorator

    def _process_write_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._write_queue.get(timeout=1)
                if task is None:
                    break
                    
                function_name, args_hash, result, ttl = task
                
                try:
                    with self._get_connection(timeout=5) as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO function_cache
                            (function_name, args_hash, result, created_at, expires_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            function_name,
                            args_hash,
                            self._serialize_result(result),
                            datetime.now(),
                            datetime.now() + timedelta(seconds=ttl)
                        ))
                        conn.commit()
                except Exception:
                    continue  # 写入失败时继续处理下一个任务
                    
            except queue.Empty:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    continue

    def _periodic_cleanup(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._cleanup_expired_cache()
                for _ in range(3600):
                    if self._stop_event.is_set():
                        break
                    threading.Event().wait(1)
            except Exception:
                if not self._stop_event.is_set():
                    continue

    def _cleanup_expired_cache(self) -> None:
        try:
            with self._get_connection(timeout=5) as conn:
                conn.execute("""
                    DELETE FROM function_cache 
                    WHERE expires_at < ?
                """, (datetime.now(),))
                conn.commit()
        except Exception:
            pass

    def clear_cache(self, function_name: Optional[str] = None) -> None:
        """清除缓存，分两步进行以避免死锁"""
        # 第一步：清除内存缓存
        try:
            def clear_mem_cache():
                self._cache.clear()
            self._safe_cache_operation(clear_mem_cache)
        except TimeoutError:
            pass  # 继续清除数据库缓存

        # 第二步：清除数据库缓存
        try:
            with self._get_connection(timeout=5) as conn:
                if function_name:
                    conn.execute(
                        "DELETE FROM function_cache WHERE function_name = ?",
                        (function_name,)
                    )
                else:
                    conn.execute("DELETE FROM function_cache")
                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to clear database cache: {e}")

    def shutdown(self) -> None:
        """安全关闭缓存实例"""
        with self._shutdown_lock:
            if self._is_shutdown:
                return
                
            self._is_shutdown = True
            self._stop_event.set()
            
            try:
                self._write_queue.put_nowait(None)
            except queue.Full:
                pass

            if hasattr(self, '_writer_thread'):
                try:
                    self._writer_thread.join(timeout=5)
                except Exception:
                    pass

            if hasattr(self, '_cleaner_thread'):
                try:
                    self._cleaner_thread.join(timeout=5)
                except Exception:
                    pass

            if hasattr(self._local, 'connection'):
                try:
                    self._local.connection.close()
                    del self._local.connection
                except Exception:
                    pass

    def __del__(self) -> None:
        self.shutdown()