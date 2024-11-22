from functools import wraps
import time

func_stats_global = {}  # 全局变量，用于存储所有函数的统计信息

def timer(func):
    """
    装饰器函数，用于记录函数的运行次数、累计总运行时间，并计算平均运行时间。
    将统计信息存储在全局变量 func_stats_global 中。

    Args:
        func: 被装饰的函数。

    Returns:
        wrapper: 装饰后的函数。
    """
    if func.__name__ not in func_stats_global:
        func_stats_global[func.__name__] = {
            'count': 0,
            'total_time': 0
        }

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        func_stats_global[func.__name__]['count'] += 1
        func_stats_global[func.__name__]['total_time'] += execution_time
        return result

    return wrapper

def print_stats():
    """
    打印所有被 timer 装饰过的函数的统计信息。
    """
    if not func_stats_global:
        print("没有函数被 timer 装饰过。")
        return

    for func_name, stats in func_stats_global.items():
        avg_time = stats['total_time'] / stats['count'] if stats['count'] else 0
        print(f"函数 {func_name}:")
        print(f"  调用次数: {stats['count']}")
        print(f"  总运行时间: {stats['total_time']:.5f} 秒")
        print(f"  平均运行时间: {avg_time:.5f} 秒")

