from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import format_exc
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from functools import partial
from core.utils.log import logger

@dataclass
class ParallelTask:
    func: Callable
    name: str
    args: tuple = ()
    kwargs: dict = None
    timeout: int = 60  # 添加超时设置
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

class ParallelExecutor:
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers
    
    def execute(self, tasks: List[ParallelTask]) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(task.func, *task.args, **task.kwargs): task
                for task in tasks
            }
            
            results = {}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[task.name] = future.result(timeout=task.timeout)
                except TimeoutError:
                    logger.error(f"Task {task.name} timed out after {task.timeout} seconds")
                    results[task.name] = self._get_default_result()
                except Exception as e:
                    logger.error(f"Error in task {task.name}: {str(e)}\n{format_exc()}")
                    results[task.name] = self._get_default_result()
            
        return results

    def _get_default_result(self):
        return {}

def create_task(func: Callable, 
                args: Union[tuple, list] = None,
                kwargs: dict = None,
                name: str = None) -> ParallelTask:
    """
    创建并行任务的辅助函数
    :param func: 要执行的函数
    :param args: 位置参数
    :param kwargs: 关键字参数
    :param name: 任务名称，默认使用函数名
    :return: ParallelTask对象
    """
    return ParallelTask(
        func=func,
        name=name or func.__name__,
        args=tuple(args) if args is not None else (),
        kwargs=kwargs or {}
    )