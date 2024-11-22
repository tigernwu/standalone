import asyncio
from typing import List, Dict, Any, Callable, TypeVar, Generic

T = TypeVar('T')
R = TypeVar('R')

class QueryExecutor(Generic[T, R]):
    def __init__(self, query_function: Callable[[T], R]):
        self.query_function = query_function

    def sequential_query(self, items: List[T]) -> List[R]:
        return [self.query_function(item) for item in items]

    async def concurrent_query(self, items: List[T]) -> List[R]:
        async def query_item(item: T) -> R:
            return await asyncio.to_thread(self.query_function, item)

        tasks = [query_item(item) for item in items]
        return await asyncio.gather(*tasks)