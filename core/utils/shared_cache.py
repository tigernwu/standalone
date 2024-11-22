

from core.utils.duck_cache import DBCache, RateLimiter

cache = DBCache()
lg_limiter = RateLimiter(name='lg', rate=1)
bg_limiter = RateLimiter(name='bg', rate=1)
cd_limiter = RateLimiter(name='cd', rate=1)
sn_limiter = RateLimiter(name='sn', rate=1)

