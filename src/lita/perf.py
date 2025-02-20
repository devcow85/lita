from functools import wraps
import time

from vllm.profiler import layerwise_profile


def perf_time(func, metric):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        metric.append(time.time() - start_time)
        print(metric)
        return result
    return wrapper

def perf_vllmprof(func, metric):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with layerwise_profile() as prof:
            result = func(*args, **kwargs)
        print(len(prof.profiler.function_events))
        return result
    return wrapper