from functools import wraps
import time

import numpy as np
from vllm.profiler import layerwise_profile

class PerfMetric:
    time_factors = {"s": 1,
                    "ms": 1000,       # 1 second = 1000 milliseconds
                    "us": 1_000_000,  # 1 second = 1,000,000 microseconds
                    "minute": 1 / 60, # 1 second = 1/60 minutes
                    "hour": 1 / 3600  # 1 second = 1/3600 hours
                    }
    latency = []
    
    def __init__(self, unit='ms'):
        self.unit = unit
        self.reset()
        
    def __call__(self, m):
        self.latency.append(m)
        
    def __len__(self):
        return len(self.latency)
    
    def summary(self):
        latency_array = np.array(self.latency)
        
        if not self.latency:
            print("No Data")
            return {"e2e": None, 
                    "ttft": None, 
                    "tbt": None,
                    "p50": None,
                    "p99": None,
                    "throughput": None,
                    "token_length": None, 
                    "time_unit": self.unit} 
        elif len(self.latency)<2:
            print("1 token generated")
            return {"e2e": latency_array[0], 
                    "ttft": latency_array[0], 
                    "tbt": None,
                    "p50": None,
                    "p99": None,
                    "throughput": None,
                    "token_length": len(latency_array), 
                    "time_unit": self.unit} 
        
        return {
            "e2e": latency_array.sum(),
            "ttft": latency_array[0],
            "tbt": sum(latency_array[1:]) / (len(latency_array) - 1),
            "p50": np.percentile(latency_array, 50),
            "p99": np.percentile(latency_array, 99),
            "throughput": len(latency_array) / latency_array.sum()*self.time_factors[self.unit] if latency_array.sum() > 0 else None,
            "token_length": len(latency_array),
            "time_unit": self.unit} 
        
    def reset(self):
        self.latency = []


def perf_time(func, metric):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        dur = time.time() - start_time
        metric(dur*1e3)    # ms
        
        return result
    return wrapper

def perf_vllmprof(func, metric):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with layerwise_profile() as prof:
            result = func(*args, **kwargs)
    
        metric(prof.profiler.self_cpu_time_total/1e3)    # ms    
        return result
    return wrapper