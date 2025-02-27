import os
import re

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlSystemGetDriverVersion, \
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetCudaComputeCapability, nvmlDeviceGetCount
    nvml_available = True
except ImportError:
    nvml_available = False
    

def repr2dict(s):
    pattern = r'(\w+)=([^,]+?)(?=(,|\)$))'
    matches = re.findall(pattern, s)
    return {key: eval(value) if value not in ["None", "False", "True"] else value for key, value, _ in matches}

def get_gpu_info():
    """pynvml을 이용한 GPU 정보 수집"""
    if not nvml_available:
        return None

    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        gpus = []
        for i in range(gpu_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpus.append({
                "gpu_index": i,
                "gpu_model": nvmlDeviceGetName(handle),
                "cuda_compute_capability": f"{nvmlDeviceGetCudaComputeCapability(handle)[0]}.{nvmlDeviceGetCudaComputeCapability(handle)[1]}",
                "driver_version": nvmlSystemGetDriverVersion(),
                "total_memory": f"{nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3):.2f} GB",
            })
        return gpus
    except Exception as e:
        return {"error": str(e)}

def get_system_info():
    """OS, CPU, 메모리, 네트워크, 프로세스 및 GPU 정보를 포함한 시스템 정보 반환"""
    system_info = {
        "os_name": os.uname().sysname,
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "kernel_version": os.uname().release if hasattr(os, "uname") else None,
        "architecture": os.uname().machine if hasattr(os, "uname") else None,

        # CPU 정보
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,

        # 메모리 정보 (Linux/macOS에서만 가능)
        "memory_page_size": os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None,
        "total_memory": (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) if hasattr(os, "sysconf") else None,

        # GPU 정보
        "gpu_info": get_gpu_info()
    }
    return system_info