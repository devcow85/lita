from torch.profiler import _ExperimentalConfig, ProfilerActivity, profile
import os
import json


ORT_PROFILER = 1
TORCH_PROFILER = 2

class UniversialProfiler:
    def __init__(self, mode, **kwargs):
        print(mode)
        self.mode = ORT_PROFILER if mode == 'ort' else TORCH_PROFILER
        
        if self.mode == TORCH_PROFILER:
            self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    record_shapes=True,
                                    with_stack=True,
                                    with_modules=True,
                                    experimental_config=_ExperimentalConfig(verbose=True))
        else:
            self.profiler = ORTProfiler(session=kwargs.get("session", None))
        print("profiler init")

    def __enter__(self):
        return self.profiler.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        print("profiler done")
        
        
class ORTProfiler:
    def __init__(self, session):
        if session is None:
            raise ValueError("No session")

        self.session = session
        
        self.providers = session.get_providers()
        _provider_options = session.get_provider_options()
        self.provider_options = [_provider_options[p] for p in self.providers]
                
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        file_path = self.session.end_profiling()
        print(file_path)
        
        with open(file_path, "r") as f:
            self.result = json.load(f)
        
        clear_profile_data(os.environ.get("LITA_PROFILE_DIR"))
        
        sess_options = self.session.get_session_options()
        sess_options.enable_profiling = True
        
        self.session._sess_options = sess_options
        self.session._reset_session(self.providers, self.provider_options)
        
    
def clear_profile_data(path):
    
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"delete {file_path}")
            
    return

