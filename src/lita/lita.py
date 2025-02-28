import gc
import torch
from lita.model import load_model, parameter_generator
from lita.utils import repr2dict


class Lita:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            cls._cleanup_prev_instance()
        
        instance = super().__new__(cls)
        cls._instance = instance
        return instance
    
    @classmethod
    def _cleanup_prev_instance(cls):
        cls._instance._cleanup()
        cls._instance = None
        
    def _cleanup(self):
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def __init__(self, model, mode, seed=7, dtype = torch.float16, device="cuda", perf=None):
        self.mode = mode
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.generation_configs = None
        
        self.model, self.tokenizer, self.metric = load_model(mode, model, seed=seed, dtype=dtype, device=device, nperf=perf)
            
    def generate(self, input_text, max_new_tokens=30, top_k=1, temperature=1.0):
        if self.metric is not None:
            self.metric.reset()
        inputs = input_text if self.mode=="vllm" else self.tokenizer.batch_encode_plus(input_text, return_tensors="pt", padding="longest", truncation=True).to(self.device)
        
        self.generation_configs = parameter_generator(self.mode, 
                                                      inputs, 
                                                      self.seed, 
                                                      max_new_tokens,
                                                      top_k,
                                                      temperature)
        
        with torch.no_grad():
            output = self.model.generate(**self.generation_configs)
        
        return [r.prompt + r.outputs[0].text for r in output] if self.mode=="vllm" else self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    def get_configs(self):
        
        if self.mode == 'vllm':
            return {**repr2dict(self.generation_configs['sampling_params'].__repr__()),
                    "dtype": str(self.model.llm_engine.model_config.dtype),
                    "device": str(self.model.llm_engine.device_config.device)}
        else:
            cfg = self.generation_configs.copy()
            cfg.pop("input_ids", None)
            cfg.pop("attention_mask", None)

            input_text = self.tokenizer.decode(self.generation_configs['input_ids'][0].detach().cpu().numpy(), skip_special_tokens=True)
            return {"input_text": input_text,
                    **cfg, 
                    "dtype":str(self.model.dtype),
                    "device":str(self.model.device)}