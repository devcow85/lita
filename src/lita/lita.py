from transformers import AutoTokenizer,AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

from functools import wraps


class Lita:
    def __init__(self, mode = 'torch', device = 'cuda'):
        self.device = device
        self.mode = mode
        
        self.metrics = {'result':[]}
        
        self.model = None
        self.tokenizer = None
    
    def wrap_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.metrics['result'].append(result)
            return result
        return wrapper
    
    def _load_model(self, model_path, cache_dir):
        if self.mode == 'torch':
            model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir = cache_dir).to(self.device)
            print(f"assign model to {self.device} device")
            model.eval()
        else:
            if self.device.__contains__('cuda'):
                provider = "CUDAExecutionProvider"
                print("assign model to cuda device")
            else:
                provider = "CPUExecutionProvider"
                print("assign model to cpu device")
            model = ORTModelForCausalLM.from_pretrained(model_path, provider = provider)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = model
        self.tokenizer = tokenizer
    
    def _register_perf(self):
        if self.model is not None:
            self.model.forward = self.wrap_function(self.model.forward)
            print("Register perf function Complete")
        
    def generate(self, input_text, return_logits = False):
        print(f'generation mode {self.mode}')
        if self.tokenizer is not None and self.model is not None:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                do_sample = True,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50,  
                num_return_sequences=1,
                temperature = 0.8,
                top_p = 0.9,
                repetition_penalty=1.0
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return_value = (generated_text, )
        
        if return_logits:
            return_value + (outputs, )
            
        return return_value
