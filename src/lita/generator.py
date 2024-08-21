import time
import warnings

import torch
from torch.profiler import profile, ProfilerActivity

class DummyProfile:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def step(self):
        pass
    
class LITAGenerator:
    def __init__(self, model, tokenizer, use_kv_cache=True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_kv_cache = use_kv_cache

    def generate(self, input_text, max_length=50, enable_profile=False):
        model_device = next(self.model.parameters()).device

        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # 입력 텐서의 device와 모델의 device 비교 및 맞춤
        input_device = input_ids.device
        if input_device != model_device:
            warnings.warn(f"Input tensor is on {input_device}, but model is on {model_device}. Moving input to {model_device}.")
            input_ids = input_ids.to(model_device)
            
        output = input_ids
        output_tot = input_ids
        
        past_key_values = None

        time_dict = {"TTFT":0, "TBT":[]}
        
        profiler = DummyProfile()
        if enable_profile:
            # tracing_schedule = schedule(skip_first=5, wait=5, warmup=2, active=2, repeat=1)
            profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                           with_stack=True, 
                           record_shapes=True, 
                           profile_memory=True, 
                           with_flops=True, 
                           with_modules=True)            
        
        with profiler as prof:
            for i in range(max_length):
                with torch.no_grad():
                    start_time = time.time()
                    
                    if prof is not None: prof.step()
                    outputs = self.model(output, past_key_values=past_key_values, use_cache=self.use_kv_cache)
                    
                    token_time = time.time()-start_time
                    
                    if i == 0:
                        time_dict["TTFT"] = token_time
                    else:
                        time_dict["TBT"].append(token_time)
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    if self.use_kv_cache:
                        past_key_values = outputs.past_key_values

                    # Greedy decoding
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    output_tot = torch.cat((output_tot, next_token_id), dim=1)
                    
                    if self.use_kv_cache:
                        output = next_token_id
                    else:
                        output = output_tot

                    # break generation loop when EOS token detected
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                
        
        time_dict["E2E"] = time_dict["TTFT"] + sum(time_dict["TBT"])
        time_dict["NUM_TOKEN"] = 1 + len(time_dict["TBT"])
        
        generated_text = self.tokenizer.decode(output_tot[0], skip_special_tokens=True)
        return generated_text, time_dict, prof