from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from vllm import LLM, SamplingParams

import os

from lita import perf

def load_model(mode, model, dtype, max_model_len=4096, seed=7, device="cuda", nperf=None):
    # Load model & tokenizer
    if mode =="vllm":
        model_ = LLM(model=model, 
                     tokenizer=model, 
                     seed=seed,
                     max_model_len=max_model_len,
                     dtype=dtype,
                     device=device)
        tokenizer_ = None
    elif mode =="hf":
        model_ = AutoModelForCausalLM.from_pretrained(model).to(device)
        tokenizer_ = AutoTokenizer.from_pretrained(model)
        
        tokenizer_.pad_token = tokenizer_.eos_token
        
    elif mode =="ort":
        lita_cache = os.environ.get("LITA_CACHE")
        onnx_cahce = os.path.join(lita_cache, 'onnx')
        model_path = os.path.join(onnx_cahce, model)
        
        if not os.path.exists(model_path):
            ort_model = ORTModelForCausalLM.from_pretrained(model, export=True, use_io_binding = True)
            ort_model.save_pretrained(model_path)
            print(f"Convert {model} to ONNX model and Save to {model_path}")
        
        model_ = ORTModelForCausalLM.from_pretrained(model_path, use_io_binding = True).to(device)
        tokenizer_ = AutoTokenizer.from_pretrained(model_path)
        
        tokenizer_.pad_token = tokenizer_.eos_token
        
    else:
        raise ValueError("Unsupported mode. Choose 'hf', 'onnx', or 'vllm'.")
    
    # Performance wrapper
    metric = None
    if nperf is not None:
        metric = perf.PerfMetric()
        wrapper = getattr(perf, f"perf_{nperf}")
        if mode =="vllm":
            model_.llm_engine.step = wrapper(model_.llm_engine.step, metric)
        else:
            model_.forward = wrapper(model_.forward, metric)
            
    return model_, tokenizer_, metric
    
def parameter_generator(mode, input_text, seed=7, max_new_tokens=30, top_k=1, temperature=1.0):
    if mode =="vllm":
        kwargs = {
            "prompts": input_text,
            "sampling_params": SamplingParams(
                temperature=temperature,
                top_k=top_k,
                max_tokens=max_new_tokens,
                seed=seed,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                ignore_eos=True
            ),
            "use_tqdm": False
        }
    elif mode in ["hf", "ort"]:
        kwargs = {
            "input_ids": input_text.get("input_ids"),
            "attention_mask": input_text.get("attention_mask"),
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "temperature": temperature,
            "use_cache":True
        }
    else:
        raise ValueError("Unsupported mode. Choose 'hf', 'onnx', or 'vllm'.")
    
    return kwargs