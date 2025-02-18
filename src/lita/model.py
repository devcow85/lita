from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from vllm import LLM, SamplingParams


def load_model(mode, model, dtype, max_model_len=4096, seed=7, device="cuda"):
    if mode =="vllm":
        return LLM(model=model, tokenizer=model, seed=seed, max_model_len=max_model_len, dtype=dtype, device=device), None
    elif mode =="hf":
        return AutoModelForCausalLM.from_pretrained(model).to(device), AutoTokenizer.from_pretrained(model)
    elif mode =="ort":
        # TODO: Add cache saving routine 
        return ORTModelForCausalLM.from_pretrained(model, export=True, use_io_binding = True).to(device), AutoTokenizer.from_pretrained(model)
    else:
        raise ValueError("Unsupported mode. Choose 'hf', 'onnx', or 'vllm'.")
    
def parameter_generator(mode, input_text, seed=7, max_new_tokens=30, top_k=1, temperature=1.0):
    if mode =="vllm":
        kwargs = {
            "prompts": input_text,
            "sampling_params": SamplingParams(
                temperature=temperature,
                top_k=top_k,
                max_tokens=max_new_tokens,
                seed=seed,
                skip_special_tokens=False,
                spaces_between_special_tokens=False
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
            "temperature": temperature
        }
    else:
        raise ValueError("Unsupported mode. Choose 'hf', 'onnx', or 'vllm'.")
    
    return kwargs