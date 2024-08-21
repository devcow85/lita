import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig

import pandas as pd

from lita import run_tests
from lita.example_prompts import INPUT_TEXT_50, INPUT_TEXT_100, INPUT_TEXT_500

def main():

    use_kv_cache = True
    input_texts = [INPUT_TEXT_50, INPUT_TEXT_100, INPUT_TEXT_500]
    device_name = "A100"
    
    stat_result = []
    for input_text in input_texts:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model_names = [
                        "EleutherAI/gpt-neox-20b",
                        "meta-llama/Llama-2-7b-hf",
                        "bigscience/bloom-7b1",
                        "tiiuae/falcon-7b",
                        "facebook/opt-6.7b"
                        ]
        
        cache_dir = "/project/performance-check/hub"

        
        for model_name in model_names:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            config = AutoConfig.from_pretrained(model_name,cache_dir=cache_dir)
            if model_name in ["EleutherAI/gpt-neox-20b", "meta-llama/Llama-2-7b-hf"]:
                print("eager mode enable")
                config._attn_implementation = "eager"
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
            model.to(torch.float16)
            model.to(device)
            
            stats = run_tests(model, tokenizer, use_kv_cache, input_text, max_length=20, num_tests=100, enable_profile = False)
            stats["model"] = model_name
            stats["device"] = device_name
            stats["input_text_length"] = len(input_text)
                        
            stat_result.append(stats)
            
    df = pd.DataFrame(stat_result)
    df.to_csv(f"overall_perfomance_{device_name}.csv")

if __name__ == "__main__":
    main()