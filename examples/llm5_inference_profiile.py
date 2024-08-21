import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig

from lita import run_tests
from lita.example_prompts import INPUT_TEXT_50

def main():

    use_kv_cache = True
    input_text = INPUT_TEXT_50
    
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_names = [
                    "EleutherAI/gpt-neox-20b",
                    "meta-llama/Llama-2-7b-hf",
                    "bigscience/bloom-7b1",
                    "tiiuae/falcon-7b",
                    "facebook/opt-6.7b"]
    
    cache_dir = "/root/00.hosts/04.llm-model/hub"

    for model_name in model_names:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        config = AutoConfig.from_pretrained(model_name,cache_dir=cache_dir)
        if model_names in ["EleutherAI/gpt-neox-20b", "meta-llama/Llama-2-7b-hf"]:
            config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        model.to(torch.float16)
        model.to(device)
        
        prof = run_tests(model, tokenizer, use_kv_cache, input_text, max_length=5, num_tests=20, enable_profile = True)

if __name__ == "__main__":
    main()