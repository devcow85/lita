import os
from lita import optimum_export

# Model list
model_name = "gpt2"
# model_name = "meta-llama/Llama-2-7b-hf"
# "EleutherAI/gpt-neox-20b",
# "meta-llama/Llama-2-7b-hf",
# "bigscience/bloom-7b1",
# "tiiuae/falcon-7b",

# Output dir
output_dir = f"data/{os.path.basename(model_name)}"

# Convert with optimum-cli
optimum_export(
    model=model_name,
    output=output_dir,
    device="cuda",
    opset=14,
    atol=1e-5,
    cache_dir="/data",  # transformers model cache location
    optimize="O1",
    fp16=True
)
