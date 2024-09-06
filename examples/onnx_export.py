import os
from lita import optimum_export

# Model list
model_name = "gpt2"
# Example models that can be used:
# "EleutherAI/gpt-neox-20b",
# "meta-llama/Llama-2-7b-hf",
# "bigscience/bloom-7b1",
# "tiiuae/falcon-7b",
# "facebook/opt-6.7b"

# Directory to save the exported ONNX model
output_dir = f"data/{os.path.basename(model_name)}"

# Convert the model to ONNX format using optimum-cli
optimum_export(
    model=model_name,  # Name of the model to export (from Hugging Face or local path)
    output=output_dir,  # Directory to store the ONNX exported model
    # device="cuda",  # Uncomment to specify the device (e.g., GPU 'cuda' or CPU 'cpu')
    # opset=14,  # Uncomment to specify a particular ONNX opset version
    atol=1e-5,  # Absolute tolerance for model conversion validation
    cache_dir="/data",  # Directory to cache the downloaded transformers model
    # optimize=0,  # Uncomment to enable ONNX model optimization (e.g., 0, 1, 2, 99)
    # fp16 = True  # Uncomment to enable FP16 (half precision) optimization for inference
)
