from lita import Lita
from lita.utils import set_seed

import torch

# This code is used to compare the output of an ONNX model and a PyTorch model (torch),
# ensuring that both produce the same results when generating text from a given prompt.

# List of models to test: ONNX and PyTorch
mode = ['onnx', 'torch']  # 'onnx' refers to the ONNX format model, 'torch' refers to the PyTorch model
device = 'cuda'  # The device to run the models on (in this case, GPU with 'cuda')

# Paths to the models: ONNX model path and PyTorch model identifier
model_path = ["data/gpt2", "gpt2"]  # First is the ONNX model path, second is the Hugging Face model name for PyTorch
cache_dir = [None, "/data"]  # Cache directory for models (None for ONNX, set directory for PyTorch)

# Loop over both models (ONNX and PyTorch) and test their output
for idx, (m, mp, cd) in enumerate(zip(mode, model_path, cache_dir)):
    set_seed(42)  # Set seed for reproducibility, ensuring the same results across runs

    # Print the model type and its corresponding path and cache directory
    print(m, mp, cd)

    # Initialize the Lita class with the model type and device
    lita = Lita(m, device)

    # Load the model (ONNX or PyTorch) from the specified path
    lita._load_model(mp, cd)

    # Optional: Convert PyTorch model to fp16 if using PyTorch, commented out for now
    # if m != 'onnx':
    #     lita.model.to(torch.float16)  # Use fp16 (half precision) for PyTorch model

    # Optional: Register performance tracking, commented out for now
    # lita._register_perf()

    # Test text generation for each model and print the results
    print(f"[{idx}] {m} test ... ")  # Indicate which model is being tested
    output_text = lita.generate('once upon a time ')  # Generate text with the prompt 'once upon a time'
    
    # Print the generated text
    print(output_text[0])  # Output the first generated text result
