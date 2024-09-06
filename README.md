# lita
LITA is a comprehensive tool designed to analyze the inference performance of large language models (LLMs)

## Installation

### Docker Setup
1. **Clone the Repository**:
```bash
    git clone -b feat/docker https://github.com/devcow85/lita.git
    cd lita
```
2. **Build the Docker Image**:
```bash
    docker-compose -f docker/docker-compose.yml up -d --build
```

3. **Access the Container**:
```bash
    docker exec -it lita-container /bin/bash
```
### Manual Installation (cuda12)
1. **Clone the Repository and Install**:
```bash
    git clone -b feat/docker https://github.com/devcow85/lita.git
    cd lita
    pip install .
```
2. **Install ONNX Runtime and Optimum**:
```bash
    pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]
    pip uninstall -y onnxruntime-gpu
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    pip install torchvision
```

## Usage

### ONNX Model Conversion
Lita supports easy conversion of models from PyTorch to ONNX for optimized inference. The following example demonstrates how to convert a model to ONNX format using Lita's `optimum_export` function.
```python
    import os
    from lita import optimum_export

    # Specify the model name (from Hugging Face) and the output directory
    model_name = "gpt2"
    output_dir = f"data/{os.path.basename(model_name)}"

    # Convert the model to ONNX format
    optimum_export(
        model=model_name,        # Model name or local path
        output=output_dir,       # Directory where the ONNX model will be saved
        atol=1e-5,               # Absolute tolerance level for validation
        cache_dir="/data",       # Cache directory for transformers model
    )
```
This code will convert the `gpt2` model into ONNX format and store it in the `data/gpt2` directory.

### Text Generation with ONNX and Pytorch
Lita can generate text from both ONNX and PyTorch models, allowing you to compare performance and results between the two frameworks.
```python
    from lita import Lita
    from lita.utils import set_seed

    # List of models: ONNX and PyTorch
    modes = ['onnx', 'torch']
    device = 'cuda'  # Use GPU if available
    model_paths = ["data/gpt2", "gpt2"]  # ONNX model path and PyTorch model identifier
    cache_dirs = [None, "/data"]  # Cache directory for PyTorch models

    # Loop through both models (ONNX and PyTorch) to generate text and compare
    for idx, (mode, model_path, cache_dir) in enumerate(zip(modes, model_paths, cache_dirs)):
        # Set random seed for reproducibility
        set_seed(42)

        print(f"Running {mode} model from {model_path}...")

        # Initialize the Lita class for the given model
        lita = Lita(mode, device)
        
        # Load the model (ONNX or PyTorch)
        lita._load_model(model_path, cache_dir)
        
        # Generate text using a given prompt
        output_text = lita.generate('Once upon a time')
        print(f"Generated text from {mode}: {output_text[0]}")
```

Example output
```bash
    onnx data/gpt2 None
    assign model to cuda device

    [0] onnx test ... 
    generation mode onnx
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    once upon a time?"


    "Yes," said Hermione. "It would be an adventure."


    "But I am a bit of a wizard, and I am not always the most sensible of wizards," said Dumbledore. "You see

    torch gpt2 /data
    assign model to cuda device

    [1] torch test ... 
    generation mode torch
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    once upon a time?"


    "Yes," said Hermione. "It would be an adventure."


    "But I am a bit of a wizard, and I am not always the most sensible of wizards," said Dumbledore. "You see
```

This example shows how Lita can load both ONNX and PyTorch models, and then generate text using the prompt "Once upon a time." You can compare the outputs from both models to ensure they are similar or identical.

