# LITA
### LLM Integrated Testing &amp; Analysis Framework
Lita is a **comprehensive testing and analysis framework for Large Language Models (LLMs)**, designed to provide an integrated environment for efficient execution, benchmarking, and profiling.

### Key Features:
- Multi-Framework Support: Run models on various execution backends, including vLLM, Hugging Face (HF), and ONNX Runtime (ORT).
- Performance Profiling: Collect detailed execution metrics using built-in profilers like vllmprof.

Lita is designed to help researchers and developers evaluate and optimize LLM workloads across different execution environments, enabling seamless integration into existing machine learning workflows.

## Installation (Development)
To set up the development environment, use the following commands:
```bash
pip install -e .
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.19
```
Ensure that PyTorch version `2.6.0` or `later` is installed.

## Setting Cache Directory
Before running the Lita framework, you can configure the cache directory by setting environment variables.
Add the following lines to your `.bashrc` (or `.bash_profile` for macOS):
```bash
export LITA_CACHE="your_cache_path"
export HF_HOME=$LITA_CACHE
```
Apply the changes by running:
```bash
source ~/.bashrc  # or source ~/.bash_profile
```
Now, Lita will use the specified cache directory.

## Usage
1. Running Models with Different Frameworks
Lita supports executing models on various frameworks such as vLLM, Hugging Face (HF), and ONNX Runtime (ORT).
    ```python
    from lita import Lita
    import time

    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    for mode in ["vllm", "hf", "ort"]:
        mm = Lita(model_name, mode)
        output_str = mm.generate('hello')
        print(f"Generation output: {output_str}")
    ```

2. Performance Measurement
Lita provides built-in performance profiling using vLLM Profiler (vllmprof).
    ```python
    from lita import Lita
    import time

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    mode = "hf"

    mm = Lita(model_name, mode, perf='vllmprof')
    output_str = mm.generate('hello')

    print(output_str)
    print(mm.metric.summary())
    ```
    This script runs the model in hf mode while enabling performance profiling with vllmprof. After text generation, it prints the output along with detailed performance metrics.
