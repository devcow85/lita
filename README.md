# lita
LLM Integrated Testing &amp; Analysis Framework

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