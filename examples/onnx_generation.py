from lita import Lita
from lita.utils import set_seed

import torch

set_seed(7)

# onnx model
mode = ['torch', 'onnx']
device = 'cuda'
model_path = ["gpt2", "data/gpt2"]
cache_dir = ["/data", None]

for idx, (m, mp, cd) in enumerate(zip(mode, model_path, cache_dir)):
    print(m, mp, cd)
    lita = Lita(m, device)
    lita._load_model(mp, cd)
    if m != 'onnx':
        lita.model.to(torch.float16)
    # lita._register_perf()

    print(f"[{idx}] {m} test ... ")
    output_text = lita.generate('once upon a time ')
    print(output_text[0])