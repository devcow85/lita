import os
import re
import json

from .generator import LITAGenerator
from .metrics import TimeMetric

def run_tests(model, tokenizer, use_kv_cache, input_text, max_length=50, num_tests=10, enable_profile = False):
    # Prepare module
    metrics = TimeMetric()
    generator = LITAGenerator(model, tokenizer, use_kv_cache)
    base_dir = f"profile_result/{os.path.basename(model.name_or_path)}"
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Warm-up
    print("Warm-up")
    for _ in range(10):
        generator.generate(input_text, max_length)

    # Testing
    for i in range(num_tests):
        generated_text, time_dict, prof = generator.generate(input_text, max_length, enable_profile = enable_profile)
        metrics.record(time_dict)
        print(f"Generated Text: {generated_text}")
        
        if enable_profile:
            prof.export_chrome_trace(f"profile_result/{os.path.basename(model.name_or_path)}/trace_{i}.json")

    # Print recorded metrics stats
    metrics.print_statistics()
    
    return prof

def load_json(file_name):
    with open(file_name, 'rb') as f:
        trace_raw_bytes = f.read()

    cleaned_bytes = re.sub(b'[^\x00-\x7F]+', b'', trace_raw_bytes)
    cleaned_text = cleaned_bytes.decode('utf-8', errors='ignore')
    return json.loads(cleaned_text)

class ProfileAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trace_json = load_json(file_path)
        
    def get_index(self, index):
        return self.trace_json['traceEvents'][index]
    
def multiple_profile_analyzer(base_path, index_json):
    
    tree_structure = load_json(index_json)
    
    