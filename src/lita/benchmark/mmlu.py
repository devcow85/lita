import numpy as np
from datasets import load_dataset

from lita.benchmark.commons import BenchmarkQAataLoader, mmlu_prompt_generator, mmlupro_prompt_generator


class MMLUDataLoader(BenchmarkQAataLoader):
    prefix_prompt = 'The following are multiple choice questions (with answers) about {}.\n\n'
    
    def __init__(self, dataset = "hails/mmlu_no_train", prompt_generator=mmlu_prompt_generator, enable_prefix_promts = True, n_shots = 0):
        self.test_data = load_dataset(dataset, 'all', trust_remote_code = True)['test']
        self.ref_data = load_dataset(dataset, 'all', trust_remote_code = True)['dev']

        self.choices = self.test_data.features['answer']._int2str
        self.enable_prefix_promts = enable_prefix_promts
        self.task = np.unique(self.ref_data['subject'])
        super().__init__(prompt_generator, n_shots)
    
class MMLUProDataLoader(BenchmarkQAataLoader):
    prefix_prompt = 'The following are multiple choice questions (with answers) about {}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
    
    def __init__(self, dataset = "TIGER-Lab/MMLU-Pro", prompt_generator=mmlupro_prompt_generator, enable_prefix_promts = True, n_shots = 0):
        self.test_data = load_dataset(dataset)['test']
        self.ref_data = load_dataset(dataset)['validation']
        
        self.choices = [chr(65 + i) for i in range(10)] # A~J
        self.enable_prefix_promts = enable_prefix_promts
        self.task = np.unique(self.ref_data['category'])
        super().__init__(prompt_generator, n_shots)
        
