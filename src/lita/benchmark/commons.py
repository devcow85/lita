import re
from tqdm import tqdm
import pandas as pd
import numpy as np


def mmlupro_prompt_generator(data, choices, last = True):
    prompt = "Question:\n"
    prompt += data['question'] + "\n"
    prompt += "Options:\n"

    for i in range(len(data['options'])):
        prompt += "{}. {}\n".format(choices[i], data['options'][i])
    if last:
        prompt += "Answer: Let's think step by step."
    else:
        cot_content = data["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
        
    return prompt

def extract_choice(text):
    return text[-1]

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
    
def mmlu_prompt_generator(data, choices, last = True):
    prompt = data['question']
    for i in range(len(choices)):
        prompt += "\n{}. {}".format(choices[i], data['choices'][i])
    prompt += "\nAnswer:"
    if not last:
        prompt += " {}\n\n".format(choices[data['answer']])
        
    return prompt

class BenchmarkQAataLoader:
    def __init__(self, prompt_generator, n_shots = 0):
        self.prompt_generator = prompt_generator
        self.n_shots = n_shots
    
    def get_prefix_prompts(self, task='all'):
        return_dict = {}
            
        for subject in (self.task if task == 'all' else [task]):
            prompt = self.prefix_prompt.format(subject)
            
            ref_key = "category" if "category" in self.ref_data.features.keys() else "subject"
            if self.n_shots > 0:
                start_ref_index = self.ref_data[ref_key].index(subject)
                for k in range(self.n_shots):
                    prompt += self.prompt_generator(self.ref_data[start_ref_index+k], self.choices, last=False)
            return_dict[subject] = prompt
        return return_dict
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        # get info
        sub_key = "category" if "category" in self.test_data.features.keys() else "subject"
        
        subject = self.test_data[idx][sub_key]
        answer = self.test_data[idx]['answer']
        
        # gen prompt
        prompt = ""
        if self.enable_prefix_promts:
            prompt = self.prefix_prompt.format(subject)
        
        # add n-shot qa from ref_data
        if self.n_shots > 0:
            start_ref_index = self.ref_data[sub_key].index(subject)
            for k in range(self.n_shots):
                prompt += self.prompt_generator(self.ref_data[start_ref_index+k], self.choices, last=False)
        
        # add current qa
        prompt += self.prompt_generator(self.test_data[idx], self.choices)
        
        return prompt, subject, answer
    
    
def run_benchmark(dataset, lita_model, max_new_tokens, extract_fn=None):
    pbar = tqdm(dataset, desc=dataset.__class__.__name__)
    
    log = []
    for idx, (question, subject, answer) in enumerate(pbar):
        res = lita_model.generate(question, max_new_tokens=max_new_tokens)

        if extract_fn:
            out = [extract_fn(ot) for ot in res]
        else:
            out = res
        
        log_dict = {"question": question,
                    "response": out,
                    "answer": answer,
                    "subject": subject}
        
        if lita_model.metric is not None:
            log_dict = {
                **log_dict,
                **lita_model.metric.summary()
            }
        log.append(log_dict)
    
    benchmark_summary(log)
    
    return log
    
    
def benchmark_summary(log):
    df = pd.DataFrame(log)
    
    df["correct"] = df.apply(lambda row: row["response"][0] == ["A", "B", "C", "D"][row["answer"]], axis=1)

    subject_accuracy = df.groupby("subject")["correct"].mean().reset_index()
    subject_accuracy.columns = ["subject", "accuracy"]
    
    overall_accuracy = df["correct"].mean()
    total_execution_time = df["e2e"].sum()

    execution_time_stats = df.groupby("subject").agg(
    mean_e2e=("e2e", np.nanmean),
    median_e2e=("e2e", np.nanmedian),
    p50_e2e=("p50", np.nanmean),
    p99_e2e=("p99", np.nanmean)).reset_index()
    
    print("### Subject Accuracy ###")
    print(subject_accuracy)

    print("\n### Execution Time Statistics ###")
    print(execution_time_stats)

    print("\n### Overall Accuracy ###")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")  
    print(f"Total Execution Time (e2e): {total_execution_time/1000:.2f} s")
