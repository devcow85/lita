import os

import pandas as pd

from lita import load_json

def main():

    results = []
    profile_index_tree = load_json("examples/profile_index_A6000ada.json")
    base_profile_dir = 'profile_result'
    max_iter = 20

    for model, block in profile_index_tree.items():
        
        for i in range(max_iter):
            file_name = os.path.join(base_profile_dir, model, f"trace_{i}.json")
            trace_json = load_json(file_name)
            
            for subblock, operations in block.items():
                
                for operation, phases in operations.items():
                    for phase, indices in phases.items():
                        for index in indices:
                            trace_event = trace_json['traceEvents'][index]
                            
                            results.append({
                                'Model': model,
                                'Iteration': i,
                                "Subblock": subblock,
                                "Operation": operation,
                                "Phase": phase,
                                "Index": index,
                                "Name": trace_event['name'],
                                "Duration": trace_event['dur'],
                                "Input_shape": trace_event['args'].get('Input Dims', None)
                            })
    df = pd.DataFrame(results)
    print(df.head())
    
    df['Input_shape_str'] = df['Input_shape'].apply(lambda x: str(x))
    grouped = df.groupby(['Model', 'Subblock', 'Operation', 'Phase', 'Index', 'Name', 'Input_shape_str'])
    stats = grouped['Duration'].quantile([0.99, 0.90, 0.50]).unstack()
    stats.columns = ['P50', 'P90', 'P99']
    stats = stats.reset_index()
    print(stats)

    stats.to_csv(os.path.join(base_profile_dir,"trace_analysis_A6000ada.csv"), index=False) ###
    
        
if __name__ == "__main__":
    main()