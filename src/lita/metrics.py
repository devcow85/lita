import numpy as np

class TimeMetric:
    def __init__(self):
        self.e2e_times = []
        self.ttft_times = []
        self.tbt_times = []
        self.num_tokens = []

    def record(self, time_dict):
        self.e2e_times.append(time_dict["E2E"])
        self.ttft_times.append(time_dict["TTFT"])
        self.tbt_times.extend(time_dict["TBT"])
        self.num_tokens.append(time_dict["NUM_TOKEN"])

    def calculate_statistics(self):
        stats = {
            'e2e_mean': np.mean(self.e2e_times),
            'e2e_p99': np.percentile(self.e2e_times, 99),
            'e2e_p90': np.percentile(self.e2e_times, 90),
            'e2e_p50': np.percentile(self.e2e_times, 50),
            'ttft_mean': np.mean(self.ttft_times),
            'ttft_p99': np.percentile(self.ttft_times, 99),
            'ttft_p90': np.percentile(self.ttft_times, 90),
            'ttft_p50': np.percentile(self.ttft_times, 50),
            'tbt_mean': np.mean(self.tbt_times),
            'tbt_p99': np.percentile(self.tbt_times, 99),
            'tbt_p90': np.percentile(self.tbt_times, 90),
            'tbt_p50': np.percentile(self.tbt_times, 50),
            'tokens_per_second': sum(self.num_tokens) / sum(self.e2e_times) if sum(self.e2e_times) > 0 else 0,
        }
        return stats

    def print_statistics(self):
        stats = self.calculate_statistics()
        print(f"End-to-End Latency (mean): {stats['e2e_mean']:.4f} seconds")
        print(f"End-to-End Latency (P90): {stats['e2e_p90']:.4f} seconds")
        print(f"End-to-End Latency (P50): {stats['e2e_p50']:.4f} seconds")
        
        print(f"Time to First Token (mean): {stats['ttft_mean']:.4f} seconds")
        print(f"Time to First Token (P90): {stats['ttft_p90']:.4f} seconds")
        print(f"Time to First Token (P50): {stats['ttft_p50']:.4f} seconds")
        
        print(f"Time Between Tokens (mean): {stats['tbt_mean']:.4f} seconds")
        print(f"Time Between Tokens (P90): {stats['tbt_p90']:.4f} seconds")
        print(f"Time Between Tokens (P50): {stats['tbt_p50']:.4f} seconds")
        
        print(f"Tokens Per Second: {stats['tokens_per_second']:.2f} tokens/sec")
