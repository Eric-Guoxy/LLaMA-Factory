import json
import os
import random

if __name__ == '__main__':
    with open("openr1_sft.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    system = "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>"

    samples_5k = random.sample(data, 5000)

    for item in samples_5k:
        item['system'] = system
    
    with open("openr1_5k_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_5k, f, ensure_ascii=False, indent=4)
    