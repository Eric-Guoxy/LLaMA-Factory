# This file extracts one question for each data source in the benchmark.
import json
import os

def process_benchmark(benchmark_path, output_path):
    with open(benchmark_path, "r", encoding="utf=8") as f:
        benchmark = json.load(f)
    
    data_sources = set()
    extracted_questions = []
    for item in benchmark:
        if item['data_source'] not in data_sources:
            extracted_questions.append(item)
            data_sources.add(item['data_source'])
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_questions, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    benchmark_path = "valid.all.json"
    output_path = "samples.json"
    process_benchmark(benchmark_path, output_path)