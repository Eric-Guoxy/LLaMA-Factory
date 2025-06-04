# This file extracts one question for each data source in the benchmark.
import json
import os
from transformers import AutoTokenizer
import tqdm
import pandas as pd

def read_jsonl_file(file_path):
    """
    Reads a JSON Lines file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries, where each dictionary
              corresponds to a JSON object from a line in the file.
              Returns an empty list if the file is empty or an error occurs.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove leading/trailing whitespace (like the newline character)
                line = line.strip()
                if line:  # Ensure the line is not empty
                    try:
                        json_object = json.loads(line)
                        data.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line due to JSON decoding error: {e} - Line: '{line}'")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return data

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

def write_jsonl_file(data_list, file_path):
    """
    Writes a list of dictionaries to a JSON Lines file.

    Args:
        data_list (list): A list of dictionaries to write.
        file_path (str): The path to the .jsonl file to be created/overwritten.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                json_string = json.dumps(item, ensure_ascii=False)
                f.write(json_string + '\n')
        print(f"Successfully wrote {len(data_list)} items to {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def add_correctness(jsonl_file_path, dataset_path, output_jsonl_path):
    diffs_items = read_jsonl_file(jsonl_file_path)
    dataset = read_jsonl_file(dataset_path)

    prompt_to_correctness = {}

    for item in dataset:
        correctness = item['correctness']
        prompt = item['prompt']
        prompt_to_correctness[prompt] = correctness
    
    new_diffs_items = []
    for diffs_item in diffs_items:
        prompt = diffs_item['prompt']
        correctness = prompt_to_correctness[prompt]
        diffs_item['correctness'] = correctness
        new_diffs_items.append(diffs_item)

    write_jsonl_file(new_diffs_items, output_jsonl_path)

def json_to_parquet(input_json_file, output_parquet_file):
    try:
        # Attempt to read as JSONL first, as it's common in this project
        # read_jsonl_file returns a list of dictionaries
        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            # If read_jsonl_file returned empty (e.g., file not found or truly empty)
            # or if it's a standard JSON file (not JSONL), try reading as standard JSON.
            # This part assumes the standard JSON file contains a list of records
            # or a structure that pandas can directly convert.
            try:
                with open(input_json_file, 'r', encoding='utf-8') as f:
                    json_data_standard = json.load(f)
                if isinstance(json_data_standard, list):
                    data = json_data_standard
                elif isinstance(json_data_standard, dict): # Handle single JSON object or dict of records
                    # If it's a single object, wrap it in a list for DataFrame conversion
                    # Or, if it's a dict of records, pandas can handle it.
                    # For simplicity, assuming it's a list or can be made into one.
                    # This might need adjustment based on the exact JSON structure.
                    data = [json_data_standard] if not isinstance(json_data_standard.get(next(iter(json_data_standard.keys()))), (list, dict)) else pd.DataFrame.from_dict(json_data_standard, orient='index').reset_index().to_dict(orient='records')

            except json.JSONDecodeError as e_json:
                print(f"File {input_json_file} is not a valid JSONL or standard JSON file. JSONDecodeError: {e_json}")
                return
            except Exception as e_std_json:
                print(f"Error reading {input_json_file} as standard JSON: {e_std_json}")
                return

        if not data:
            print(f"No data loaded from {input_json_file}. Parquet file will not be created.")
            return

        df = pd.DataFrame(data)
        df.to_parquet(output_parquet_file, engine='pyarrow', index=False)
        print(f"Successfully converted {input_json_file} to {output_parquet_file}")

    except ImportError:
        print("Pandas and/or PyArrow is not installed. Please install them by running: pip install pandas pyarrow")
    except Exception as e:
        print(f"An error occurred during JSON to Parquet conversion: {e}")

def avg_tokens_response(tokenizer_path, data_jsonl_file):
    data = read_jsonl_file(data_jsonl_file)
    responses = [item['generated_text'] for item in data]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    responses_tokenized = tokenizer(
        responses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16384,
        add_special_tokens=True
    )

def overlap(d1_path, d2_path, d1_output_path, d2_output_path, cutoff_len=4):
    with open(d1_path, "r", encoding="utf-8") as f:
        d1 = json.load(f)
    with open(d2_path, 'r', encoding='utf-8') as f:
        d2 = json.load(f)
    
    d1_prompts = set([item['prompt'] for item in d1])
    overlap_prompts = set()

    for item in d2:
        if item['prompt'] in d1_prompts:
            overlap_prompts.add(item['prompt'])
    
    d1_new = []
    d2_new = []

    for item in d1:
        if item['prompt'] in overlap_prompts:
            d1_new.append(item)
    for item in d2:
        if item['prompt'] in overlap_prompts:
            d2_new.append(item)

    if len(d1_new) > cutoff_len:
        d1_new = d1_new[:cutoff_len]
    if len(d2_new) > cutoff_len:
        d2_new = d2_new[:cutoff_len]

    with open(d1_output_path, 'w', encoding='utf-8') as f:
        json.dump(d1_new, f, ensure_ascii=False, indent=4)
    with open(d2_output_path, 'w', encoding='utf-8') as f:
        json.dump(d2_new, f, ensure_ascii=False, indent=4)
   


if __name__ == '__main__':
    # output_jsonl_path = "/home/inspur/cth/LLaMA-Factory/visualize/models/Qwen2.5-Math-7B-Oat-Zero/Qwen2.5-Math-7B-Oat-Zero-eval.jsonl"
    # data = read_jsonl_file(output_jsonl_path)
    # import pdb; pdb.set_trace()
    d1_path = "dataset-oat-base.json"
    # output_parquet_file = "dataset_oat_base.parquet"
    # json_to_parquet(input_json_file, output_parquet_file)
    d1_output_path = "dataset_oat_base_sample.json"
    d2_path = "dataset_sft_base.json"
    d2_output_path = "dataset_sft_base_sample.json"

    overlap(d1_path, d2_path, d1_output_path, d2_output_path)