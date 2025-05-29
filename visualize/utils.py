# This file extracts one question for each data source in the benchmark.
import json
import os

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
    

    


if __name__ == '__main__':
    output_jsonl_path = "/home/inspur/cth/LLaMA-Factory/visualize/models/Qwen2.5-Math-7B-curr-part2/Qwen2.5-Math-7B_base/all_samples_diff_probs_summary_add_correctness.jsonl"
    data = read_jsonl_file(output_jsonl_path)
    import pdb; pdb.set_trace()