# This file compares the correctness of the 7b base model
# and the oat version as well as the sft version on the 
# evaluation dataset and extract those answers that are
# wrong for the 7b base but correct for oat/sft.
# 
# This file will then visualize the diff prob for these answers.
import json
from utils import read_jsonl_file
from transformers import AutoTokenizer

def pipeline(base_answers_path, trained_answers_path, tokenizer_path, output_file_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    base_items = read_jsonl_file(base_answers_path)
    trained_items = read_jsonl_file(trained_answers_path)

    assert len(base_items) == len(trained_items), f"The number of items in {base_answers_path} and {trained_answers_path} should be the same."

    # Collect all the answers that are correct for the trained_answers_path but wrong for the base_answers_path
    desired_items = [trained_items[i] for i in range(len(trained_items)) if not base_items[i]['correctness'] and trained_items[i]['correctness']] 
    desired_items.sort(key=lambda item: len(tokenizer.encode(item['generated_text'])), reverse=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(desired_items, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    base_answers_path = "Qwen2.5-Math-7B-base.jsonl"
    oat_answers_path = "Qwen2.5-Math-7B-Oat-Zero-eval.jsonl"
    sft_answers_path = "Qwen2.5-Math-7B-full-sft-final-sft.jsonl"

    oat_output_file = "dataset-oat-base.json"
    sft_output_file = "dataset_sft_base.json"

    tokenizer_path = "/home/inspur/cth/models/Qwen2.5-Math-7B"

    print("Processing oat vs. base")
    pipeline(
        base_answers_path=base_answers_path,
        trained_answers_path=oat_answers_path,
        tokenizer_path=tokenizer_path,
        output_file_path=oat_output_file
    )
    print("End processing oat vs. base")

    print("Processing sft vs. base")
    pipeline(
        base_answers_path=base_answers_path,
        trained_answers_path=sft_answers_path,
        tokenizer_path=tokenizer_path,
        output_file_path=sft_output_file
    )
    print("End processing sft vs. base")