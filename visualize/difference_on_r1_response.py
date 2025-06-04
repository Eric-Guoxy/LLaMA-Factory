import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import json
import warnings
from tqdm import tqdm
import html as html_lib

import torch.nn.functional as F


def calculate_text_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
) -> List[Dict[str, Union[str, float]]]:
    """
    使用 transformers 的标准 forward 接口计算每个 response token 的 log probability。
    """
    # 拼接 full input（prompt + response）
    full_text = prompt + response
    full_input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # response 在 input 中的位置
    response_start = prompt_input_ids.shape[1]
    response_token_ids = full_input_ids[0, response_start:]

    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    # 对 logits 取 softmax 得到概率，再取 log 概率
    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = []
    for i, token_id in enumerate(response_token_ids):
        pos = response_start + i
        token_log_prob = log_probs[0, pos - 1, token_id].item()  # 注意是前一位的预测
        token_text = tokenizer.decode(token_id)
        token_log_probs.append({
            "token": token_text,
            "log_prob": token_log_prob,
            "token_id": token_id.item()
        })

    return token_log_probs


def visualize_log_prob_differences_only_prob(
    prompt,
    final_model_name,
    ref_model_name,
    trained_log_probs: List[Dict[str, Union[str, float]]],
    ref_log_probs: List[Dict[str, Union[str, float]]],
    output_file: Optional[str] = None,
) -> str:
    """
    只用概率变化色彩（蓝-白-红）可视化。
    """
    gen_log_probs = [item["log_prob"] for item in trained_log_probs]
    resp_log_probs = [item["log_prob"] for item in ref_log_probs]
    tokens = [item["token"] for item in trained_log_probs]
    gen_probs = [np.exp(log_p) for log_p in gen_log_probs]
    resp_probs = [np.exp(log_p) for log_p in resp_log_probs]
    diffs = [g - r for g, r in zip(gen_probs, resp_probs)]
    max_diff = max(abs(min(diffs)), abs(max(diffs)))
    if max_diff == 0:
        max_diff = 1.0
    normalized_diffs = [d / max_diff for d in diffs]
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; padding: 20px; background-color: #f9f9f9; color: #333; }
    .text-container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; line-height: 1.8; font-size: 16px; overflow-wrap: break-word; }
    .token { display: inline; padding: 2px 1px; border-radius: 2px; }
    .legend { display: flex; margin: 10px 0 20px; justify-content: center; font-size: 14px; }
    .legend-item { margin: 0 15px; display: flex; align-items: center; }
    .color-box { width: 20px; height: 20px; margin-right: 5px; border-radius: 3px; }
    h1 { text-align: center; color: #444; margin-bottom: 25px; }
    .model-info-container {
        display: flex;
        justify-content: center; /* Centers the items horizontally */
        align-items: center; /* Aligns items vertically if they have different heights */
        gap: 25px; /* Adds space between the model info items */
        padding: 12px; /* Adds some internal spacing */
        margin-bottom: 20px; /* Space below this section */
        background-color: #f0f0f0; /* A light background to distinguish the section */
        border-radius: 5px; /* Rounded corners for a softer look */
        font-size: 14px; /* Consistent font size, similar to legend */
        color: #555; /* Slightly muted text color */
        flex-wrap: wrap; /* Allows items to wrap to the next line on smaller screens */
    }
    </style>
    </head>
    <body>
    <h1>Token Probability Visualization</h1>
    <div class="model-info-container">
        <span><strong>"""
    html += f"""Generate Model:</strong> {final_model_name} </span>
        <span><strong>Reference Model:</strong> {ref_model_name} </span>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="color-box" style="background-color: rgba(100, 149, 237, 0.7);"></div>
            <span>Lower Probability After SFT</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: white;"></div>
            <span>Similar Probability</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: rgba(240, 128, 128, 0.7);"></div>
            <span>Higher Probability After SFT</span>
        </div>
    </div>
    <div class="text-container">
    """
    html += f"<span>{prompt}</span><br>"
    for token, diff, gen_p, resp_p in zip(tokens, normalized_diffs, gen_probs, resp_probs):
        intensity = abs(diff)
        if diff < 0:
            color = f"rgba(100, 149, 237, {intensity})"
        elif diff > 0:
            color = f"rgba(240, 128, 128, {intensity})"
        else:
            color = "transparent"
        escaped_token = html_lib.escape(token)
        html += (
            f'<span class="token" style="background-color: {color}" '
            f'title="Prob M1: {gen_p:.6f}, Prob M2: {resp_p:.6f}, Diff: {gen_p-resp_p:.6f}">' 
            f'{escaped_token}</span>'
        )
    html += """
    </div>
    </body>
    </html>
    """
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
    return html

# Example usage
if __name__ == "__main__":
    # Load models and tokenizer

    model_name_1 = "/home/inspur/cth/models/Qwen2.5-Math-7B"
    model_name_2 = "/home/inspur/cth/models/Qwen2.5-Math-7B-Oat-Zero"
    # Set device
    device = "cpu"
    print(f"Using device: {device}")

    # Load tokenizer (using the first model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)

    # Load first model for generation
    print("Loading first model...")
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_name_1,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_2 = AutoModelForCausalLM.from_pretrained(
        model_name_2,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    
    from utils.utils import read_file
    texts = read_file('compare/dataset_oat_base_sample.json')
    text = texts[0]
    prompt = text['prompt']
    temp = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    print(f"Prompt: {temp}")
    generated_text = text['target'][0]['content']
    print(f"Generated text: {generated_text}")
    # Calculate log probs with second model
    print("\nCalculating log probabilities with second model...")
    ref_log_probs = calculate_text_log_probs(
        model_1, tokenizer, temp, generated_text
    )
    trained_log_probs = calculate_text_log_probs(
        model_2, tokenizer, temp, generated_text
    )
    

    with open('response_log_probs.json', 'w') as f:
        json.dump(ref_log_probs, f, indent=4)
    with open('after_sft_log_probs.json', 'w') as f:
        json.dump(trained_log_probs, f, indent=4)
    # Convert log probs to actual probabilities for statistics
    gen_probs = [np.exp(item["log_prob"]) for item in trained_log_probs]
    resp_probs = [np.exp(item["log_prob"]) for item in ref_log_probs]
    prob_diffs = [g - r for g, r in zip(gen_probs, resp_probs)]

    # Create simple visualization with subtle colors
    html_prob = visualize_log_prob_differences_only_prob(
        temp,
        trained_log_probs = trained_log_probs,
        ref_log_probs = ref_log_probs,
        ref_model_name=model_name_1,
        final_model_name=model_name_2,
        output_file="probability_visualization.html"
    )
    print(f"\nProbability visualization saved to probability_visualization.html")

    # Print statistics using actual probabilities
    print("\nProbability difference statistics:")
    print(f"Mean difference: {sum(prob_diffs) / len(prob_diffs):.6f}")
    print(f"Max difference: {max(prob_diffs):.6f}")
    print(f"Min difference: {min(prob_diffs):.6f}")
    print(f"Std deviation: {np.std(prob_diffs):.6f}")