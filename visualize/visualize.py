import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import json
import warnings
from tqdm import tqdm
import html as html_lib
import os
from vllm import LLM, SamplingParams


def generate_with_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_sdpa: bool = True,
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Generate text with the model and record log probabilities for each generated token.
    """
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = input_ids.shape[1]

    # Track token log probabilities
    token_log_probs = []

    # Filter warnings if specified
    if not use_sdpa:
        warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")

    # Create progress bar
    pbar = tqdm(total=max_new_tokens, desc="Generating tokens", ncols=500)

    # Initialize past_key_values for KV-cache
    past_key_values = None

    # Generate text token by token
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Use past_key_values for faster generation
            outputs = model(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

        # Get logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / max(temperature, 1e-8)

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float("Inf")

        # Get probabilities
        probs = torch.softmax(next_token_logits, dim=-1)

        # Sample or greedy selection
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # Get the log probability of the selected token
        log_prob = torch.log(probs[0, next_token[0]]).item()
        token_text = tokenizer.decode(next_token[0])

        # Store token and its log probability
        token_log_probs.append({
            "token": token_text,
            "log_prob": log_prob,
            "token_id": next_token[0].item()
        })

        # Add the next token to input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Update progress bar
        pbar.update(1)

        # Check if we've reached the end of text token
        if next_token[0].item() == tokenizer.eos_token_id:
            break

    # Close progress bar
    pbar.close()

    # Generate the full text
    generated_text = tokenizer.decode(input_ids[0][prompt_length:], skip_special_tokens=True)

    return generated_text, token_log_probs


def calculate_text_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    use_sdpa: bool = True,
    batch_size: int = 32
) -> List[Dict[str, Union[str, float]]]:
    """
    Calculate log probabilities for each token in a given response text.

    This function computes the log probabilities that the model assigns to each token
    in the response when conditioned on the prompt. It processes the full text
    (prompt + response) through the model and extracts probabilities for just the
    response tokens.

    Args:
        model: The language model
        tokenizer: The tokenizer corresponding to the model
        prompt: The input prompt text
        response: The response text to calculate log probabilities for
        use_sdpa: Whether to use scaled dot product attention
        batch_size: Batch size for processing tokens

    Returns:
        List of dictionaries containing token, log probability, and token ID
    """
    # Tokenize the prompt and response
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_text = prompt + response
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    # Identify where response tokens start
    response_start = prompt_ids.shape[1]

    # Filter warnings if specified
    if not use_sdpa:
        warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")

    # Calculate log probabilities for each response token
    token_log_probs = []

    # Handle empty response case
    if response_start >= full_ids.shape[1]:
        return token_log_probs

    # Create progress bar for token processing
    total_tokens = full_ids.shape[1]
    num_response_tokens = total_tokens - response_start + 1
    pbar = tqdm(total=num_response_tokens, desc="Calculating token probabilities", ncols=100)

    # Process tokens in batches
    past_key_values = None
    current_pos = 0

    while current_pos < total_tokens:
        # Determine batch end for this iteration
        batch_end = min(current_pos + batch_size, total_tokens)

        with torch.no_grad():
            if past_key_values is None:
                # First pass: process multiple tokens at once
                outputs = model(
                    full_ids[:, current_pos:batch_end],
                    past_key_values=None,
                    use_cache=True
                )
                # Process each position in the batch (except the last one since we need the next token)
                for i in range(batch_end - current_pos - 1):
                    pos = current_pos + i
                    next_token_id = full_ids[0, pos + 1].item()

                    # Calculate probability distribution
                    current_logits = outputs.logits[0, i, :]
                    probs = torch.softmax(current_logits, dim=-1)

                    # Get log probability of the actual next token
                    log_prob = torch.log(probs[next_token_id]).item()

                    # Only record tokens that are part of the response
                    if pos >= response_start - 1:
                        token_text = tokenizer.decode(full_ids[0, pos + 1:pos + 2])
                        token_log_probs.append({
                            "token": token_text,
                            "log_prob": log_prob,
                            "token_id": next_token_id
                        })
                        # Update progress bar
                        pbar.update(1)

                # Save the past key values for the next iteration
                past_key_values = outputs.past_key_values
                current_pos = batch_end - 1  # Move position to the last token processed
            else:
                # Subsequent passes: use KV cache and process one token at a time
                outputs = model(
                    full_ids[:, current_pos:current_pos + 1],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values

                # Only process if we're not at the last token
                if current_pos < total_tokens - 1:
                    next_token_id = full_ids[0, current_pos + 1].item()

                    # Calculate probability distribution
                    current_logits = outputs.logits[0, 0, :]
                    probs = torch.softmax(current_logits, dim=-1)

                    # Get log probability of the actual next token
                    log_prob = torch.log(probs[next_token_id]).item()

                    # Only record tokens that are part of the response
                    if current_pos >= response_start - 1:
                        token_text = tokenizer.decode(full_ids[0, current_pos + 1:current_pos + 2])
                        token_log_probs.append({
                            "token": token_text,
                            "log_prob": log_prob,
                            "token_id": next_token_id
                        })
                        # Update progress bar
                        pbar.update(1)

                # Move to the next position
                current_pos += 1

    # Close progress bar
    pbar.close()

    return token_log_probs


def save_log_probs(log_probs: List[Dict[str, Union[str, float]]], output_file: str):
    """
    Save token log probabilities to a file.

    Args:
        log_probs: List of dictionaries containing token and log probability
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_probs, f, ensure_ascii=False, indent=2)


def visualize_log_prob_differences_only_prob(
    prompt,
    final_model_name,
    ref_model_name,
    token_log_probs: List[Dict[str, Union[str, float]]],
    response_log_probs: List[Dict[str, Union[str, float]]],
    output_file: Optional[str] = None,
) -> str:
    """
    只用概率变化色彩（蓝-白-红）可视化。
    """
    gen_log_probs = [item["log_prob"] for item in token_log_probs]
    resp_log_probs = [item["log_prob"] for item in response_log_probs]
    tokens = [item["token"] for item in token_log_probs]
    gen_probs = [np.exp(log_p) for log_p in gen_log_probs]
    resp_probs = [np.exp(log_p) for log_p in resp_log_probs]
    diffs = [g - r for g, r in zip(gen_probs, resp_probs)]
    max_diff = max(abs(min(diffs)), abs(max(diffs)))
    if max_diff == 0:
        max_diff = 1.0
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
            <span>Lower Probability After LUFFY</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: white;"></div>
            <span>Similar Probability</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: rgba(240, 128, 128, 0.7);"></div>
            <span>Higher Probability After LUFFY</span>
        </div>
    </div>
    <div class="text-container">
    """
    html += f"<span>{prompt}</span><br>"
    for token, diff, gen_p, resp_p in zip(tokens, diffs, gen_probs, resp_probs):
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

def pipeline(all_QA_pairs, save_path, tokenizer, ref_model_path, final_model_name, ref_model_name):
    # Load ref model for response evaluation
    print(f"Loading {ref_model_name}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    saved_diff_probs = []
    for QA_pair in tqdm(all_QA_pairs, desc="Calculating differences in probs"):
        prompt = QA_pair['prompt']
        question = QA_pair['question']
        generated_text = QA_pair['generated_text']
        token_log_probs = QA_pair['token_log_probs']
        

        save_path = os.path.join(save_path, ref_model_name)
        os.makedirs(save_path, exist_ok=True)

        # Calculate log probs with ref model
        print(f"\nCalculating log probabilities with {ref_model_name}...")
        response_log_probs = calculate_text_log_probs(
            ref_model, tokenizer, prompt, generated_text, use_sdpa=False, batch_size=16
        )

        # Convert log probs to actual probabilities for statistics
        gen_probs = [np.exp(item["log_prob"]) for item in token_log_probs]
        resp_probs = [np.exp(item["log_prob"]) for item in response_log_probs]
        
        prob_diffs = [g - r for g, r in zip(gen_probs, resp_probs)]

        # Create simple visualization with subtle colors
        html_prob = visualize_log_prob_differences_only_prob(
            prompt,
            final_model_name,
            ref_model_name,
            token_log_probs, 
            response_log_probs,
            output_file=os.path.join(save_path, f"probability_visualization_{question['data_source']}_{ref_model_name}.html")
        )
        print(f"\nProbability visualization saved to probability_visualization.html")

        # Print statistics using actual probabilities
        print("\nProbability difference statistics:")
        print(f"Mean difference: {sum(prob_diffs) / len(prob_diffs):.6f}")
        print(f"Max difference: {max(prob_diffs):.6f}")
        print(f"Min difference: {min(prob_diffs):.6f}")
        print(f"Std deviation: {np.std(prob_diffs):.6f}")

        diff_probs = {
            "prompt": prompt,
            "generated_text": generated_text,
            "diff_probs": prob_diffs
        }

        saved_diff_probs.append(diff_probs)
    
    # Save the saved_diff_probs to a file
    with open(os.path.join(save_path, f"saved_diff_probs.json"), "w", encoding="utf-8") as f:
        json.dump(saved_diff_probs, f, ensure_ascii=False, indent=4)
    
    # Clean up the memory of gpus for later usage
    if torch.distributed.is_available():
        torch.distributed.empty_cache()

def main(generated_text_path=None):
    # Load models and tokenizer
    model_name_final = "/data/cth/saves/Qwen2.5-Math-7B/full/sft_curr_part1"
    model_name = "Qwen2.5-Math-7B-curr-part1"
    model_name_base = "/home/inspur/cth/models/Qwen2.5-Math-7B"
    final_model_name = "Qwen2.5-Math-7B-curr-part1"
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", model_name)
    os.makedirs(save_path, exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer (using the first model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name_final)

    # Generate text with first model and evaluate with second model
    with open("samples.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    prompts = []
    prompt_to_q = {}
    for question in questions:
        raw_prompt = question['prompt']
        prompt = tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        prompt_to_q[prompt] = question

    # Load the final model
    final_model = AutoModelForCausalLM(
        model_name_final,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    all_QA_pairs = []
    # Determine whether the generated text has already been stored to a file
    if generated_text_path is not None:
        with open(generated_text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            generated_text = item['generated_text']
            prompt = item['prompt']

            token_log_probs = calculate_text_log_probs(
                final_model, 
                tokenizer,
                prompt,
                generated_text,
                use_sdpa=False,
                batch_size=16
            )

            QA_pair = {
                "prompt": prompt,
                "question": prompt_to_q[prompt],
                "generated_text": generated_text,
                "token_log_probs": token_log_probs
            }

            all_QA_pairs.append(QA_pair)
    else:
        # Clean up the gpu memory for future usage
        if torch.distributed.is_available():
            torch.distinguish.empty_cache()

        # Parameters for Sampling params
        vllm_temperature = 0.6
        vllm_top_p = 1.0
        vllm_max_new_tokens = 10240

        sampling_params = SamplingParams(
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            max_tokens=vllm_max_new_tokens,
            logprobs=1
        )
        
        vllm_instance = LLM(model=model_name_final, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
        print("Generating responses with vLLM...")
        vllm_outputs = vllm_instance.generate(prompts, sampling_params)
        print("vLLM generation complete!")

        for i, vllm_output in enumerate(vllm_outputs):
            prompt = vllm_output.prompt
            generated_text = vllm_output.outputs[0].text
            vllm_token_ids = vllm_output.outputs[0].token_ids
            vllm_logprobs = vllm_output.outputs[0].logprobs

            token_log_probs = []
            for token_idx, token_id in enumerate(vllm_token_ids):
                token_log_prob = vllm_logprobs[token_idx].get(token_id).logprob
                token_text = tokenizer.decode([token_id])
                token_log_probs.append({
                    "token": token_text,
                    "log_prob": token_log_prob,
                    "token_id": token_id
                })
            
            QA_pair = {
                "prompt": prompt,
                "question": prompt_to_q[prompt],
                "generated_text": generated_text,
                "token_log_probs": token_log_probs
            }

            all_QA_pairs.append(QA_pair)

    # Calculate relative probabilities
    pipeline(
        all_QA_pairs=all_QA_pairs,
        save_path=save_path,
        tokenizer=tokenizer,
        ref_model_path=model_name_base,
        final_model_name=final_model_name,
        ref_model_name="Qwen2.5-Math-7B (base)"
    )


if __name__ == "__main__":
    main()