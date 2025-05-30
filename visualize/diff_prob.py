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
from generate_vllm import main as generate_vllm_main
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

def batched_calculate_text_log_probs(
    model: Union[AutoModelForCausalLM, torch.nn.DataParallel], # Model can be DP-wrapped
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    processing_batch_size: int = 8,
    primary_device: Optional[torch.device] = None # Pass the primary device
) -> List[List[Dict[str, Union[str, float]]]]:
    """
    Calculate log probabilities for each token in given responses using Hugging Face Transformers.
    Internally processes the input prompts/responses in chunks of `processing_batch_size`.
    If model is DataParallel, primary_device should be set for correct input tensor placement.
    """
    if not prompts or not responses:
        if len(prompts) == 0 and len(responses) == 0:
            return []
        raise ValueError("Prompts and responses lists must be non-empty and of the same length, or both empty.")
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses lists must be of the same length.")

    num_total_samples = len(prompts)
    all_log_probs_for_all_samples = [[] for _ in range(num_total_samples)]

    # Determine the device for input tensors
    if primary_device is None:
        # Fallback if not provided, but it's better to pass it explicitly
        if isinstance(model, torch.nn.DataParallel):
            # DataParallel replicates the module, parameters are on multiple devices.
            # Input should go to the device of the first module replica (usually cuda:0)
            # This assumes model.module exists and has parameters.
            primary_device = next(model.module.parameters()).device
        else:
            primary_device = next(model.parameters()).device
    
    # Get the actual model config, handling DataParallel wrapper
    model_config = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config


    for i in tqdm(range(0, num_total_samples, processing_batch_size), desc="Calculating log probabilities (HF)"):
        current_prompts_chunk = prompts[i:i + processing_batch_size]
        current_responses_chunk = responses[i:i + processing_batch_size]

        full_texts_chunk = [p + r for p, r in zip(current_prompts_chunk, current_responses_chunk)]
        
        prompt_token_lengths_chunk = [
            len(tokenizer.encode(p, add_special_tokens=True)) for p in current_prompts_chunk
        ]

        tokenized_sub_batch = tokenizer(
            full_texts_chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
            add_special_tokens=True
        )

        # Move input tensors to the primary device for DataParallel
        sub_batch_input_ids = tokenized_sub_batch.input_ids.to(primary_device)
        sub_batch_attention_mask = tokenized_sub_batch.attention_mask.to(primary_device)
        
        num_samples_in_chunk = sub_batch_input_ids.shape[0]
        log_probs_for_current_chunk_processing = [[] for _ in range(num_samples_in_chunk)]

        with torch.no_grad():
            # DataParallel handles scattering the input and gathering the output
            outputs = model(sub_batch_input_ids, attention_mask=sub_batch_attention_mask)
            logits_sub_batch = outputs.logits
            
            relevant_logits = logits_sub_batch[:, :-1, :]
            target_ids_for_logprobs = sub_batch_input_ids[:, 1:]
            
            log_softmax_of_logits = torch.log_softmax(relevant_logits, dim=-1)
            chunk_all_token_log_probs = torch.gather(
                log_softmax_of_logits,
                dim=2,
                index=target_ids_for_logprobs.unsqueeze(-1)
            ).squeeze(-1)

            for j in range(num_samples_in_chunk):
                actual_seq_len_in_chunk = sub_batch_attention_mask[j].sum().item()
                response_start_index_in_sample_chunk = prompt_token_lengths_chunk[j]
                
                response_token_ids_this_sample = []
                response_log_probs_this_sample = []

                for k_loop_idx in range(actual_seq_len_in_chunk - 1):
                    token_original_idx_in_full_sequence = k_loop_idx + 1
                    if token_original_idx_in_full_sequence >= response_start_index_in_sample_chunk:
                        token_id = sub_batch_input_ids[j, token_original_idx_in_full_sequence].item()
                        log_prob_val = chunk_all_token_log_probs[j, k_loop_idx].item() 
                        response_token_ids_this_sample.append(token_id)
                        response_log_probs_this_sample.append(log_prob_val)
                
                if response_token_ids_this_sample:
                    tokens_to_decode_for_batch_decode = [[tid] for tid in response_token_ids_this_sample]
                    decoded_token_texts = tokenizer.batch_decode(
                        tokens_to_decode_for_batch_decode,
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    current_sample_output = []
                    for idx, token_id_val in enumerate(response_token_ids_this_sample):
                        current_sample_output.append({
                            "token": decoded_token_texts[idx],
                            "log_prob": response_log_probs_this_sample[idx],
                            "token_id": token_id_val
                        })
                    log_probs_for_current_chunk_processing[j] = current_sample_output
                else:
                    log_probs_for_current_chunk_processing[j] = []
        
        for chunk_idx_inner, sample_log_probs in enumerate(log_probs_for_current_chunk_processing):
            original_sample_idx = i + chunk_idx_inner
            all_log_probs_for_all_samples[original_sample_idx] = sample_log_probs

    return all_log_probs_for_all_samples


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

def vllm_calculate_text_log_probs(
    llm_engine: LLM,
    tokenizer: AutoTokenizer, # Hugging Face tokenizer for prompt length and decoding
    prompts: List[str],
    responses: List[str]
) -> List[List[Dict[str, Union[str, float]]]]:
    """
    EXPERIMENTAL: Attempts to calculate log probabilities for each token in given responses using vLLM,
    by setting `prompt_logprobs` to the vocabulary size.

    WARNINGS:
    - This approach is highly experimental and likely to be very inefficient (slow and memory-intensive).
    - vLLM's `prompt_logprobs` API is designed for top-N, not the full vocabulary.
    - This may not work as expected or could lead to errors or extremely poor performance.
    - For reliable scoring of fixed sequences, the standard Hugging Face Transformers approach
      (full logits -> log_softmax -> gather) is generally preferred.

    Args:
        llm_engine: An initialized vLLM.LLM engine.
        tokenizer: A Hugging Face AutoTokenizer, used for determining prompt length
                   and decoding tokens for the output.
        prompts: A list of input prompt texts.
        responses: A list of response texts corresponding to each prompt.

    Returns:
        A list of lists, where each inner list contains dictionaries of
        (token, log_prob, token_id) for the response of the corresponding input sample.
        Returns an empty list for a token if its log_prob couldn't be found (e.g., not in vLLM's output).
    """
    if not prompts:
        return []
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses lists must be of the same length.")

    full_texts_for_vllm = []
    hf_prompt_token_lengths = []

    for p_str, r_str in zip(prompts, responses):
        full_texts_for_vllm.append(p_str + r_str)
        # Use HF tokenizer to determine where the response part starts.
        # Ensure add_special_tokens matches how vLLM tokenizes/expects prompts.
        # If vLLM adds BOS automatically, set add_special_tokens=False here for prompt length.
        # This part is crucial and might need adjustment based on specific model and vLLM behavior.
        prompt_token_ids_hf = tokenizer.encode(p_str, add_special_tokens=False) 
        hf_prompt_token_lengths.append(len(prompt_token_ids_hf))
        
    if not full_texts_for_vllm:
        return [[] for _ in prompts]

    try:
        # Attempt to get vocab_size from the vLLM engine
        # This might vary depending on vLLM version and how model_config is structured
        vocab_size = llm_engine.llm_engine.model_config.get_vocab_size()
        print(f"Attempting to use vocab_size for prompt_logprobs: {vocab_size}")
    except AttributeError as e:
        print(f"Could not automatically determine vocab_size from llm_engine: {e}")
        print("Please ensure llm_engine.llm_engine.model_config.get_vocab_size() is valid or set vocab_size manually.")
        # Fallback or raise error - for this example, let's raise it.
        raise ValueError("Failed to get vocab_size from vLLM engine.") from e


    sampling_params = SamplingParams(
        temperature=0.0, # Does not affect logprob calculation for a fixed sequence
        max_tokens=1,    # We are not generating, just processing the input.
                         # Set to 1 to ensure the engine runs over the prompt.
        prompt_logprobs=20 # EXPERIMENTAL: Request logprobs for all vocab items
    )

    print(f"Requesting vLLM generation with prompt_logprobs set to vocab_size ({vocab_size}). This may be very slow.")
    request_outputs = llm_engine.generate(
        full_texts_for_vllm, 
        sampling_params, 
        use_tqdm=True
    )

    all_results = []
    for i, req_output in enumerate(request_outputs):
        response_log_probs_for_sample = []
        
        # start_index_of_response_approx is the length of the prompt in terms of vLLM tokens.
        # We are using hf_prompt_token_lengths as an approximation.
        # vLLM's RequestOutput gives prompt_token_ids which are the *full* (prompt+response) token IDs.
        # The prompt part of these vLLM tokens should correspond to our hf_prompt_token_lengths.
        # A more robust way might be to re-tokenize the prompt with vLLM's internal tokenizer if accessible,
        # or rely on the fact that vLLM's prompt_logprobs are for the *input prompt* tokens.
        
        # The prompt_logprobs are for each token in the input `full_texts_for_vllm[i]`.
        # `req_output.prompt_token_ids` are the token IDs for `full_texts_for_vllm[i]`.
        
        vllm_full_text_token_ids = req_output.prompt_token_ids
        # vllm_logprobs_per_position[k] is a dict {token_id: log_prob} for the k-th input token
        vllm_logprobs_per_position = req_output.prompt_logprobs 

        if vllm_full_text_token_ids is None or vllm_logprobs_per_position is None:
            print(f"Warning: vLLM did not return token_ids or prompt_logprobs for sample {i} ('{prompts[i][:30]}...').")
            all_results.append([])
            continue

        # The number of tokens in the prompt part, according to HF tokenizer.
        # This is used to identify where the "response" part begins in the vllm_full_text_token_ids.
        num_prompt_tokens_hf = hf_prompt_token_lengths[i]

        # Iterate over the tokens in the full text (prompt + response) as tokenized by vLLM
        # k is the index in vllm_full_text_token_ids and vllm_logprobs_per_position
        for k in range(len(vllm_full_text_token_ids)):
            # We are interested in the log_prob of vllm_full_text_token_ids[k]
            # *given* vllm_full_text_token_ids[0...k-1].
            # This log_prob should be available in vllm_logprobs_per_position[k].

            # Check if the current token position 'k' is part of the "response"
            # This assumes that the first `num_prompt_tokens_hf` tokens in `vllm_full_text_token_ids`
            # correspond to the prompt. This alignment is critical and can be tricky.
            if k < num_prompt_tokens_hf:
                continue # Skip tokens considered part of the prompt

            actual_token_id_at_k = vllm_full_text_token_ids[k]
            
            # logprob_dict_for_k should contain log_probs for all vocab if vocab_size was used.
            # It's for predicting the token *at* position k, given tokens 0 to k-1.
            # However, vLLM's prompt_logprobs are often structured such that prompt_logprobs[k]
            # are the logprobs for tokens *at* position k, given 0..k-1.
            # The actual token at position k is vllm_full_text_token_ids[k].
            
            if k >= len(vllm_logprobs_per_position) or vllm_logprobs_per_position[k] is None:
                # This can happen if the sequence is too short or vLLM doesn't provide logprobs for the last token.
                # print(f"Debug: No logprobs dict at position {k} for sample {i}")
                continue
            
            logprob_dict_for_k_th_token = vllm_logprobs_per_position[k]

            if actual_token_id_at_k in logprob_dict_for_k_th_token:
                log_prob = logprob_dict_for_k_th_token[actual_token_id_at_k]
                # Decode using the passed HF tokenizer for consistency in output format
                token_text = tokenizer.decode([actual_token_id_at_k])
                response_log_probs_for_sample.append({
                    "token": token_text,
                    "log_prob": log_prob,
                    "token_id": actual_token_id_at_k
                })
            else:
                # If vocab_size was truly used and worked, this case should ideally not be hit
                # unless there's a mismatch or vLLM still did some form of top-k.
                # print(f"Debug: Token {actual_token_id_at_k} ('{tokenizer.decode([actual_token_id_at_k])}') not in logprobs dict for sample {i} at pos {k}. This is unexpected if prompt_logprobs=vocab_size worked.")
                # Fallback: append with a placeholder or skip
                print(f"Log prob not found for token: {tokenizer.decode([actual_token_id_at_k])}")
                response_log_probs_for_sample.append({
                    "token": tokenizer.decode([actual_token_id_at_k]),
                    "log_prob": float('-inf'), # Or None, or some indicator of missing
                    "token_id": actual_token_id_at_k,
                    "error": "Logprob not found in vLLM output"
                })
        
        all_results.append(response_log_probs_for_sample)
        
    return all_results

def pipeline(prompts: List[str], 
             responses: List[str], 
             all_correctness: List[str],
             data_sources: List[str], 
             all_token_log_probs: List[List[Dict[str, Union[str, float]]]], 
             save_path: str, 
             tokenizer: AutoTokenizer, 
             ref_model_path: str, 
             final_model_name: str, 
             ref_model_name: str,
             hf_processing_batch_size: int = 8,
):
    # Determine primary device for model loading
    primary_device_for_models = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Primary device for loading models: {primary_device_for_models}")

    # Load ref model for response evaluation
    print(f"Loading {ref_model_name} to {primary_device_for_models}...")
    _ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16,
        # No device_map="auto" here
    )
    _ref_model.to(primary_device_for_models)
    _ref_model.eval()
    
    # print(f"Compiling {ref_model_name} with torch.compile(mode=\"reduce-overhead\")...")
    # try:
    #     _ref_model_compiled = torch.compile(_ref_model, mode="reduce-overhead")
    #     print(f"{ref_model_name} compiled successfully.")
    # except Exception as e:
    #     print(f"Failed to compile {ref_model_name}: {e}. Using uncompiled model.")
    #     _ref_model_compiled = _ref_model # Fallback to uncompiled

    # ref_model_for_dp = _ref_model_compiled # Use compiled model for DP
    ref_model_for_dp = _ref_model

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using torch.nn.DataParallel for {ref_model_name} across {torch.cuda.device_count()} GPUs.")
        ref_model_for_dp = torch.nn.DataParallel(ref_model_for_dp)
        # Note: After DataParallel, accessing original attributes might need .module
        # e.g., ref_model_for_dp.module.config
    
    # The model passed to batched_calculate_text_log_probs is now potentially DP-wrapped
    # Its parameters are on multiple devices if DP is used.
    # Input tensors inside batched_calculate_text_log_probs need to go to primary_device_for_models.

    print(f"\nCalculating log probabilities with {ref_model_name} for all samples (batched)...")
    all_ref_model_response_log_probs = batched_calculate_text_log_probs(
        model=ref_model_for_dp, # Pass the (potentially DP-wrapped) model
        tokenizer=tokenizer,
        prompts=prompts, 
        responses=responses, 
        processing_batch_size=hf_processing_batch_size,
        primary_device=primary_device_for_models # Specify primary device for input tensors
    )

    # ref_llm_engine = LLM(
    #     model=ref_model_path,
    #     tokenizer=ref_model_path,
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     dtype="bfloat16",
    #     trust_remote_code=True
    # )

    # all_ref_model_response_log_probs = vllm_calculate_text_log_probs(
    #     llm_engine=ref_llm_engine,
    #     tokenizer=tokenizer,
    #     prompts=prompts,
    #     responses=responses,
    # )
    
    saved_diff_probs_overall = []
    # Define the specific save path for this ref_model's results
    # Sanitize ref_model_name for use in path
    safe_ref_model_name_for_path = ref_model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    current_ref_model_output_path = os.path.join(save_path, safe_ref_model_name_for_path)
    os.makedirs(current_ref_model_output_path, exist_ok=True)

    num_samples = len(prompts)
    for i in tqdm(range(num_samples), desc="Processing samples for diff & visualization"):
        current_prompt = prompts[i]
        # Ensure data_sources has an entry for each sample, or provide a default
        current_data_source = data_sources[i] if i < len(data_sources) else f"sample_{i}"
        current_generated_text = responses[i]
        
        # Log probabilities from the final_model (passed as argument)
        token_log_probs_from_final_model = all_token_log_probs[i]
        # Log probabilities from the ref_model (calculated above in batch)
        response_log_probs_from_ref_model = all_ref_model_response_log_probs[i]

        if not token_log_probs_from_final_model or not response_log_probs_from_ref_model:
            print(f"Warning: Skipping diff for sample {i} ('{current_data_source}') due to missing log_probs from one or both models.")
            diff_entry = {
                "prompt": current_prompt, "generated_text": current_generated_text, "diff_probs": [],
                "data_source": current_data_source, "error": "Missing log_probs"
            }
            saved_diff_probs_overall.append(diff_entry)
            continue

        # Convert log probs to actual probabilities for statistics
        gen_probs = [np.exp(item["log_prob"]) for item in token_log_probs_from_final_model if "log_prob" in item]
        resp_probs = [np.exp(item["log_prob"]) for item in response_log_probs_from_ref_model if "log_prob" in item]
        
        min_len = min(len(gen_probs), len(resp_probs))
        if len(gen_probs) != len(resp_probs):
            print(f"Warning: Mismatch in token count for diff calculation (sample {i}, '{current_data_source}'). Gen: {len(gen_probs)}, Resp: {len(resp_probs)}. Using min_len: {min_len}")
        
        prob_diffs = [g - r for g, r in zip(gen_probs[:min_len], resp_probs[:min_len])]

        if not prob_diffs and min_len > 0: # Check if diffs are empty despite having tokens
             print(f"Warning: prob_diffs is empty for sample {i} ('{current_data_source}') even though min_len is {min_len}. This might indicate an issue with token lists or log_prob items.")
        if not prob_diffs : # Handles cases where min_len is 0 or prob_diffs became empty
            print(f"Skipping visualization and stats for sample {i} ('{current_data_source}') as no comparable probabilities were found.")
            diff_entry = {
                "prompt": current_prompt, "generated_text": current_generated_text, "diff_probs": [],
                "data_source": current_data_source, "error": "No comparable probabilities"
            }
            saved_diff_probs_overall.append(diff_entry)
            continue

        # Sanitize data_source for use in filename
        safe_data_source_name = current_data_source.replace('/', '_').replace(' ', '_')
        html_output_filename = os.path.join(current_ref_model_output_path, f"prob_viz_{safe_data_source_name}_{i}.html")
        
        # Create visualization
        visualize_log_prob_differences_only_prob(
            prompt=current_prompt,
            final_model_name=final_model_name, # Use the passed final_model_name
            ref_model_name=ref_model_name,     # Use the passed ref_model_name
            token_log_probs=token_log_probs_from_final_model[:min_len], 
            response_log_probs=response_log_probs_from_ref_model[:min_len],
            output_file=html_output_filename
        )

        # Store diff information
        diff_entry = {
            "prompt": current_prompt,
            "generated_text": current_generated_text,
            "diff_probs": prob_diffs,
            "data_source": current_data_source,
            "mean_diff": sum(prob_diffs) / len(prob_diffs) if prob_diffs else None,
            "max_diff": max(prob_diffs) if prob_diffs else None,
            "min_diff": min(prob_diffs) if prob_diffs else None,
            "std_diff": np.std(prob_diffs) if prob_diffs else None,
            "correctness": all_correctness[i]
        }
        saved_diff_probs_overall.append(diff_entry)
    
    # Save the overall diff_probs summary to a file
    overall_summary_filename = os.path.join(current_ref_model_output_path, f"all_samples_diff_probs_summary.jsonl")
    with open(overall_summary_filename, "w", encoding="utf-8") as f:
        for entry in saved_diff_probs_overall:
            f.write(json.dumps(entry) + '\n')
    print(f"\nOverall diff_probs summary saved to {overall_summary_filename}")
    
    # Clean up the reference model and GPU memory
    print(f"Deleting reference model: {ref_model_name}")
    del _ref_model 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # If torch.distributed was initialized by this model and needs specific cleanup:
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     torch.distributed.destroy_process_group() # If applicable
    #     torch.distributed.empty_cache()
    print(f"Finished processing for reference model: {ref_model_name}")

if __name__ == "__main__":
    # ... (setup paths, names, etc.) ...
    generated_text_path = "/home/inspur/cth/LLaMA-Factory/visualize/models/Qwen2.5-Math-7B-Oat-Zero-oat-prompt/Qwen2.5-Math-7B-Oat-Zero-oat-prompt-eval.jsonl"
    model_name_final_hf = "/home/inspur/cth/models/Qwen2.5-Math-7B-Oat-Zero" # Renamed for clarity
    model_name_base_hf = "/home/inspur/cth/models/Qwen2.5-Math-7B" # Renamed for clarity
    dataset_path = "~/cth/LLaMA-Factory/data/valid.all.parquet"
    final_model_display_name = "Qwen2.5-Math-7B-Oat-Zero-oat-prompt"
    ref_model_display_name = "Qwen2.5-Math-7B (base)"
    
    save_path_root = os.path.join("models", final_model_display_name) 
    os.makedirs(save_path_root, exist_ok=True)
    dataset_path = os.path.expanduser(dataset_path)

    # Determine primary device for model loading
    primary_device_for_models = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Primary device for loading models: {primary_device_for_models}")
    print(f"Using device: {primary_device_for_models}") # General device info

    tokenizer = AutoTokenizer.from_pretrained(model_name_final_hf)

    # ... (load data_sources, generate texts if needed) ...
    try:
        df = pd.read_parquet(dataset_path)
        data_sources = df['data_source'].tolist()
    except Exception as e:
        print(f"Error reading data_sources from {dataset_path}: {e}")
        data_sources = []

    if not os.path.exists(generated_text_path):
        print(f"Generated text file not found at {generated_text_path}. Generating...")
        # Assuming generate_vllm_main is correctly defined and imported
        generate_vllm_main(
            input_file=dataset_path,
            output_file=generated_text_path,
            model_path=model_name_final_hf, # Assuming vLLM uses the same path
            tokenizer_path=model_name_final_hf,
            max_tokens=16384,
            remove_system=True, 
            template='oat',     
            add_oat_evaluate=True, 
            tensor_parallel_size=4
        )
    else:
        print(f"Using existing generated text file: {generated_text_path}")
    
    data = read_jsonl_file(generated_text_path)
    prompts = []
    responses = []
    all_correctness = []
    for item in data:
        prompts.append(item.get('prompt', ''))
        responses.append(item.get('generated_text', item.get('answer', '')))
        all_correctness.append(item.get('correctness', ''))
    
    if not prompts:
        print(f"No prompts/responses loaded from {generated_text_path}. Exiting.")
        exit(1)
    
    if data_sources and len(data_sources) != len(prompts):
        print(f"Warning: Length of data_sources ({len(data_sources)}) does not match "
              f"number of prompts/responses ({len(prompts)}). Adjusting data_sources.")
        if len(data_sources) > len(prompts):
            data_sources = data_sources[:len(prompts)]
        else:
            data_sources.extend([f"sample_{idx+len(data_sources)}" for idx in range(len(prompts) - len(data_sources))])


    # Load the final model
    print(f"Loading final model ({final_model_display_name}) from {model_name_final_hf} to {primary_device_for_models}...")
    _final_model = AutoModelForCausalLM.from_pretrained(
        model_name_final_hf,
        torch_dtype=torch.bfloat16,
        # No device_map="auto"
    )
    _final_model.to(primary_device_for_models)
    _final_model.eval()

    # print(f"Compiling final model ({final_model_display_name}) with torch.compile(mode=\"reduce-overhead\")...")
    # try:
    #     _final_model_compiled = torch.compile(_final_model, mode="reduce-overhead")
    #     print(f"Final model ({final_model_display_name}) compiled successfully.")
    # except Exception as e:
    #     print(f"Failed to compile final model ({final_model_display_name}): {e}. Using uncompiled model.")
    #     _final_model_compiled = _final_model # Fallback

    # final_model_for_dp = _final_model_compiled
    final_model_for_dp = _final_model

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using torch.nn.DataParallel for final model ({final_model_display_name}) across {torch.cuda.device_count()} GPUs.")
        final_model_for_dp = torch.nn.DataParallel(final_model_for_dp)
    
    print(f"Calculating log probabilities with HF for final_model: {final_model_display_name}")
    all_token_log_probs = batched_calculate_text_log_probs(
        model=final_model_for_dp, # Pass the (potentially DP-wrapped) model
        tokenizer=tokenizer,
        responses=responses,
        prompts=prompts,
        processing_batch_size=4, # This will be split across GPUs by DataParallel
        primary_device=primary_device_for_models # Specify primary device
    )

    print(f"Deleting final model: {final_model_display_name}")
    del _final_model, final_model_for_dp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Call pipeline
    pipeline(
        prompts=prompts,
        responses=responses,
        all_correctness=all_correctness,
        data_sources=data_sources,
        all_token_log_probs=all_token_log_probs,
        save_path=save_path_root,
        tokenizer=tokenizer,
        ref_model_path=model_name_base_hf,
        final_model_name=final_model_display_name,
        ref_model_name=ref_model_display_name,
        hf_processing_batch_size=4 # This will be split across GPUs by DataParallel inside pipeline
    )

    print("Visualization pipeline finished.")