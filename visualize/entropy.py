from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import read_jsonl_file
import torch
import argparse
import math
import random
from tqdm import tqdm
import os
import json

def mean_entropy(generated_text, model, tokenizer, device):
    token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return [], 0.0
    
    effective_token_ids = []
    has_bos_prepended = False
    if tokenizer.bos_token_id is not None:
        effective_token_ids.append(tokenizer.bos_token_id)
        has_bos_prepended = True
    effective_token_ids.extend(token_ids)
    
    if not effective_token_ids:
        return [float('nan')] * num_tokens, 0.0

    effective_token_tensors = torch.tensor([effective_token_ids], device=device)

    entorpies = []

    with torch.no_grad():
        try:
            outputs = model(effective_token_tensors)
            all_logits = outputs.logits[0]  # Shape: (sequence_length, vocab_size)
        except Exception as e:
            print(f"Error during model inference: {e}")
            return [float('nan')] * num_tokens, 0.0
        
        logits = torch.empty((0, all_logits.shape[-1]), device=device)

        if has_bos_prepended:
            if all_logits.shape[0] >= num_tokens:
                logits = all_logits[:num_tokens, :]
        else:
            if num_tokens > 1 and all_logits.shape[0] >= num_tokens - 1:
                logits = all_logits[:num_tokens-1, :]
        
        if logits.nelement() > 0:
            probs = torch.softmax(logits, dim=-1)
            entropies_values = -torch.sum(probs * torch.log2(probs.clamp_min(1e-9)), dim=-1)
            entropies = entropies_values.tolist()
    
    final_token_entropies = [float('nan')] * num_tokens
    if has_bos_prepended:
        for i in range(min(len(entropies), num_tokens)):
            final_token_entropies[i] = entropies[i]
    elif num_tokens > 0: # No BOS, entropies are for c2, ..., cN
        for i in range(len(entropies)):
            if (i + 1) < num_tokens:
                 final_token_entropies[i + 1] = entropies[i]
    
    valid_entropies_for_mean = [e for e in final_token_entropies if not math.isnan(e)]
    mean_entropy = 0.0
    if valid_entropies_for_mean:
        mean_entropy = sum(valid_entropies_for_mean) / len(valid_entropies_for_mean)
    
    return final_token_entropies, mean_entropy

def batched_mean_entropy(texts: list[str], model, tokenizer, device):
    """
    Calculates the mean entropy for a batch of texts.
    The mean entropy is the average of the conditional entropies of the model's
    predictive distribution for each token in a given text.
    """
    if not texts:
        return []

    # 1. Tokenize content (no special tokens yet, no padding yet)
    # Truncation is on by default if texts are too long for the model
    content_tokenizations = tokenizer(
        texts, 
        add_special_tokens=False, 
        padding=False, # We'll pad after manually adding BOS if needed
        truncation=True
    )
    content_ids_list = content_tokenizations["input_ids"]
    num_content_tokens_list = [len(ids) for ids in content_ids_list]

    # 2. Prepare model input sequences (add BOS if tokenizer uses it)
    model_input_ids_list = []
    has_bos_prepended = tokenizer.bos_token_id is not None
    for ids in content_ids_list:
        if has_bos_prepended:
            model_input_ids_list.append([tokenizer.bos_token_id] + ids)
        else:
            model_input_ids_list.append(ids)

    # 3. Pad model input sequences for batching
    # Ensure tokenizer has a pad_token_id; if not, set it (e.g., to eos_token_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback: Add a new pad token if necessary, though this is less ideal
            # For most pretrained models, eos_token_id or an existing pad_token_id should be available.
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer)) # If new token added
            raise ValueError("Tokenizer must have a pad_token_id for batch padding.")

    padded_inputs = tokenizer.pad(
        {"input_ids": model_input_ids_list},
        padding=True, # Pad to the longest sequence in the batch
        return_tensors="pt",
        return_attention_mask=True
    )
    batch_input_ids = padded_inputs["input_ids"].to(device)
    batch_attention_mask = padded_inputs["attention_mask"].to(device)

    # 4. Model Inference
    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    batch_all_logits = outputs.logits  # Shape: (batch_size, seq_len_model_input, vocab_size)

    # 5. Calculate mean entropies for each text in the batch
    mean_entropies_result = []
    for i in range(len(texts)):
        item_num_content_tokens = num_content_tokens_list[i]
        
        if item_num_content_tokens == 0: # Handle empty original texts
            mean_entropies_result.append(0.0)
            continue

        # Get logits for the current item, considering only the unpadded part
        # The model outputs logits for each position in batch_input_ids (up to padded length)
        item_logits = batch_all_logits[i]  # Shape: (seq_len_model_input, vocab_size)

        logits_to_consider_for_entropy = torch.empty((0, item_logits.shape[-1]), device=device)

        if has_bos_prepended:
            # If BOS was prepended, model input was [BOS, t1, ..., tN].
            # Logits item_logits[0] predicts t1 (given BOS).
            # Logits item_logits[N-1] predicts tN (given BOS, t1, ..., tN-1).
            # We need item_num_content_tokens logit vectors.
            end_idx_for_logits = item_num_content_tokens
            # Slice item_logits: take rows 0 to end_idx_for_logits-1
            # These correspond to predictions for each of the content tokens.
            logits_to_consider_for_entropy = item_logits[:end_idx_for_logits, :]
        else:  # No BOS prepended
            # Model input was [t1, ..., tN].
            # Logits item_logits[0] predicts t2 (given t1).
            # Logits item_logits[N-2] predicts tN (given t1, ..., tN-1).
            # We need item_num_content_tokens - 1 logit vectors.
            if item_num_content_tokens > 1:
                end_idx_for_logits = item_num_content_tokens - 1
                logits_to_consider_for_entropy = item_logits[:end_idx_for_logits, :]
            else:  # Single content token, no BOS: no conditional entropy can be computed this way
                mean_entropies_result.append(0.0)
                continue
        
        if logits_to_consider_for_entropy.nelement() > 0:
            probs = torch.softmax(logits_to_consider_for_entropy, dim=-1)
            # Add a small epsilon for numerical stability before log2
            entropies_tensor = -torch.sum(probs * torch.log2(probs.clamp_min(1e-9)), dim=-1)
            current_mean_entropy = torch.mean(entropies_tensor).item()
            mean_entropies_result.append(current_mean_entropy)
        else: # No logits were considered (e.g. single token without BOS, or already handled empty string)
            mean_entropies_result.append(0.0)
            
    return mean_entropies_result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate token entropies for generated texts.")
    parser.add_argument("--generated_text_path", type=str, required=True,
                        help="Path to the .jsonl file containing generated texts.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model directory.")
    parser.add_argument("--save_path", type=str, default="dsr_entropies_sequential.jsonl",
                        help="Path to save the output .jsonl file (default: dsr_entropies_sequential.jsonl).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda', 'cpu'). Autodetects if None (default: None).")
    parser.add_argument("--sample_size", type=int, default=256,
                        help="Number of samples to process from the input file (default: 256).")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed for sampling (default: 43).")
    parser.add_argument("--use_batched_version", action='store_true',
                        help="Use the batched version for entropy calculation.")


    args = parser.parse_args()

    # Use arguments
    generated_text_path = args.generated_text_path
    model_path = args.model_path
    save_path = args.save_path
    device_arg = args.device
    SAMPLE_SIZE = args.sample_size
    seed = args.seed

    random.seed(seed)

    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading data from {generated_text_path}...")
    data = read_jsonl_file(generated_text_path) # Ensure read_jsonl_file is defined/imported

    print(f"Loading model and tokenizer from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure pad_token is set for batching
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Set tokenizer.pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
            else:
                # A fallback, though most tokenizers should have eos or pad.
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer)) # Important if new token added
                print(f"Added new pad_token: {tokenizer.pad_token}")
        
        model.to(device)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        exit(1)

    SAMPLE_SIZE = 256
    samples = random.sample(data, SAMPLE_SIZE)

    if args.use_batched_version:
        all_generated_texts = [sample_item.get('generated_text', '') for sample_item in samples]

        print(f"Calculating mean entropies for {len(all_generated_texts)} texts in batch...")
        # Call the batched function
        all_calculated_mean_entropies = batched_mean_entropy(
            texts=all_generated_texts,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("Batch entropy calculation complete.")

        with open(save_path, "w", encoding="utf-8") as f:
            for i, sample_item in tqdm(enumerate(samples), desc="Saving batched results", total=len(samples)):
                prompt_text = sample_item.get('prompt', '')
                # Get the corresponding mean entropy from the batched calculation
                current_mean_entropy = all_calculated_mean_entropies[i] if i < len(all_calculated_mean_entropies) else 0.0
                
                metrics = {
                    "prompt": prompt_text,
                    "mean_entropy": current_mean_entropy
                }
                f.write(json.dumps(metrics) + "\n")
    else: 
        with open(save_path, "w", encoding="utf-8") as f:
            for i, sample_item in tqdm(enumerate(samples), desc="Saving results", total=len(samples)):
                prompt_text = sample_item.get('prompt', '')
                generated_text = sample_item.get('generated_text', '')

                _, avg_entropy = mean_entropy(generated_text=generated_text, model=model, tokenizer=tokenizer, device=device)
                
                # Since the request was "I only need mean entropies", 
                # final_token_entropies is not calculated/returned by the batched function.
                metrics = {
                    "prompt": prompt_text,
                    "mean_entropy": avg_entropy
                }
                f.write(json.dumps(metrics) + "\n")
    
    print(f"Results saved to {save_path}")
