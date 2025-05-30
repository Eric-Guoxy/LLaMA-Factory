from typing import List, Dict, Tuple, Union, Optional, Any # Added Any for flexibility
import torch.nn.functional as F # For F.log_softmax
from torch.nn.parallel import DistributedDataParallel as DDP # For DDP type hint
import argparse
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import glob
import importlib.util # Added for dynamic import
import sys # Added for sys.path manipulation

# --- Imports for dynamic generation ---
# Assuming __file__ is /home/inspur/cth/LLaMA-Factory/visualize/diff_prob_ddp.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))
llama_factory_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

if llama_factory_root_dir not in sys.path:
    sys.path.insert(0, llama_factory_root_dir)

try:
    from eval.generate_vllm import main as generate_vllm_main
except ImportError as e:
    print(f"Could not import generate_vllm.main: {e}")
    print("Please ensure LLaMA-Factory root is in PYTHONPATH or script is run from a location that allows the import.")
    generate_vllm_main = None # Placeholder if import fails
# --- End imports for dynamic generation ---

# #############################################
# # Helper Functions (mostly from visualize.py)
# #############################################

def read_jsonl_file(file_path: str) -> List[Dict]:
    """Reads a JSON Lines file and returns a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        # In DDP, rank 0 might handle this by exiting or raising an error that stops all processes.
        # For now, it returns empty; calling code (main_worker) should check.
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
    return data

def save_log_probs(log_probs: List[Dict[str, Union[str, float]]], output_file: str):
    """Saves token log probabilities to a JSON Lines file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in log_probs:
                f.write(json.dumps(entry) + '\\n')
    except Exception as e:
        print(f"Error saving log_probs to {output_file}: {e}")

def visualize_log_prob_differences_only_prob(
    prompt: str,
    final_model_name: str,
    ref_model_name: str,
    token_log_probs: List[Dict[str, Union[str, float]]], # Log probs from final_model
    response_log_probs: List[Dict[str, Union[str, float]]], # Log probs from ref_model
    output_file: Optional[str] = None,
) -> str:
    """Visualizes probability differences using color-coding (blue-white-red)."""
    if not token_log_probs or not response_log_probs:
        print(f"Warning: Empty log_probs for prompt '{prompt[:50]}...'. Skipping visualization.")
        return ""

    # Ensure tokens align if lengths are different (take min length)
    min_len = min(len(token_log_probs), len(response_log_probs))
    if min_len == 0:
        print(f"Warning: Zero min_len of tokens for prompt '{prompt[:50]}...'. Skipping visualization.")
        return ""

    token_log_probs_aligned = token_log_probs[:min_len]
    response_log_probs_aligned = response_log_probs[:min_len]
    
    tokens = [item["token"] for item in token_log_probs_aligned]
    gen_log_probs = [item["log_prob"] for item in token_log_probs_aligned]
    resp_log_probs = [item["log_prob"] for item in response_log_probs_aligned]

    gen_probs = [np.exp(log_p) for log_p in gen_log_probs]
    resp_probs = [np.exp(log_p) for log_p in resp_log_probs]
    diffs = [g - r for g, r in zip(gen_probs, resp_probs)]
    
    # Handle cases where all diffs are zero or very small to avoid division by zero
    abs_diffs = [abs(d) for d in diffs]
    max_abs_diff = max(abs_diffs) if abs_diffs else 0
    if max_abs_diff < 1e-9: # Effectively zero
        max_abs_diff = 1.0 # Avoid division by zero, treat as neutral

    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; padding: 20px; background-color: #f9f9f9; color: #333; }
    .text-container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; line-height: 1.8; font-size: 16px; overflow-wrap: break-word; word-wrap: break-word; }
    .token { display: inline; padding: 2px 1px; border-radius: 2px; margin: 0 1px; }
    .prompt-text { color: #555; }
    .legend { display: flex; margin: 10px 0 20px; justify-content: center; font-size: 14px; }
    .legend-item { margin: 0 15px; display: flex; align-items: center; }
    .color-box { width: 20px; height: 20px; margin-right: 5px; border-radius: 3px; border: 1px solid #ccc; }
    h1 { text-align: center; color: #444; margin-bottom: 25px; }
    .model-info-container { display: flex; justify-content: center; align-items: center; gap: 25px; padding: 12px; margin-bottom: 20px; background-color: #f0f0f0; border-radius: 5px; font-size: 14px; color: #555; flex-wrap: wrap; }
    </style>
    </head>
    <body>
    <h1>Token Probability Visualization</h1>
    <div class="model-info-container">
        <span><strong>Target Model:</strong> """+final_model_name+"""</span>
        <span><strong>Reference Model:</strong> """+ref_model_name+"""</span>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="color-box" style="background-color: rgba(100, 149, 237, 0.7);"></div><span>Lower Probability (Target vs. Ref)</span></div>
        <div class="legend-item"><div class="color-box" style="background-color: white;"></div><span>Similar Probability</span></div>
        <div class="legend-item"><div class="color-box" style="background-color: rgba(240, 128, 128, 0.7);"></div><span>Higher Probability (Target vs. Ref)</span></div>
    </div>
    <div class="text-container">
    <span class="prompt-text"><b>Prompt:</b> """ + prompt.replace("\\n", "<br>") + """</span><br><br><b>Response:</b><br>
    """
    for token, diff_val in zip(tokens, diffs):
        normalized_diff = diff_val / max_abs_diff if max_abs_diff != 0 else 0
        alpha = min(0.8, abs(normalized_diff) * 0.7 + 0.1) # Ensure some visibility, cap max alpha
        
        if normalized_diff > 0.05:  # Higher probability in final_model
            color = f"rgba(240, 128, 128, {alpha})"  # Light Red
        elif normalized_diff < -0.05:  # Lower probability in final_model
            color = f"rgba(100, 149, 237, {alpha})"  # Light Blue
        else:  # Similar probability
            color = "rgba(255, 255, 255, 0)" # Transparent background, effectively white or container background
        
        # Sanitize token for HTML display
        display_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\\n", "↵<br>").replace(" ", "&#32;")
        if display_token == "↵<br>": # Handle newlines better
             html += "<br>"
        elif display_token.isspace() and len(display_token) > 1 : # Multiple spaces
            html += "".join(["&nbsp;" for _ in display_token])
        else:
            html += f'<span class="token" style="background-color: {color}">{display_token}</span>'
    html += """
    </div>
    </body>
    </html>
    """
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            print(f"Error writing HTML to {output_file}: {e}")
    return html

# #############################################
# # DDP Setup and Cleanup
# #############################################

def setup_ddp(rank: int, world_size: int, master_port: str = "12355"):
    """Initializes the DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0: print(f"DDP setup complete for rank {rank}/{world_size} on GPU {torch.cuda.current_device()}.")

def cleanup_ddp():
    """Destroys the DDP process group."""
    dist.destroy_process_group()
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 : # Check if rank 0 before printing
         print("DDP cleanup complete.")

# #############################################
# # Core Logic Functions (DDP Adapted)
# #############################################

def batched_calculate_text_log_probs_ddp(
    model: DDP,
    tokenizer: AutoTokenizer,
    prompts: List[str],  # Shard for the current rank
    responses: List[str],  # Shard for the current rank
    processing_batch_size: int,
    rank: int
) -> List[List[Dict[str, Union[str, float]]]]:
    """
    Calculates log probabilities for response tokens using DDP.
    Processes input prompts/responses in chunks of `processing_batch_size`.
    """
    if not prompts or not responses:
        return [[] for _ in range(len(prompts))] # Return empty lists matching input structure
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses lists must have the same length.")

    num_samples_in_shard = len(prompts)
    all_log_probs_for_shard = [[] for _ in range(num_samples_in_shard)]
    
    # model.device is the cuda:rank for this DDP process
    device = model.device 
    model_config = model.module.config # Access config from the original model via .module

    # Disable tqdm if not rank 0 or if no samples
    use_tqdm_local = rank == 0 and num_samples_in_shard > 0
    
    outer_loop_desc = "Calculating log_probs (DDP)"
    if use_tqdm_local:
        pbar_outer = tqdm(range(0, num_samples_in_shard, processing_batch_size), desc=outer_loop_desc, position=0, leave=True)
    else:
        pbar_outer = range(0, num_samples_in_shard, processing_batch_size)

    for i in pbar_outer:
        batch_prompts = prompts[i : i + processing_batch_size]
        batch_responses = responses[i : i + processing_batch_size]
        
        if not batch_prompts: continue # Should not happen if loop range is correct

        # Tokenize prompts and full texts (prompt + response)
        # Pad to the longest sequence in the batch for prompts and full_texts separately
        tokenized_prompts = tokenizer(batch_prompts, padding=True, return_tensors="pt", truncation=True).to(device)
        
        batch_full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        tokenized_full_texts = tokenizer(batch_full_texts, padding=True, return_tensors="pt", truncation=True).to(device)

        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1)
        full_lengths = tokenized_full_texts.attention_mask.sum(dim=1)

        with torch.no_grad():
            # Get logits for the full texts
            outputs = model(input_ids=tokenized_full_texts.input_ids, attention_mask=tokenized_full_texts.attention_mask)
            logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

        # Calculate log probabilities
        log_probs_full_sequence = F.log_softmax(logits, dim=-1)

        for batch_idx in range(len(batch_prompts)):
            current_sample_log_probs = []
            # Response tokens start after prompt tokens.
            # Log prob for token T is taken from position T-1's logits.
            # So, for the first response token (at index `prompt_lengths[batch_idx]`),
            # we need logits from `prompt_lengths[batch_idx] - 1`.
            # Loop from `prompt_lengths[batch_idx]` up to `full_lengths[batch_idx] -1` for input IDs
            # to get log_probs for tokens from `prompt_lengths[batch_idx]+1` up to `full_lengths[batch_idx]`
            
            # Iterate over token positions in the response part of the sequence
            for seq_pos in range(prompt_lengths[batch_idx].item(), full_lengths[batch_idx].item()):
                # The token whose probability we are calculating is at tokenized_full_texts.input_ids[batch_idx, seq_pos]
                # The logits used to predict this token are at logits[batch_idx, seq_pos - 1, :]
                # However, we use log_probs_full_sequence which is already aligned with logits.
                # So, log_prob for token at `seq_pos` is `log_probs_full_sequence[batch_idx, seq_pos -1, token_id_at_seq_pos]`
                
                # This logic was slightly off in some HF impls. Correct:
                # Log prob of token `t_k` given `t_1...t_{k-1}` is `log_softmax(logits_for_t_k)[id(t_k)]`
                # `logits_for_t_k` are `outputs.logits[batch, k-1, :]`.
                # `id(t_k)` is `tokenized_full_texts.input_ids[batch, k]`.
                
                # So, for a token at `tokenized_full_texts.input_ids[batch_idx, seq_idx]`,
                # its log probability is `log_probs_full_sequence[batch_idx, seq_idx - 1, tokenized_full_texts.input_ids[batch_idx, seq_idx]]`
                # This applies for `seq_idx` from `prompt_lengths[batch_idx]` to `full_lengths[batch_idx] - 1`.

                if seq_pos == 0: # Cannot get log_prob for the very first token if it's part of response (empty prompt)
                    continue

                token_id = tokenized_full_texts.input_ids[batch_idx, seq_pos].item()
                log_prob_value = log_probs_full_sequence[batch_idx, seq_pos - 1, token_id].item()
                
                # Decode token for storage (careful with special tokens if not desired)
                # Using `decode` on single token ID might add prefixes like 'Ġ' for SentencePiece.
                # It's often better to decode the whole sequence and then segment if exact token strings are critical.
                # For simplicity here, decode individual token_id.
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)

                current_sample_log_probs.append({
                    "token": token_str,
                    "token_id": token_id,
                    "log_prob": log_prob_value
                })
            all_log_probs_for_shard[i + batch_idx] = current_sample_log_probs
    
    if use_tqdm_local:
        pbar_outer.close()

    return all_log_probs_for_shard


def pipeline_ddp(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    prompts_full: List[str],
    responses_full: List[str],
    correctness_full: List[Optional[Union[str, float, bool]]], # Added correctness_full
    all_final_model_log_probs_full_gathered: List[List[Dict[str, Union[str, float]]]]
):
    """
    Main pipeline for DDP: loads ref_model, calculates its log_probs, and rank 0 visualizes/saves.
    `all_final_model_log_probs_full_gathered` is assumed to be already computed and gathered on rank 0.
    `correctness_full` contains correctness data for each sample, available on rank 0.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.ref_model_path, 
        trust_remote_code=True,
        use_fast=getattr(args, 'use_fast_tokenizer', True)
    )

    if rank == 0: print(f"Loading reference model: {args.ref_model_path} for DDP log prob calculation.")
    # Load model on the current rank's device
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16, # or torch.float32
        trust_remote_code=True,
        # device_map={"": rank} # Not needed if .to(rank) is used before DDP
    ).to(rank)
    ref_model = DDP(ref_model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters)
    ref_model.eval()

    # Shard data for ref_model log_prob calculation
    # Each rank processes its slice of the full dataset
    prompts_shard = prompts_full[rank::world_size]
    responses_shard = responses_full[rank::world_size]

    if rank == 0: print(f"Calculating log_probs with ref_model ({args.ref_model_path}) across {world_size} GPUs...")
    
    all_ref_model_log_probs_shard = batched_calculate_text_log_probs_ddp(
        model=ref_model, tokenizer=tokenizer,
        prompts=prompts_shard, responses=responses_shard,
        processing_batch_size=args.hf_processing_batch_size, rank=rank
    )

    # Gather results for all_ref_model_log_probs to rank 0
    gathered_ref_log_probs_obj = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered_ref_log_probs_obj, all_ref_model_log_probs_shard)
    else:
        gathered_ref_log_probs_obj = [all_ref_model_log_probs_shard]
    
    # Rank 0 performs visualization and saving
    if rank == 0:
        print("Rank 0: Gathering and processing results for reference model.")
        all_ref_model_log_probs_full_gathered = []
        for i in range(len(prompts_full)):
            worker_idx = i % world_size
            item_idx_in_worker_shard = i // world_size
            if worker_idx < len(gathered_ref_log_probs_obj) and \
               item_idx_in_worker_shard < len(gathered_ref_log_probs_obj[worker_idx]):
                all_ref_model_log_probs_full_gathered.append(gathered_ref_log_probs_obj[worker_idx][item_idx_in_worker_shard])
            else:
                 all_ref_model_log_probs_full_gathered.append([])

        print(f"Finished calculating and gathering ref_model log_probs. Total samples: {len(all_ref_model_log_probs_full_gathered)}")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Created save directory: {args.save_dir}")

        final_model_name_sanitized = args.final_model_name.replace("/", "_").replace("\\\\","_")
        ref_model_name_sanitized = args.ref_model_name.replace("/", "_").replace("\\\\","_")

        num_samples_to_process = len(prompts_full)
        print(f"Generating visualizations for the first {args.num_visualization_examples} samples and prob diff data for all {num_samples_to_process} samples...")

        all_samples_prob_diff_data = []

        for i in tqdm(range(num_samples_to_process), desc="Rank 0: Processing samples for viz/data"):
            prompt_text = prompts_full[i]
            correctness_value = correctness_full[i] # Get correctness for the current sample
            
            final_model_lp_sample = all_final_model_log_probs_full_gathered[i]
            ref_model_lp_sample = all_ref_model_log_probs_full_gathered[i]

            sample_prob_diff_entry = {
                "prompt": prompt_text,
                "correctness": correctness_value, # Add correctness to the entry
                "final_model_name": args.final_model_name,
                "ref_model_name": args.ref_model_name,
                "token_analysis": []
            }

            if final_model_lp_sample and ref_model_lp_sample:
                min_len = min(len(final_model_lp_sample), len(ref_model_lp_sample))
                if min_len > 0:
                    tokens = [item["token"] for item in final_model_lp_sample[:min_len]]
                    final_log_probs = [item["log_prob"] for item in final_model_lp_sample[:min_len]]
                    ref_log_probs = [item["log_prob"] for item in ref_model_lp_sample[:min_len]]

                    final_probs = [np.exp(lp) for lp in final_log_probs]
                    ref_probs = [np.exp(lp) for lp in ref_log_probs]
                    
                    for token_idx in range(min_len):
                        token_str = tokens[token_idx]
                        fp = final_probs[token_idx]
                        rp = ref_probs[token_idx]
                        diff = fp - rp
                        sample_prob_diff_entry["token_analysis"].append({
                            "token": token_str,
                            "final_model_prob": float(fp),
                            "ref_model_prob": float(rp),
                            "prob_difference": float(diff)
                        })
            all_samples_prob_diff_data.append(sample_prob_diff_entry)

            # Generate HTML visualization only for the specified number of examples
            if i < args.num_visualization_examples:
                output_html_filename = os.path.join(
                    args.save_dir,
                    f"prob_diff_target_{final_model_name_sanitized}_vs_ref_{ref_model_name_sanitized}_sample_{i}.html"
                )
                
                visualize_log_prob_differences_only_prob(
                    prompt=prompt_text,
                    final_model_name=args.final_model_name,
                    ref_model_name=args.ref_model_name,
                    token_log_probs=final_model_lp_sample,
                    response_log_probs=ref_model_lp_sample,
                    output_file=output_html_filename
                )
        
        if args.num_visualization_examples > 0 and args.num_visualization_examples <= num_samples_to_process :
             print(f"Visualizations for the first {args.num_visualization_examples} samples saved to {args.save_dir}")
        elif args.num_visualization_examples > num_samples_to_process :
             print(f"Visualizations for all {num_samples_to_process} samples saved to {args.save_dir}")


        # Save the collected probability difference data
        prob_diff_data_output_filename = os.path.join(
            args.save_dir,
            f"prob_diff_details_target_{final_model_name_sanitized}_vs_ref_{ref_model_name_sanitized}.jsonl"
        )
        try:
            with open(prob_diff_data_output_filename, 'w', encoding='utf-8') as f_jsonl:
                for entry in all_samples_prob_diff_data:
                    f_jsonl.write(json.dumps(entry) + '\\n')
            print(f"Detailed probability differences saved to {prob_diff_data_output_filename}")
        except Exception as e:
            print(f"Error saving probability difference data to {prob_diff_data_output_filename}: {e}")

    # Synchronize all processes before cleanup, especially if rank 0 was doing I/O
    if world_size > 1:
        dist.barrier()


# #############################################
# # Main DDP Worker and Entry Point
# #############################################

def main_worker(rank, world_size, args):
    """Main worker function for each DDP process."""
    setup_ddp(rank, world_size, args.master_port)
    
    # Expand user paths for data loading (already done for args.data_path before spawn)
    # Tokenizer path and model paths are handled by from_pretrained
    
    # Load tokenizer (same for all ranks, used by both models)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.final_model_path, 
        trust_remote_code=True,
        use_fast=getattr(args, 'use_fast_tokenizer', True)
    )

    # Rank 0 loads data and prepares it for broadcasting
    data_to_broadcast = {}
    if rank == 0:
        print(f"Rank 0: Loading data from {args.data_path}") # args.data_path is already absolute
        raw_data = read_jsonl_file(args.data_path) # read_jsonl_file handles os.path.expanduser internally
        if not raw_data:
            print(f"Rank 0: No data loaded from {args.data_path}. Exiting.")
            data_to_broadcast = {"prompts": [], "responses": [], "correctness": [], "error": True}
        else:
            prompts_full = [item[args.prompt_column] for item in raw_data]
            responses_full = [item[args.response_column] for item in raw_data]
            correctness_full = [item.get(args.correctness_column) for item in raw_data] # Get correctness
            data_to_broadcast = {
                "prompts": prompts_full, 
                "responses": responses_full, 
                "correctness": correctness_full, # Add correctness to broadcast
                "error": False
            }
            print(f"Rank 0: Loaded {len(prompts_full)} samples.")

    # Broadcast data object from rank 0 to all other ranks
    if world_size > 1:
        object_list = [data_to_broadcast if rank == 0 else None for _ in range(world_size)]
        dist.broadcast_object_list(object_list, src=0)
        if rank != 0:
            data_to_broadcast = object_list[0] 

    if data_to_broadcast.get("error", False) and rank ==0:
        print("Error in data loading on Rank 0. Aborting.")
        cleanup_ddp()
        return
    elif data_to_broadcast.get("error", False):
        cleanup_ddp()
        return

    prompts_full = data_to_broadcast["prompts"]
    responses_full = data_to_broadcast["responses"]
    correctness_full = data_to_broadcast["correctness"] # Retrieve correctness

    if not prompts_full:
        if rank == 0: print("No data to process after broadcast. Exiting.")
        cleanup_ddp()
        return

    # --- Calculate log_probs for the "final" or "target" model ---
    if rank == 0: print(f"Loading final/target model: {args.final_model_path} for DDP log prob calculation.")
    final_model = AutoModelForCausalLM.from_pretrained(
        args.final_model_path,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
        trust_remote_code=True,
    ).to(rank)
    final_model = DDP(final_model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters)
    final_model.eval()

    prompts_shard = prompts_full[rank::world_size]
    responses_shard = responses_full[rank::world_size]
    
    if rank == 0: print(f"Calculating log_probs with final/target model ({args.final_model_path}) across {world_size} GPUs...")
    
    all_final_model_log_probs_shard = batched_calculate_text_log_probs_ddp(
        model=final_model, tokenizer=tokenizer,
        prompts=prompts_shard, responses=responses_shard,
        processing_batch_size=args.hf_processing_batch_size, rank=rank
    )

    gathered_final_log_probs_obj = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered_final_log_probs_obj, all_final_model_log_probs_shard)
    else:
        gathered_final_log_probs_obj = [all_final_model_log_probs_shard]

    all_final_model_log_probs_full_gathered = []
    if rank == 0:
        print("Rank 0: Gathering and processing results for final/target model.")
        for i in range(len(prompts_full)):
            worker_idx = i % world_size
            item_idx_in_worker_shard = i // world_size
            if worker_idx < len(gathered_final_log_probs_obj) and \
               item_idx_in_worker_shard < len(gathered_final_log_probs_obj[worker_idx]):
                all_final_model_log_probs_full_gathered.append(gathered_final_log_probs_obj[worker_idx][item_idx_in_worker_shard])
            else:
                all_final_model_log_probs_full_gathered.append([])
        print(f"Finished calculating and gathering final/target_model log_probs. Total samples: {len(all_final_model_log_probs_full_gathered)}")

    # Clean up final_model to free GPU memory before loading ref_model, if memory is tight.
    del final_model 
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

    pipeline_ddp(
        rank=rank, world_size=world_size, args=args,
        prompts_full=prompts_full,
        responses_full=responses_full,
        correctness_full=correctness_full, # Pass correctness here
        all_final_model_log_probs_full_gathered=all_final_model_log_probs_full_gathered
    )
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize log probability differences using DDP.")
    parser.add_argument("--final_model_path", type=str, required=True, help="Path to the final/target model.")
    parser.add_argument("--ref_model_path", type=str, required=True, help="Path to the reference model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (if different from model path).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the .jsonl data file. If not found, attempts to generate it if --gen_input_path is provided.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts in the JSONL file.")
    parser.add_argument("--response_column", type=str, default="generated_text", help="Column name for responses in the JSONL file.")
    parser.add_argument("--correctness_column", type=str, default="correctness", help="Column name for correctness data in the JSONL file (optional).")
    parser.add_argument("--save_dir", type=str, default="./vis_results_ddp", help="Directory to save visualizations and potentially generated data.")
    parser.add_argument("--final_model_name", type=str, default="FinalModel", help="Name for the final model in visualizations.")
    parser.add_argument("--ref_model_name", type=str, default="ReferenceModel", help="Name for the reference model in visualizations.")
    parser.add_argument("--hf_processing_batch_size", type=int, default=8, help="Batch size for Hugging Face log prob calculation.")
    parser.add_argument("--num_visualization_examples", type=int, default=10, help="Number of samples to generate visualizations for.")
    parser.add_argument("--master_port", type=str, default="12355", help="Master port for DDP initialization.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model precision (bf16, fp16, fp32).")
    parser.add_argument("--find_unused_parameters", action='store_true', help="Set find_unused_parameters=True for DDP.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="Whether to use fast tokenizer if available.")

    # Arguments for generating data_path file if it doesn't exist
    gen_group = parser.add_argument_group('Generation Arguments (for auto-creating data_path)')
    gen_group.add_argument("--gen_script_path", type=str, default="../eval/generate_vllm.py", help="Path to the generate_vllm.py script, relative to this script's directory or absolute.")
    gen_group.add_argument("--gen_input_path", type=str, default=None, help="Path to the input Parquet dataset for generate_vllm.py (e.g., valid_all.parquet). Required if data_path is to be auto-generated.")
    gen_group.add_argument("--gen_model_path", type=str, default=None, help="Model path for generate_vllm.py. Defaults to --final_model_path.")
    gen_group.add_argument("--gen_tokenizer_path", type=str, default=None, help="Tokenizer path for generate_vllm.py. Defaults to --tokenizer_path, then --gen_model_path, then --final_model_path.")
    gen_group.add_argument("--gen_template", type=str, default="own", help="Template for generate_vllm.py. E.g., 'own', 'qwen', 'oat'.")
    gen_group.add_argument("--gen_max_new_tokens", type=int, default=8192, help="max_tokens for generate_vllm.py (maps to its max_tokens).")
    gen_group.add_argument("--gen_top_p", type=float, default=1.0, help="top_p for generate_vllm.py.")
    gen_group.add_argument("--gen_temperature", type=float, default=0.6, help="temperature for generate_vllm.py.")
    gen_group.add_argument("--gen_tp_size", type=int, default=1, help="Tensor parallel size for generate_vllm.py.")
    gen_group.add_argument("--gen_overwrite_output", action="store_true", help="Allow generate_vllm.py to overwrite its own intermediate output file (maps to its force_generate=True).")
    gen_group.add_argument("--gen_remove_system", action="store_true", default=True, help="remove_system flag for generate_vllm.py (default: True).")
    gen_group.add_argument("--gen_no_remove_system", action="store_false", dest="gen_remove_system", help="Explicitly do not remove system prompt for generate_vllm.py (sets remove_system=False).")
    gen_group.add_argument("--gen_n", type=int, default=1, help="Number of sequences to generate per prompt for generate_vllm.py.")
    gen_group.add_argument("--gen_add_think_before_answer", action="store_true", help="add_think_before_answer flag for generate_vllm.py.")
    gen_group.add_argument("--gen_add_oat_evaluate", action="store_true", help="add_oat_evaluate flag for generate_vllm.py.")
    gen_group.add_argument("--gen_skip_scoring", action="store_true", default=False, help="skip_scoring flag for generate_vllm.py. If True, correctness column might be missing. Overridden to False if --correctness_column is set for diff_prob_ddp.")
    gen_group.add_argument("--gen_no_split_think", action="store_true", help="no_split_think flag for generate_vllm.py.")

    args = parser.parse_args()

    # Resolve paths and prepare for potential generation
    args.final_model_path = os.path.abspath(os.path.expanduser(args.final_model_path))
    args.ref_model_path = os.path.abspath(os.path.expanduser(args.ref_model_path))
    if args.tokenizer_path:
        args.tokenizer_path = os.path.abspath(os.path.expanduser(args.tokenizer_path))
    args.data_path = os.path.abspath(os.path.expanduser(args.data_path))
    args.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    if args.gen_input_path:
        args.gen_input_path = os.path.abspath(os.path.expanduser(args.gen_input_path))
    if args.gen_model_path:
        args.gen_model_path = os.path.abspath(os.path.expanduser(args.gen_model_path))
    if args.gen_tokenizer_path:
        args.gen_tokenizer_path = os.path.abspath(os.path.expanduser(args.gen_tokenizer_path))

    if not os.path.isabs(args.gen_script_path):
        args.gen_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.gen_script_path))
    else:
        args.gen_script_path = os.path.abspath(os.path.expanduser(args.gen_script_path))

    os.makedirs(args.save_dir, exist_ok=True)

    generate_vllm_main_func = None
    if os.path.exists(args.gen_script_path):
        original_sys_path = list(sys.path)
        gen_script_module_dir = os.path.dirname(args.gen_script_path)
        project_root_for_gen_script = os.path.dirname(gen_script_module_dir)
        try:
            sys.path.insert(0, project_root_for_gen_script) 
            sys.path.insert(0, gen_script_module_dir)
            
            spec = importlib.util.spec_from_file_location("generate_vllm_module", args.gen_script_path)
            if spec and spec.loader:
                generate_vllm_module = importlib.util.module_from_spec(spec)
                sys.modules["generate_vllm_module"] = generate_vllm_module
                spec.loader.exec_module(generate_vllm_module)
                generate_vllm_main_func = generate_vllm_module.main
                print(f"Successfully imported main from {args.gen_script_path}")
            else:
                print(f"Warning: Could not create spec or loader for {args.gen_script_path}")
        except ImportError as e:
            print(f"Warning: Could not import main from {args.gen_script_path} due to ImportError: {e}")
            print(f"Current sys.path during import attempt: {sys.path}")
        except Exception as e:
            print(f"Warning: Could not import main from {args.gen_script_path} due to an unexpected error: {e}")
        finally:
            sys.path = original_sys_path
    else:
        print(f"Warning: Generation script path {args.gen_script_path} does not exist. Automatic data generation will not be possible.")

    if not os.path.exists(args.data_path):
        if args.gen_input_path and generate_vllm_main_func:
            print(f"Data file not found at {args.data_path}. Attempting to generate it using {args.gen_script_path}...")
            
            actual_gen_model_path = args.gen_model_path if args.gen_model_path else args.final_model_path
            actual_gen_tokenizer_path = args.gen_tokenizer_path if args.gen_tokenizer_path else \
                                       (args.tokenizer_path if args.tokenizer_path else actual_gen_model_path)

            gen_params = {
                "input_file": args.gen_input_path,
                "output_file": args.data_path,
                "model_path": actual_gen_model_path,
                "tokenizer_path": actual_gen_tokenizer_path,
                "remove_system": args.gen_remove_system,
                "template": args.gen_template,
                "temperature": args.gen_temperature,
                "top_p": args.gen_top_p,
                "max_tokens": args.gen_max_new_tokens,
                "n": args.gen_n,
                "force_generate": args.gen_overwrite_output,
                "add_think_before_answer": args.gen_add_think_before_answer,
                "add_oat_evaluate": args.gen_add_oat_evaluate,
                "skip_scoring": args.gen_skip_scoring,
                "no_split_think": args.gen_no_split_think,
                "tensor_parallel_size": args.gen_tp_size,
            }

            if args.correctness_column and args.correctness_column.strip():
                if gen_params["skip_scoring"]:
                    print(f"Warning: --correctness_column ('{args.correctness_column}') is set, "
                          f"but --gen_skip_scoring was also True. Overriding gen_skip_scoring to False for generation.")
                gen_params["skip_scoring"] = False
            
            print(f"Calling {args.gen_script_path} main function with parameters:")
            for key, value in gen_params.items():
                print(f"  {key}: {value}")

            try:
                os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
                generate_vllm_main_func(**gen_params)
                print(f"Successfully generated data at {args.data_path}.")
                if not os.path.exists(args.data_path):
                    potential_decoded_file = args.data_path.replace('.jsonl', '') + '.decoded.jsonl'
                    if os.path.exists(potential_decoded_file):
                         print(f"Note: The primary output file {args.data_path} was not found, "
                               f"but its .decoded version {potential_decoded_file} exists. "
                               "This might indicate an issue in the scoring/final saving step of generate_vllm.py. "
                               "diff_prob_ddp.py will proceed with the main data_path if it now exists, or fail if it's still missing.")
                    else:
                        print(f"Error: Generation script seemed to complete, but the output file {args.data_path} was not created, "
                              f"and its intermediate .decoded.jsonl version was also not found.")
                    raise RuntimeError(f"Failed to generate data file {args.data_path}: File not found after generation call.")

            except Exception as e:
                print(f"Error during data generation with {args.gen_script_path}: {e}")
                potential_decoded_file = args.data_path.replace('.jsonl', '') + '.decoded.jsonl'
                if os.path.exists(potential_decoded_file):
                    print(f"An intermediate decoded file was found: {potential_decoded_file}. "
                          "This might contain the generated texts but not the full scored output. "
                          "The error likely occurred during the scoring or final saving phase in generate_vllm.py.")
                raise RuntimeError(f"Failed to generate data file {args.data_path} using {args.gen_script_path}") from e

        elif not args.gen_input_path and generate_vllm_main_func:
            print(f"Data file not found at {args.data_path}, but --gen_input_path (for the Parquet source) is not provided. Cannot auto-generate.")
            raise FileNotFoundError(f"Required data file {args.data_path} not found and --gen_input_path missing.")
        elif not generate_vllm_main_func:
            print(f"Data file not found at {args.data_path}, and the generation script's main function could not be loaded from {args.gen_script_path}. Cannot auto-generate.")
            raise FileNotFoundError(f"Required data file {args.data_path} not found and generation script main function unavailable/unloadable from {args.gen_script_path}.")
        else: 
             raise FileNotFoundError(f"Required data file {args.data_path} not found and unable to determine generation possibility due to an unexpected state.")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} still not found after checking/attempting generation. Cannot proceed.")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices found. DDP requires CUDA.")
    print(f"Found {world_size} CUDA devices.")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

