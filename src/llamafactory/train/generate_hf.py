import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, GenerationConfig
import torch
import logging

from math_verify import parse, verify 
from typing import Optional
import os

# Setup logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed\{\} for closed-form results like multiple choices or mathematical solutions. "
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def generate_with_hf_model(
    model,  # PyTorch model (the one from DeepSpeed, on Rank 0's GPU)
    tokenizer, # Hugging Face tokenizer
    prompts: list[str], # List of fully formatted prompt strings
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    # Add other necessary HF generate params if needed
):
    device = model.device
    logger.info(f"Generating with Hugging Face model on device: {device} for {len(prompts)} prompts.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Tokenizer pad_token_id was None, set to eos_token_id ({tokenizer.eos_token_id}).")

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Determine if sampling should be used based on temperature and top_p
    # Transformers' generate enables sampling if temperature > 0.0 or top_p < 1.0, etc.
    # and do_sample is True (which is default if temp/top_p suggest sampling)
    do_sample_flag = False
    if temperature > 0.0 and temperature != 1.0: # Explicit 0.0 for greedy, 1.0 is often default
        do_sample_flag = True
    if top_p is not None and top_p < 1.0 and top_p > 0.0: # Explicit 1.0 for no top_p
        do_sample_flag = True
    
    # If temperature is 0, it implies greedy, so do_sample should be False.
    if temperature == 0.0:
        do_sample_flag = False


    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample_flag else None, # TF uses temp only if do_sample=True
        top_p=top_p if do_sample_flag else None,             # TF uses top_p only if do_sample=True
        do_sample=do_sample_flag,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Add other relevant parameters from your sampling_params if needed
    )
    
    hf_outputs = []
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config
        )

        # Decode generated sequences (slicing off the prompt part)
        for i in range(len(prompts)):
            # Get the length of the input prompt for slicing
            # input_ids may be padded, so count non-pad tokens for actual length
            prompt_length = inputs.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
            
            generated_tokens = output_sequences[i][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Construct an output object similar to what the test function expects from vLLM
            # vLLM output: RequestOutput(prompt, outputs=[CompletionOutput(text)])
            class MockCompletionOutput:
                def __init__(self, text):
                    self.text = text
            
            class MockRequestOutput:
                def __init__(self, prompt_text, completion_text):
                    self.prompt = prompt_text # The formatted prompt string
                    self.outputs = [MockCompletionOutput(completion_text)]

            hf_outputs.append(MockRequestOutput(prompts[i], generated_text))
            if i == 0: # Log first generated sample
                 logger.info(f"Sample generated text (decoded): {generated_text[:500]}...")
                 
    return hf_outputs

def test( # Renamed from previous `test` that called vLLM
    model, # PyTorch model from callback
    tokenizer, # HF tokenizer from callback
    # Paths below are mostly for reference or if some part still needs them,
    # but model and tokenizer objects are primary now.
    base_model_path: str, 
    current_lora_adapter_path: Optional[str],
    tokenizer_path_for_vllm: str, # Can be used to confirm tokenizer identity
    input_file: str, 
    output_file: Optional[str], 
    debug: bool = False, 
    remove_system: bool = True, 
    template: str = 'own', 
    temperature: float = 0.6, 
    top_p: float = 1.0, # HF default for top_p is often 1.0 if not sampling
    max_tokens: int = 8192 # This will be max_new_tokens
):
    logger.info(f"Starting PyTorch-based test. Input: {input_file}, Template: {template}")
    
    df = pd.read_parquet(input_file)
    # raw_messages_from_file is a list of lists of dicts (chat format)
    raw_messages_from_file = df['prompt'].tolist() 
    
    answers_data = df['reward_model'].tolist()
    # Ensure 'ground_truth' exists, provide default if not
    answers = [ad.get('ground_truth', '') if isinstance(ad, dict) else '' for ad in answers_data]

    data_sources = df['data_source'].tolist()
    assert len(raw_messages_from_file) == len(answers) == len(data_sources), \
        "Mismatch in lengths of messages, answers, or data_sources."

    # Prepare prompts for the model
    prompts_for_generation = []
    original_prompts_for_saving = [] # To store the version before system message removal

    # Handle system message removal and template application
    messages_to_format = raw_messages_from_file
    if remove_system:
        logger.info('Attempting to remove system messages for prompt formatting.')
        processed_messages_list = []
        assert raw_messages_from_file[0][0]['role'] == 'system'
        for msg_list in raw_messages_from_file:
            processed_messages_list.append(msg_list[1:])
        messages_to_format = processed_messages_list
        logger.info('System message removal step completed for formatting.')
    
    original_prompts_for_saving = messages_to_format # Save this version

    for i, chat_messages in enumerate(messages_to_format):
        formatted_prompt_str = ""
        try:
            if template == 'own':
                formatted_prompt_str = tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            elif template == 'qwen':
                # Qwen template expects the first user question string
                question_content = ""
                if chat_messages and isinstance(chat_messages, list) and chat_messages[0]:
                    question_content = chat_messages[0].get('content', "")
                formatted_prompt_str = apply_qwen_math_template(question_content)
            elif template == 'prime':
                question_content = ""
                if chat_messages and isinstance(chat_messages, list) and chat_messages[0]:
                    question_content = chat_messages[0].get('content', "")
                formatted_prompt_str = make_conv_zero(question_content)
            elif template == 'no': # Raw content of first message
                if chat_messages and isinstance(chat_messages, list) and chat_messages[0]:
                     formatted_prompt_str = chat_messages[0].get('content', "")
                else:
                    formatted_prompt_str = str(chat_messages) # Fallback
            else: 
                logger.error(f'Invalid template for generation: {template}')
                continue 
        except Exception as e:
            logger.error(f"Error formatting prompt for message index {i} with template {template}: {e}", exc_info=True)
            continue
        prompts_for_generation.append(formatted_prompt_str)

    if not prompts_for_generation:
        logger.warning("No valid prompts were generated for the Hugging Face model.")
        return {"error": "No prompts to generate after formatting."}
    
    if debug and prompts_for_generation:
        logger.info(f"Example formatted prompt for HF model (template: {template}): {prompts_for_generation[0][:500]}...")

    # Call the new PyTorch-based generation function
    hf_generation_outputs = generate_with_hf_model(
        model,
        tokenizer,
        prompts_for_generation, # List of formatted strings
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens # Renamed from max_tokens for clarity
    )
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg_correct = 0

    if not hf_generation_outputs:
        logger.warning("generate_with_hf_model returned no outputs.")
        return {"warning": "No outputs from Hugging Face model generation."}

    for i, output_item in enumerate(hf_generation_outputs): # output_item is MockRequestOutput
        if i >= len(answers) or i >= len(data_sources) or i >= len(original_prompts_for_saving):
            logger.warning(f"Output index {i} is out of bounds for answers/data_sources/original_prompts. Skipping.")
            continue
        
        # output_item.prompt is the string that was fed to model.generate
        # original_prompts_for_saving[i] is the chat structure before final string formatting
        prompt_to_save = original_prompts_for_saving[i] 
        
        generated_text = output_item.outputs[0].text # From MockRequestOutput
        answer = answers[i]
        
        # THOUGHT_DELIMITER logic (applied to the generated_text)
        # This assumes the model itself generates these delimiters.
        if output_item.prompt.endswith(THOUGHT_DELIMITER_START+'\n'): # Check against the formatted prompt
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            
        if THOUGHT_DELIMITER_START in generated_text and THOUGHT_DELIMITER_END in generated_text:
            parts = generated_text.split(THOUGHT_DELIMITER_END, 1)
            if len(parts) > 1:
                generated_text = parts[1] # Text after </think>
            else:
                logger.warning(f"Could not properly split thought from: {generated_text}")
        
        labels = labeling_responses([generated_text], answer) # labeling_responses expects a list
        
        current_data_source = data_sources[i]
        rets[current_data_source].append(labels[0])
        
        save_data.append({
            'prompt': prompt_to_save, # Original chat structure (post system msg removal)
            'formatted_prompt_string': output_item.prompt, # The string fed to the model
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg_correct += 1
    
    metrics = {}
    accs = []    
    for data_source_key, label_list in rets.items():
        if not label_list: continue
        acc = np.array(label_list).mean()
        accs.append(acc)
        metrics[str(data_source_key)] = float(acc) # Ensure key is string for JSON

    if accs:
        metrics['accuracy_mean_overall'] = float(np.mean(accs))
    else:
        metrics['accuracy_mean_overall'] = 0.0
    
    if output_file:
        try:
            # Ensure output_file path exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed PyTorch test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save detailed PyTorch test results to {output_file}: {e}", exc_info=True)

    logger.info(f"Calculated PyTorch test metrics: {metrics}")
    return metrics