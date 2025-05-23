#export HF_ENDPOINT=https://hf-mirror.com  
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import logging

from math_verify import parse, verify

# Setup logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if run within LlamaFactory
    logging.basicConfig(level=logging.INFO)

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    # print(golden_answers)
    # print(predict_answers)
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

def test(base_model_path, current_lora_adapter_path, tokenizer_path_for_vllm, 
         input_file, output_file, debug=False, remove_system=True, 
         template='own', temperature=0.6, top_p=1.0, max_tokens=8192):
    logger.info(f"Starting test function with base_model_path: {base_model_path}, "
                f"current_lora_adapter_path: {current_lora_adapter_path}, "
                f"tokenizer_path_for_vllm: {tokenizer_path_for_vllm}")
    # 数据处理
    df = pd.read_parquet(input_file)
    messages = df['prompt'].tolist()
    
    if remove_system:
        logger.info('Attempting to remove system messages.')
        assert messages[0][0]['role'] == 'system'
        messages = [message[1:] for message in messages]
        logger.info('System messages removed.')

    answers = df['reward_model'].tolist()
    answers = [answer['ground_truth'] for answer in answers]
    assert len(messages) == len(answers), "Mismatch between number of messages and answers."
    data_sources = df['data_source'].tolist()
            
    if messages:
        logger.info(f"Sample message structure for vLLM: {messages[0]}")
    else:
        logger.warning("No messages to process for vLLM test.")
        return {"error": "No messages provided for testing."}

    # Updated call to generate_vllm
    outputs = generate_vllm(
        messages, 
        base_model_path, 
        current_lora_adapter_path,
        tokenizer_path_for_vllm,
        template=template, 
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_tokens
    )
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg_correct = 0 # Renamed from avg to avoid confusion

    if not outputs:
        logger.warning("generate_vllm returned no outputs.")
        return {"warning": "No outputs from vLLM generation."}

    for i, output_item in enumerate(outputs):
        if i >= len(answers) or i >= len(data_sources):
            logger.warning(f"Output index {i} is out of bounds for answers/data_sources. Skipping.")
            continue

        prompt = output_item.prompt
        generated_text = output_item.outputs[0].text
        answer = answers[i]

        if prompt.endswith(THOUGHT_DELIMITER_START+'\n'):
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            
        if THOUGHT_DELIMITER_START in generated_text and THOUGHT_DELIMITER_END in generated_text:
            parts = generated_text.split(THOUGHT_DELIMITER_END, 1)
            if len(parts) > 1:
                generated_text = parts[1]
            else:
                logger.warning(f"Could not properly split thought from: {generated_text}")
        
        labels = labeling_responses([generated_text,], answer)
        
        rets[data_sources[i]].append(labels[0])
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg_correct += 1
    
    metrics = {}
    accs = []    
    for data_source, label_list in rets.items(): # Renamed labels to label_list
        if not label_list: continue
        acc = np.array(label_list).mean()
        accs.append(acc)
        metrics[data_source] = float(acc) # Ensure JSON serializable

    if accs:
        metrics['accuracy_mean_overall'] = float(np.mean(accs)) # Ensure JSON serializable
    else:
        metrics['accuracy_mean_overall'] = 0.0
    
    # Save detailed results if output_file is specified
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save detailed test results to {output_file}: {e}")

    logger.info(f"Calculated test metrics: {metrics}")
    return metrics

# Modified generate_vllm function signature
def generate_vllm(messages, base_model_path, current_lora_adapter_path, tokenizer_path_for_vllm,
                  template='own', temperature=0.6, top_p=0.95, max_tokens=8192):
    
    # Load tokenizer for prompt formatting (apply_chat_template)
    try:
        formatting_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_for_vllm, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path_for_vllm} for prompt formatting: {e}", exc_info=True)
        return [] # Cannot proceed without tokenizer for formatting

    stop_tokens = []
    if template == 'qwen': # This should match the template name used in LLaMA-Factory config
        stop_tokens.append("<|im_end|>")
    
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_tokens, # Use passed max_tokens
        stop=stop_tokens, 
        skip_special_tokens=False
    )

    llm_params = {
        "model": base_model_path,
        "tokenizer": tokenizer_path_for_vllm, # vLLM loads its own tokenizer based on this
        "tokenizer_mode": "auto",
        "tensor_parallel_size": 1,#torch.cuda.device_count(),
        "trust_remote_code": True
    }
    
    lora_request_obj = None
    if current_lora_adapter_path:
        llm_params["enable_lora"] = True
        llm_params["max_loras"] = 1 # Max concurrent LoRAs for this test
        lora_request_obj = LoRARequest(
            lora_name="checkpoint_adapter", # Arbitrary name for the request
            lora_int_id=1, # Arbitrary unique int ID for the LoRA
            lora_local_path=current_lora_adapter_path
        )
        logger.info(f"vLLM configured to use LoRA adapter from: {current_lora_adapter_path}")
    else:
        logger.info(f"vLLM configured to load base model from: {base_model_path} (no separate LoRA adapter path provided)")
    
    try:
        llm = LLM(**llm_params)
    except Exception as e:
        logger.error(f"Failed to initialize vLLM LLM: {e}", exc_info=True)
        return []

    gen_prompts = []
    if not messages:
        logger.warning("generate_vllm received an empty messages list.")
        return []

    for i, cur_message_list_or_content in enumerate(messages):
        gen_prompt = ""
        try:
            if template == 'own': 
                # Assuming cur_message_list_or_content is in chat format [{role:user, content: ...}, ...]
                gen_prompt = formatting_tokenizer.apply_chat_template(
                    cur_message_list_or_content,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif template == 'qwen': # This should match the template name used in LLaMA-Factory config
                # Assuming cur_message_list_or_content is the raw question string or first user message content
                question_content = cur_message_list_or_content[0]['content'] if isinstance(cur_message_list_or_content, list) and cur_message_list_or_content else cur_message_list_or_content
                gen_prompt = apply_qwen_math_template(question_content)
            elif template == 'prime':
                question_content = cur_message_list_or_content[0]['content'] if isinstance(cur_message_list_or_content, list) and cur_message_list_or_content else cur_message_list_or_content
                gen_prompt = make_conv_zero(question_content)
            elif template == 'no':
                gen_prompt = cur_message_list_or_content[0]['content'] if isinstance(cur_message_list_or_content, list) and cur_message_list_or_content else cur_message_list_or_content
            else: 
                logger.error(f'Invalid template for vLLM: {template}')
                # Potentially skip this message or raise error
                continue 
        except Exception as e:
            logger.error(f"Error formatting prompt for message index {i} with template {template}: {e}", exc_info=True)
            continue # Skip malformed prompt

        gen_prompts.append(gen_prompt)
        if i == 0: # Log only the first example prompt
            logger.info(f"Example input prompt for vLLM (template: {template}): {gen_prompt[:500]}...") # Log a snippet

    if not gen_prompts:
        logger.warning("No valid prompts were generated for vLLM.")
        return []

    # Generate (example part removed for brevity, main generation follows)
    # outputs = llm.generate([gen_prompts[0]], sampling_params, lora_request=lora_request_obj) # If you need to debug one
    # with open("eval_example.txt", "w", encoding="utf-8") as f:
    #     f.write(gen_prompts[0] + outputs[0].outputs[0].text)

    try:
        logger.info(f"Starting vLLM generation for {len(gen_prompts)} prompts...")
        vllm_outputs = llm.generate(gen_prompts, sampling_params, lora_request=lora_request_obj)
        logger.info(f"vLLM generation completed. Received {len(vllm_outputs)} outputs.")
    except Exception as e:
        logger.error(f"vLLM generation failed: {e}", exc_info=True)
        return []
        
    return vllm_outputs
