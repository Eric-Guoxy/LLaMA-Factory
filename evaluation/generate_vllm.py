#export HF_ENDPOINT=https://hf-mirror.com  
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import os

from math_verify import parse, verify
from oat_math_grader import boxed_reward_fn as oat_evaluate

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    with open("golden_answers.json", "w", encoding="utf-8") as f:
        json.dump(golden_answers, f, ensure_ascii=False)
    with open("predict_answers.json", "w", encoding="utf-8") as f:
        json.dump(predict_answers, f, ensure_ascii=False)
    labels = list(map(verify, golden_answers, predict_answers))
    return labels


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {{thoughts}} </think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{{}} for closed-form results like multiple choices or mathematical solutions. "
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def main(
    input_file, 
    output_file, 
    model_path, 
    tokenizer_path=None,
    debug=False, 
    remove_system=True, 
    template='own', 
    temperature=0.6, 
    top_p=1.0, 
    max_tokens=8192, 
    n=1, 
    force_generate=True, 
    add_think_before_answer=False, 
    add_oat_evaluate=False, 
    any_true=False, 
    skip_scoring=False, 
    output_eval=None, 
    no_split_think=False,
    tensor_parallel_size=1
    ):
    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'

    if force_generate or (not os.path.exists(dec_output_path)):
        # 数据处理
        messages = df['prompt'].tolist()
        assert remove_system is False
        if remove_system:
            print('remove system')
            assert messages[0][0]['role'] == 'system'
            messages = [message[1:] for message in messages]
            
        else:
            assert remove_system is False
            print('not remove system')
            
        answers = df['reward_model'].tolist()
        answers = [answer['ground_truth'] for answer in answers]
        # if debug:
            # answers = answers[:10]
        assert len(messages) == len(answers)
                
        print(messages[0])
        print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}")
        outputs = generate_vllm(messages, model_path, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, tensor_parallel_size=tensor_parallel_size)
        # rets = {}
        
        # save the outputs first
        with open(dec_output_path, 'w') as fo:
            for i, output in enumerate(outputs):
                prompt = output.prompt
                for j in range(n):
                    generated_text = output.outputs[j].text
                    item = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'answer': answers[i]
                    }
                    fo.write(json.dumps(item) + '\n')
                    
        # format sort prompts, outputs, answers
        assert len(outputs[0].outputs) == n
        prompts = [out.prompt for out in outputs for j in range(n)]
        answers = [answers[i] for i in range(len(outputs)) for j in range(n)]
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]
    else:
        print('Found already decoded file, skip decoding...')
        jss = []
        with open(dec_output_path, 'r') as f:
            for line in f:
                jss.append(json.loads(line))
        
        outputs = [item['generated_text'] for item in jss]
        prompts = [item['prompt'] for item in jss]
        answers = [item['answer'] for item in jss]
    
    data_sources = df['data_source'].tolist()
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0
    from tqdm import tqdm

    print('Scoring...')
    if skip_scoring:
        return
    
    # for i, output in tqdm(enumerate(outputs)):
    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        # print(i)
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        think_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            think_format = True
        if no_split_think:
            think_format = False

        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                labels = [False]
                
        if labels is None:
            if not add_oat_evaluate:
                try:
                    labels = labeling_responses([generated_text,], answer)
                except Exception as e:
                    labels = [False]
            else: # use oat evaluate
                new_label = oat_evaluate(generated_text, answer, fast=False)
                new_label = new_label[1] == 1.0
                labels = [new_label]
        
        rets[data_sources[i]].append(labels[0])
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg += 1

    
    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    
    accs = []
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
        accs.append(acc)
    
    print('avg acc: ', np.array(accs).mean())
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')

def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, tensor_parallel_size=1, n=1):
    #vllm模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # guarantee the stop tokens of each response
    stop_tokens = []
    if template == 'qwen' or template == 'own':
        stop_tokens.append("<|im_end|>")
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop_tokens, skip_special_tokens=False, n=n)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)  # 替换成本地路径

    gen_prompts = []
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
        )
        elif template == 'qwen':
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
        )
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)
        if i == 0:
            print('Example input: ', gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs

if __name__ == "__main__":
    import fire
    fire.Fire(main)