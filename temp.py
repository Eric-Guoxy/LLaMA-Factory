from pathlib import Path
import pandas as pd
import json
import numpy as np
import uuid
import random
from typing import List, Dict, Optional, Union


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def read_file(file_path):
    """读取文件并返回 list[dict] 格式，自动处理 Parquet 和 JSON 文件"""
    file_path = Path(file_path)
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
        df = df.reset_index(drop=True)
        data = df.to_dict(orient='records')
        return data

    elif file_path.suffix == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    decoder = json.JSONDecoder()
                    idx, data = 0, []
                    while idx < len(content):
                        obj, end_idx = decoder.raw_decode(content, idx)
                        data.append(obj)
                        idx = end_idx
                    return data
            except Exception as e:
                print(f"Error parsing concatenated JSON: {e}")
                return []
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return []
        
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
def save_to_path(data, output_path):
    file_path = Path(output_path)
    if file_path.suffix == '.parquet':
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
    elif file_path.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)
    elif file_path.suffix == '.jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False,cls=NumpyEncoder) + '\n')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
def merge_dataset(data_path_list):
    merged_data = []
    for data_path in data_path_list:
        data = read_file(data_path)
        print(f'dataset_{data_path}_length:',len(data))
        merged_data = merged_data + data
    random.shuffle(merged_data)
    print('merged_dataset_length:',len(merged_data))
    return merged_data
        
def api_response_to_ground_truth(data_path, output_path):
    data = read_file(data_path)
    dataset = []
    for item in data:  # Changed 'line' to 'item' for clarity
        reasoning = item.get('deepseek_reasoning_content', '')  # Safely get reasoning content
        content = item.get('deepseek_content', '')
        if '</think>' in reasoning:
            # If </think> exists, prepend <think>\n to reasoning
            reasoning = '<think>\n' + reasoning 
        else:
            # If </think> doesn't exist, prepend <think>\n and append \n</think>\n\n
            reasoning = '<think>\n' + reasoning + '</think>'
        #print(reasoning[-30:-1])
        query = item['prompt'][0]['content']#item['query']#
        answer = reasoning+content
        ground_truth = item['ground_truth']#item['reward_model']['ground_truth']
        converted = {
            'query': query,
            'answer': answer,
            'ground_truth': ground_truth
        }

        dataset.append(converted)
    
    save_to_path(dataset, output_path)
    
def api_response_to_sft_answer(data_path, output_path):
    data = read_file(data_path)
    dataset = []
    for item in data:  # Changed 'line' to 'item' for clarity

        query = item['query']
        answer = item['answer']
        converted = {
            'query': query,
            'answer': answer,
        }

        dataset.append(converted)
    
    save_to_path(dataset, output_path)

def read_dataset(data_path):
    data = read_file(data_path)
    save_to_path(data[0],'case.json')
    print('first row of data',data[0])
    print('size of dataset:',len(data))
    
def select_data(data_path,range_low,range_high,output_path):
    df = read_file(data_path)
    df_selected = df[range_low:range_high]
    save_to_path(df_selected,output_path)
    print('len of dataset:',len(df_selected))
    
def assign_uid_to_dataset(data_path, output_path):
    df = read_file(data_path)
    df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    save_to_path(df,output_path)

def filter_score_to_valid_uids(data, score):
    return [list(item.keys())[0] for item in data if sum(item[list(item.keys())[0]]) == score]

def create_uid2idx_dict(dataset):
    """创建 UID 到索引的映射字典"""
    uid2idx_dict = {}
    for idx, line in enumerate(dataset):
        uid = line['uid']
        uid2idx_dict[uid] = idx
    return uid2idx_dict

def create_query2idx_dict(dataset):
    query2idx_dict = {}
    for idx, line in enumerate(dataset):
        query = line['query']#line['prompt'][0]['content'] #
        query2idx_dict[query] = idx
    return query2idx_dict

def extract_query_valid_dataset(valid_querys, data_path, output_path):
    """根据 valid uids 提取有效的数据行"""
    valid_dataset = []
    dataset = read_file(data_path)
    uid2idx_dict = create_query2idx_dict(dataset)
    for uid in valid_querys:
        idx = uid2idx_dict[uid]
        valid_dataset.append(dataset[idx])
    save_to_path(valid_dataset, output_path)
     
def extract_valid_dataset(valid_uids, data_path, output_path):
    """根据 valid uids 提取有效的数据行"""
    valid_dataset = []
    dataset = read_file(data_path)
    uid2idx_dict = create_uid2idx_dict(dataset)
    for uid in valid_uids:
        idx = uid2idx_dict[uid]
        valid_dataset.append(dataset[idx])
    save_to_path(valid_dataset, output_path)
    
def distract_dataset(valid_uids, data_path, output_path):
    """根据 uids 删除无效的数据行"""
    dataset = read_file(data_path)
    uid2idx_dict = create_uid2idx_dict(dataset)
    idxs_to_remove = []
    for uid in valid_uids:
        idx = uid2idx_dict.get(uid)
        if idx is not None:
            idxs_to_remove.append(idx)
            
    idxs_to_remove.sort(reverse=True)
    print(idxs_to_remove)
    for idx in idxs_to_remove:
        dataset.pop(idx)
        
    save_to_path(dataset, output_path)

def distract_query_dataset(valid_uids, data_path, output_path):
    """根据 uids 删除无效的数据行"""
    dataset = read_file(data_path)
    uid2idx_dict = create_query2idx_dict(dataset)
    idxs_to_remove = []
    for query in valid_uids:
        idx = uid2idx_dict.get(query)
        if idx is not None:
            idxs_to_remove.append(idx)

    idxs_to_remove.sort(reverse=True)
    print(idxs_to_remove)
    for idx in idxs_to_remove:
        dataset.pop(idx)
    
    print('len_of_rest_dataset:',len(dataset))
    save_to_path(dataset, output_path)
    
def score_to_score_uid_dict(score_file_path, output_path, max_score): 
    score_data = read_file(score_file_path)
    result_dict = {}
    for i in range(max_score+1):
        valid_uids = filter_score_to_valid_uids(score_data, i)
        print(f'score{i}',len(valid_uids))
        result_dict[f'score{i}'] = valid_uids
    save_to_path([result_dict], output_path)

def convert_dataset_entry(entry):
    return {
        "data_source": "deepscaler",
        "prompt": json.dumps([
            {"content": "", "role": "system"},
            {"content": entry["query"], "role": "user"}
        ]),
        "target": json.dumps([
            {"content": entry["answer"], "role": "assistant"}
        ]),
        "ability": "",
        "reward_model": json.dumps({
            "ground_truth": entry["ground_truth"],
            "style": "rule"
        }),
        "extra_info": json.dumps({
            "index": -1,
            "split": "default"
        })
    }
    

def convert_dataset(input_path, output_path):
    data = read_file(input_path)
    converted_data = [convert_dataset_entry(entry) for entry in data]
    save_to_path(converted_data, output_path)
    
    
read_dataset('data/openr1_16k_sft.json')