from utils.utils import read_file, save_to_path, read_dataset

data = read_file("/home/inspur/cth/LUFFY/data/openr1.parquet")

def process_data(data):
    # Assuming data is a list of dictionaries
    processed_data = []
    for item in data:
        # Process each item as needed
        processed_item = {
            "system": item['prompt'][0]['content'],
            "query": item['prompt'][1]['content'],
            "answer": item['target'][0]['content']
        }
        processed_data.append(processed_item)
    return processed_data



#save_to_path(process_data(data), "openr1.json")
read_dataset("openr1.json")