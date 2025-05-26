from utils.utils import read_file, save_to_path, read_dataset

#read_dataset('/home/inspur/cth/LUFFY/data/openr1.parquet')

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("")
#model = AutoModelForCausalLM.from_pretrained("Elliott/LUFFY-Qwen-Math-7B-Zero")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Elliott/LUFFY-Qwen-Math-7B-Zero",
    local_dir="/mnt/chenth",
    local_dir_use_symlinks=False
)
