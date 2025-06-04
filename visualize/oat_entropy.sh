CUDA_VISIBLE_DEVICES=4,5,6,7 python entropy.py \
    --generated_text_path "/home/inspur/cth/LLaMA-Factory/results/Qwen2.5-Math-7B/full/Qwen2.5-Math-7B-Oat-Zero-final-sft.jsonl" \
    --model_path "/home/inspur/cth/models/Qwen2.5-Math-7B-Oat-Zero" \
    --save_path "oat_entropy_cuda.jsonl" \
    --device "cuda" \
    --use_batched_version \
    --batch_size 32