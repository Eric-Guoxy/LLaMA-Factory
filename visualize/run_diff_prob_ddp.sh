export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

python diff_prob_ddp.py \
    --final_model_path "/data/cth/saves/Qwen2.5-Math-7B/full/sft_curr_part2" \
    --ref_model_path "/home/inspur/cth/models/Qwen2.5-Math-7B" \
    --data_path "/home/inspur/cth/LLaMA-Factory/visualize/models/Qwen2.5-Math-7B-Oat-Zero/Qwen2.5-Math-7B-Oat-Zero-eval.jsonl" \
    --final_model_name "Qwen2.5-Math-7B-curr-part2" \
    --ref_model_name "Qwen2.5-Math-7B (base)" \
    --hf_processing_batch_size 10 \
    --num_visualization_examples 0 \

# hf_processing_batch_size取10就是极限了