export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# --- OLD H20 ---
# Train the model
cd ~/cth/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen_2_5_math_full_rope_sft_incorrect.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluation/eval_7b_rope_incorrect.sh

# --- NEW H20 ---
# # Train the model
# cd ~/cth/LLaMA-Factory
# llamafactory-cli train examples/train_full/qwen2_5_math_7b_full_sft.yaml

# # Evaluate the model
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash evaluation/eval_7b_sft.sh
