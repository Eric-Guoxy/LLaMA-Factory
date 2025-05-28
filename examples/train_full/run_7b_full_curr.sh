export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# # Train the model
# cd ~/cth/LLaMA-Factory
# llamafactory-cli train examples/train_full/qwen2_5_math_7b_full_curr_part1.yaml

# # Evaluate the model
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash evaluation/eval_7b_curr_part1.sh

# Train the model
cd ~/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen2_5_math_7b_full_curr_part2.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluation/eval_7b_curr_part2.sh

# Train the model
cd ~/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen2_5_math_7b_full_curr_part3.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluation/eval_7b_curr_part3.sh

