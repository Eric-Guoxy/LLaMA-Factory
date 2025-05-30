export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Train the model
cd ~/cth/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen_1_5b_distill_kl.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=4,5,6,7
bash evaluation/eval_1_5b_kl.sh

