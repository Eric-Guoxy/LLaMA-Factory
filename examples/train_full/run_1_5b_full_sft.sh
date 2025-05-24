export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Train the model
cd ~/cth/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen_1_5b_distill_sft.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluation/eval_1_5b_sft.sh

