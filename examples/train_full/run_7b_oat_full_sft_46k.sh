export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Train the model
cd ~/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen_2_5_oat_full_sft_46k.yaml

# Evaluate the model
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluation/eval_7b_oat_sft_46k.sh
