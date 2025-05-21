export CUDA_VISIBLE_DEVICES="4,5,6,7"
export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
#export NCCL_NVLS_ENABLE=0

# Train the model
cd /home/inspur/cth/LLaMA-Factory
llamafactory-cli train examples/train_full/llama3_full_sft.yaml

# Evaluate the model
bash evaluation/eval.sh