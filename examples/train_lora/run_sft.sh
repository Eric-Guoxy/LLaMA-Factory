export CUDA_VISIBLE_DEVICES="0,1,2,3"
export ALLOW_EXTRA_ARGS=1
export NCCL_DEBUG=INFO
#export NCCL_NVLS_ENABLE=0
cd /home/inspur/cth/LLaMA-Factory
llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml