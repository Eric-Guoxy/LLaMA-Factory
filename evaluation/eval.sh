ROOT=/home/inspur/cth/LUFFY
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=/home/inspur/cth/LLaMA-Factory/saves/Qwen2.5-Math-7B/full/sft
MODEL_NAME=Qwen2.5-Math-7B-full

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=qwen
fi

CUDA_VISIBLE_DEVICES=5,6,7,8 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --tensor_parallel_size=4 \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
  