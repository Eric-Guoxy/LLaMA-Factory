# --- OLD H20 ---
ROOT=~/cth/cth/LLaMA-Factory
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=$ROOT/results/Qwen2.5-Math-7B/rope/basic
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_NVLS_ENABLE=0

# --- Configuration for the base model ---
# This is the main directory, which might contain the final model and/or checkpoint subdirectories
BASE_MODEL_PATH=/root/cth/cth/models/open-r1--Qwen2.5-Math-7B-RoPE-300k
# This base name will be used for naming output files
BASE_MODEL_NAME=Qwen2.5-Math-7B_rope

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=qwen_basic
fi

# --- Function to run evaluation ---
# $1: Path to the model/checkpoint to evaluate
# $2: Suffix for the output file name (e.g., "final-sft", "checkpoint-100")
run_evaluation() {
  local current_model_to_eval_path="$1"
  local name_suffix="$2"
  local output_log_name="${BASE_MODEL_NAME}-${name_suffix}"

  echo ""
  echo "========================================================================"
  echo "Starting evaluation for: $current_model_to_eval_path"
  echo "Output files will be named: ${output_log_name}.jsonl / .log"
  echo "========================================================================"

  # Ensure the model path exists before attempting to run
  if [ ! -d "$current_model_to_eval_path" ]; then
    echo "Error: Model path '$current_model_to_eval_path' not found. Skipping."
    echo "========================================================================"
    return
  fi

  # Assuming generate_vllm.py is in $ROOT/eval_scripts/
  # Adjust python path if generate_vllm.py is elsewhere (e.g., relative to this script)
  (python $ROOT/evaluation/generate_vllm.py \
    --model_path "$current_model_to_eval_path" \
    --input_file "$DATA" \
    --remove_system True \
    --add_oat_evaluate True \
    --output_file "$OUTPUT_DIR/${output_log_name}.jsonl" \
    --tensor_parallel_size 4 \
    --max_tokens 8192 \
    --template "$TEMPLATE") 2>&1 | tee "$OUTPUT_DIR/${output_log_name}.log" 
  
  # Check exit status of the python script
  if [ $? -eq 0 ]; then
    echo "Successfully completed evaluation for: $current_model_to_eval_path"
  else
    echo "Error during evaluation for: $current_model_to_eval_path. Check log: $OUTPUT_DIR/${output_log_name}.log"
  fi
  echo "Log file: $OUTPUT_DIR/${output_log_name}.log"
  echo "JSONL output: $OUTPUT_DIR/${output_log_name}.jsonl"
  echo "========================================================================"
}

# --- Evaluate the main/final model specified by BASE_MODEL_PATH ---
# The suffix "final-sft" clearly indicates this is the evaluation of the main sft path
run_evaluation "$BASE_MODEL_PATH" "final-sft"

# --- Find and evaluate all checkpoint subdirectories within BASE_MODEL_PATH ---
echo ""
echo "Searching for checkpoints in $BASE_MODEL_PATH ..."
found_checkpoints=0
if [ -d "$BASE_MODEL_PATH" ]; then
  for checkpoint_dir in "$BASE_MODEL_PATH"/checkpoint-*/; do
    # The glob pattern /checkpoint-*/ might return the pattern itself if no matches are found.
    # So, we must check if it's a directory and actually exists.
    if [ -d "$checkpoint_dir" ]; then
      found_checkpoints=$((found_checkpoints + 1))
      checkpoint_name=$(basename "$checkpoint_dir") # Extracts "checkpoint-XXX"
      run_evaluation "$checkpoint_dir" "$checkpoint_name"
    fi
  done
fi

if [ "$found_checkpoints" -eq 0 ]; then
  echo "No checkpoint directories (e.g., 'checkpoint-XXX') found under '$BASE_MODEL_PATH'."
fi

echo ""
echo "All evaluations finished."

# --- Visualization Section ---
echo "========================================================================"
echo "Starting visualization..."
echo "Log directory: $OUTPUT_DIR"
echo "========================================================================"




