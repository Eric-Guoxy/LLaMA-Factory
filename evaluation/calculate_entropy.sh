


(python $ROOT/evaluation/entropy.py \
  --generated_text_path "$output_file" \
  --model_path "$BASE_MODEL_PATH" \
  --save_path "$OUTPUT_DIR/${BASE_MODEL_NAME}-entropies.jsonl" \
  --use_batched_version False) 2>&1 | tee "$OUTPUT_DIR/${output_log_name}.log" 