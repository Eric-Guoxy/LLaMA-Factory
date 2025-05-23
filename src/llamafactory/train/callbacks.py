# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from .generate_hf import test


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key) for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu")

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class CustomTestingCallback(TrainerCallback):
    def __init__(self, model_args, finetuning_args, tokenizer):
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.original_model_devices = {}
        self.tokenizer = tokenizer
        self.trainer = None
    
    @override
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # The trainer instance is often available in kwargs here, or as the first positional arg
        # For Hugging Face Trainer, `self` of the callback is usually bound to the trainer.
        # However, explicitly capturing it from kwargs if available is safer.
        if "trainer" in kwargs:
            self.trainer = kwargs["trainer"]
        elif hasattr(self, "trainer_ref"): # Some HF versions might pass it as trainer_ref
             self.trainer = self.trainer_ref # Or however it's named if directly available
        # If the callback is directly part of a Trainer instance, self.trainer might already be populated
        # by the Trainer's __init__ when adding callbacks.
        # A common pattern is that the Trainer instance itself is passed as the first argument to callbacks
        # or the callback instance has a `parent` or `trainer` attribute set by the Trainer.
        # For now, let's assume it might be in kwargs or we'll try to get it in on_save.

    @override
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Another point where the trainer might be explicitly passed or accessible
        if "trainer" in kwargs and self.trainer is None:
            self.trainer = kwargs["trainer"]
        # If the callback is a component of the Trainer, self.trainer might be set by the Trainer.
        # If your Trainer instance is accessible globally or via a specific context, that's another way.
        # For now, we'll primarily rely on on_save's kwargs or what was set in on_init_end.

    @override
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model") # Get the model instance (DeepSpeed-wrapped or regular)
        
        # Determine if this is a test step based on global_step and model_args
        is_test_step = (
            self.model_args.do_test
            and state.global_step > 0
            and state.global_step % self.model_args.test_steps == 0
        )

        # --- Barrier 1: All processes sync before any action ---
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # --- Phase 1: All ranks move their model to CPU if it's a test step ---
        if is_test_step:
            if model is not None:
                # Store original device. args.local_rank is crucial for multi-GPU.
                # model.device might give 'cuda' generally, but we need the specific local rank's GPU.
                # The actual device for a rank is usually `torch.device("cuda", args.local_rank)`
                current_device = torch.device("cuda", args.local_rank)
                self.original_model_devices[args.local_rank] = current_device
                
                logger.info(f"CustomTestingCallback (Rank {args.local_rank}): Moving model from {current_device} to CPU.")
                model.to("cpu") # Each rank moves its model (or its shard's parameters if ZeRO3, replica if ZeRO2/DDP)
                
                if torch.cuda.is_available():
                    # Each rank clears its own GPU's cache
                    torch.cuda.set_device(args.local_rank) 
                    torch.cuda.empty_cache()
                logger.info(f"CustomTestingCallback (Rank {args.local_rank}): Model moved to CPU and cache emptied for GPU {args.local_rank}.")
            else:
                logger.warning(f"CustomTestingCallback (Rank {args.local_rank}): Model not found in kwargs, cannot move to CPU.")

            # --- Barrier 2: All processes sync after ensuring models are on CPU ---
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        # --- Phase 2: Rank 0 performs the test ---
        if state.is_world_process_zero:
            if is_test_step:
                model_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                logger.info(f"CustomTestingCallback (Rank 0): Preparing for test at global_step {state.global_step} using checkpoint {model_checkpoint_path}")

                if not os.path.exists(model_checkpoint_path):
                    logger.warning(
                        f"CustomTestingCallback (Rank 0): Test skipped. Checkpoint {model_checkpoint_path} not found."
                    )
                else:
                    trainer = self.trainer

                    # Determine paths for vLLM
                    vllm_base_model_path = self.model_args.model_name_or_path
                    vllm_lora_adapter_path = None
                    is_lora_finetuning = hasattr(self.finetuning_args, 'finetuning_type') and self.finetuning_args.finetuning_type == "lora"
                    if is_lora_finetuning and os.path.exists(os.path.join(model_checkpoint_path, "adapter_config.json")):
                        vllm_lora_adapter_path = model_checkpoint_path
                    else:
                        vllm_base_model_path = model_checkpoint_path # Assume checkpoint is full model if not LoRA
                    tokenizer_path_for_vllm = self.model_args.model_name_or_path
                    
                    logger.info(f"CustomTestingCallback (Rank 0): Running test with vLLM. Base: {vllm_base_model_path}, Adapter: {vllm_lora_adapter_path}")
                    
                    try:
                        metrics_result = test( 
                            model=model,
                            tokenizer=self.tokenizer,
                            base_model_path=vllm_base_model_path,
                            current_lora_adapter_path=vllm_lora_adapter_path,
                            tokenizer_path_for_vllm=tokenizer_path_for_vllm,
                            input_file=self.model_args.test_input_file,
                            output_file=self.model_args.test_output_file,
                            debug=self.model_args.test_debug,
                            remove_system=self.model_args.test_remove_system,
                            template=self.model_args.test_template,
                            temperature=self.model_args.test_temperature,
                            top_p=self.model_args.test_top_p,
                            max_tokens=self.model_args.test_max_tokens
                        )

                        metrics_to_log = {}
                        if isinstance(metrics_result, dict):
                            for data_source, acc_val in metrics_result.items():
                                if isinstance(acc_val, dict):
                                     for metric_name, val in acc_val.items():
                                        metrics_to_log[f"test/{data_source}/{metric_name}"] = val
                                else:
                                    metrics_to_log[f"test/{data_source}"] = acc_val
                        else:
                            logger.warning(f"CustomTestingCallback (Rank 0): Test metrics format not recognized: {type(metrics_result)}")

                        if metrics_to_log and trainer:
                            trainer.log(metrics_to_log)
                        
                        logger.info(f"CustomTestingCallback (Rank 0): Test finished at global_step {state.global_step}")

                    except Exception as e:
                        logger.error(f"CustomTestingCallback (Rank 0): Error during test execution: {e}", exc_info=True)
            # else: # This case is if it's Rank 0 but not a test step
                # logger.info(f"CustomTestingCallback (Rank 0): Not a designated test step ({state.global_step} % {self.model_args.test_steps} != 0 or not do_test). Skipping test logic.")
        
        # --- Barrier 3: All processes sync before moving model back to GPU ---
        # Rank 0 waits here after testing (or skipping test logic if not a test step). 
        # Other ranks also arrive here (after Barrier 2 if it was a test step, or after Barrier 1 if not).
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # --- Phase 3: All ranks move their model back to original GPU if it was a test step ---
        if is_test_step:
            if model is not None:
                original_device = self.original_model_devices.get(args.local_rank)
                if original_device:
                    logger.info(f"CustomTestingCallback (Rank {args.local_rank}): Moving model back from CPU to {original_device}.")
                    model.to(original_device) # Each rank moves its model back
                    if torch.cuda.is_available():
                        torch.cuda.set_device(args.local_rank)
                        torch.cuda.empty_cache()
                    logger.info(f"CustomTestingCallback (Rank {args.local_rank}): Model moved back to {original_device}.")
                else:
                    # This case should ideally not happen if device was stored correctly
                    logger.warning(f"CustomTestingCallback (Rank {args.local_rank}): Original device not found for rank. Cannot move model back.")
            # Clear the stored devices for the next on_save call if needed, though it will be overwritten.
            # self.original_model_devices.pop(args.local_rank, None) 

        # --- Barrier 4: All processes sync after restoring model (or if no restoration was needed) ---
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)
            

class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )
