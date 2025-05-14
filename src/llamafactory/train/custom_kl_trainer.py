import torch
import torch.nn.functional as F
from transformers import Trainer
from peft import PeftModel # Important for LoRA
from .sft.trainer import CustomSeq2SeqTrainer

from typing import TYPE_CHECKING, Dict, Union, Tuple, Optional
if TYPE_CHECKING:
    from ..hparams import FinetuningArguments
    from transformers import PreTrainedModel # For type hinting model

class SFTKLTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = getattr(self.finetuning_args, "kl_beta", 0.0)
        self.use_kl_loss = getattr(self.finetuning_args, "use_kl_loss", False)
        print(f"SFTKLTrainer __init__ called on rank {self.args.process_index}. use_kl_loss: {self.use_kl_loss}, kl_beta: {self.kl_beta}", flush=True) # ADD THIS
        
        if self.use_kl_loss and self.kl_beta > 0:
            # Ensure the model is a PeftModel for LoRA KL divergence
            actual_peft_model = None
            if hasattr(self.model, 'module'):
                actual_peft_model = self.model.module
            else:
                # If not using DeepSpeed or a similar wrapper, self.model might be the PeftModel directly.
                actual_peft_model = self.model

            if not isinstance(actual_peft_model, PeftModel):
                raise ValueError(f"SFTKLTrainer expects a PeftModel when use_kl_loss is True for LoRA KL. Got {type(actual_peft_model)}")
            print(f"SFTKLTrainer initialized: KL loss enabled with beta={self.kl_beta}. "
                  "Reference model will be the base of the LoRA model.")
        elif self.use_kl_loss and self.kl_beta == 0:
            print("SFTKLTrainer: use_kl_loss is True, but kl_beta is 0. KL loss will not be applied.")

    def compute_loss(
        self, 
        model: "PeftTrainedModel", # This will be the PeftModel 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Initialize the logging dict
        log_data = {}

        # 1. Standard SFT Loss (Cross-Entropy) from the LoRA-adapted model
        # The original Trainer's compute_loss for language modeling might need labels popped
        
        outputs = model(**inputs)

        sft_loss = outputs.loss # This is the primary SFT loss
        current_logits = outputs.logits # Logits from the LoRA-adapted model

        total_loss = sft_loss
        kl_divergence_raw = torch.tensor(0.0, device=sft_loss.device) # Initialize

        # 2. KL Divergence Loss (if enabled and model is PeftModel)
        if self.use_kl_loss and self.kl_beta > 0: # and isinstance(model, PeftModel):
            peft_model_instance = model.module if hasattr(model, 'module') else model

            # --- BEGIN DEBUG LOGGING for PEFT state ---
            if self.state.is_world_process_zero: # Log only on rank 0
                log_data['debug_peft_instance_type'] = str(type(peft_model_instance))
                if isinstance(peft_model_instance, PeftModel):
                    log_data['debug_peft_config'] = str(getattr(peft_model_instance, 'peft_config', 'N/A'))
                    try:
                        log_data['debug_peft_active_adapters'] = str(peft_model_instance.active_adapters)
                    except Exception as e_active:
                        log_data['debug_peft_active_adapters'] = f"Error: {str(e_active)}"
                    log_data['debug_peft_internal_active_adapter'] = str(getattr(peft_model_instance, '_active_adapter', 'N/A'))
                    
                    peft_config_dict = getattr(peft_model_instance, 'peft_config', None)
                    if isinstance(peft_config_dict, dict):
                        log_data['debug_peft_config_keys'] = str(list(peft_config_dict.keys()))
                    else:
                        log_data['debug_peft_config_keys'] = 'Not a dict or N/A'

                    lora_layers_found_log = []
                    for name, module_item in peft_model_instance.named_modules():
                        if "LoraLayer" in str(type(module_item)):
                            lora_layers_found_log.append(name)
                    log_data['debug_lora_layers_found_count'] = len(lora_layers_found_log)
                    log_data['debug_lora_layers_example'] = str(lora_layers_found_log[:5]) if lora_layers_found_log else "None"
                else:
                    log_data['debug_peft_is_instance'] = False
            # --- END DEBUG LOGGING for PEFT state ---

            # Temporarily disable adapters to get reference logits from the base model's architecture
            #try:
            #    peft_model_instance.disable_adapters() # This is where the error occurs
            #except ValueError as e:
            #    if self.state.is_world_process_zero:
            #        print(f"ERROR [Step {self.state.global_step}]: Failed to disable adapters: {e}", flush=True)
            #        print(f"  peft_model_instance.peft_config at error: {getattr(peft_model_instance, 'peft_config', 'N/A')}", flush=True)
            #    raise e # Re-raise the error to stop execution as before
            original_training_state_hf_model = peft_model_instance.training
            peft_model_instance.eval()

            with torch.no_grad():
                with peft_model_instance.disable_adapter():
                    # Make sure inputs for base_model don't include labels if it doesn't expect them
                    base_model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    reference_outputs = peft_model_instance(**base_model_inputs)
                    reference_logits = reference_outputs.logits
            
            # Restore base model's original training state if it was changed
            if original_training_state_hf_model:
                peft_model_instance.train()
            
            # Align and mask logits for KL divergence calculation
            labels = inputs.get("labels")
            if labels is None:
                # If no labels, KL divergence over all tokens might be too noisy or not meaningful
                if self.state.is_world_process_zero:
                    print("Warning: Labels not found in inputs for KL divergence calculation. Skipping KL loss.")
            else:
                # Shift logits and labels for Causal LM
                shift_current_logits = current_logits[..., :-1, :].contiguous()
                shift_reference_logits = reference_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens and apply mask (ignore padding tokens)
                loss_mask = shift_labels.ne(self.label_ignore_index if hasattr(self, 'label_ignore_index') else -100)

                active_current_logits = shift_current_logits.view(-1, shift_current_logits.size(-1))[loss_mask.view(-1)]
                active_reference_logits = shift_reference_logits.view(-1, shift_reference_logits.size(-1))[loss_mask.view(-1)]

                # --- BEGIN DEBUG PRINTS ---
                if self.state.is_world_process_zero: # Only log for rank 0 to avoid clutter
                    # You might want to make this conditional on global_step too, e.g., self.state.global_step % 1 == 0
                    print(f"--- Debug KL - Step {self.state.global_step} [Rank {self.args.process_index}] ---")
                    print(f"  Labels shape: {labels.shape}")
                    print(f"  Shift_labels shape: {shift_labels.shape}")
                    print(f"  Loss_mask sum: {loss_mask.sum().item()} (out of {loss_mask.numel()})")
                    print(f"  Active_current_logits numel: {active_current_logits.numel()}")
                    print(f"  Active_reference_logits numel: {active_reference_logits.numel()}")
                    if active_current_logits.numel() == 0 or active_reference_logits.numel() == 0:
                        print(f"  >> KL DIVERGENCE WILL BE SKIPPED THIS MICRO-BATCH ON RANK 0 <<")
                    else:
                        # Temporarily compute and print KL here for debugging before the actual computation
                        temp_current_log_probs = F.log_softmax(active_current_logits, dim=-1)
                        temp_reference_log_probs = F.log_softmax(active_reference_logits, dim=-1)
                        temp_kl_div = F.kl_div(temp_current_log_probs, temp_reference_log_probs, reduction='batchmean', log_target=True)
                        print(f"  >> Tentative kl_divergence_raw if computed: {temp_kl_div.item()}")
                    print(f"--- End Debug KL ---")
                # --- END DEBUG PRINTS ---

                if active_current_logits.numel() > 0 and active_reference_logits.numel() > 0:
                    # Compute log probabilities for KL divergence 
                    current_log_probs = F.log_softmax(active_current_logits, dim=-1) 
                    reference_log_probs = F.log_softmax(active_reference_logits, dim=-1)
                    
                    kl_divergence_raw = F.kl_div(
                        current_log_probs,
                        reference_log_probs,
                        reduction='batchmean',
                        log_target=True
                    )
                    log_data['kl_divergence_raw'] = kl_divergence_raw.detach().item()
                    total_loss = sft_loss + self.kl_beta * kl_divergence_raw

                    # Store first 10 elements for logging
                    if self.state.is_world_process_zero: # Log only on main process
                        num_elements_to_log = 10
                        # Ensure we don't go out of bounds if tensors are smaller
                        actual_elements_current = min(num_elements_to_log, current_log_probs.numel())
                        actual_elements_reference = min(num_elements_to_log, reference_log_probs.numel())
                        
                        # Flatten and take the first N elements, then convert to list for logging
                        log_data['current_log_probs_head'] = current_log_probs.view(-1)[:actual_elements_current].detach().cpu().tolist()
                        log_data['reference_log_probs_head'] = reference_log_probs.view(-1)[:actual_elements_reference].detach().cpu().tolist()
                else:
                    if self.state.is_world_process_zero:
                        print("Warning: No active tokens found for KL divergence after masking. Skipping KL for this step.")

        # Logging
        if self.state.is_world_process_zero:
            log_data['sft_loss'] = sft_loss.detach().item()
            #if self.use_kl_loss and self.kl_beta > 0:
            #    kl_loss = kl_divergence_raw.detach().item()
            #    print("The kl loss is: ", kl_loss)
            #    log_data["kl_loss_raw"] = kl_loss
            #log_data["total_loss"] = total_loss.detach().item()
            self.log(log_data)

        return (total_loss, outputs) if return_outputs else total_loss