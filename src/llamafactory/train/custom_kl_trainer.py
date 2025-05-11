import torch
import torch.nn.functional as F
from transformers import Trainer
from peft import PeftModel # Important for LoRA
from .sft.trainer import CustomSeq2SeqTrainer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..hparams import FinetuningArguments

class SFTKLTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure kl_beta and use_kl_loss are part of training_args
        # This might require modifying LLaMA Factory's argument parsing
        # or adding them to the TrainingArguments dataclass definition.
        self.kl_beta = getattr(self.finetuning_args, "kl_beta", 0.0)
        self.use_kl_loss = getattr(self.finetuning_args, "use_kl_loss", False)
        
        if self.use_kl_loss and self.kl_beta > 0:
            print(f"SFTKLTrainer initialized: KL loss enabled with beta={self.kl_beta}. "
                  "Reference model will be the base of the LoRA model.")

    def compute_loss(self, model, inputs, return_outputs=False):
        # `model` is the model being trained (PeftModel for LoRA)
        
        # 1. Standard SFT Loss (Cross-Entropy) from the LoRA-adapted model
        # The original Trainer's compute_loss for language modeling might need labels popped
        labels = inputs.pop("labels", None) 
        
        outputs = model(**inputs)
        current_logits = outputs.logits # Logits from the LoRA-adapted model

        loss_fct = torch.nn.CrossEntropyLoss()
        # Move labels to correct device if they are not already
        if labels is not None:
            labels = labels.to(current_logits.device)
        
        # Shift so that tokens < n predict n
        shift_logits = current_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() if labels is not None else None
        
        # Flatten the tokens
        active_loss_mask = shift_labels.view(-1) != -100 if shift_labels is not None else torch.ones_like(shift_logits.view(-1, shift_logits.size(-1))[:,0], dtype=torch.bool) # Default to all active if no labels
        
        active_current_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss_mask]
        
        if shift_labels is None: # Should not happen in typical SFT
            sft_loss = torch.tensor(0.0).to(current_logits.device) # Or handle error
        else:
            active_shift_labels = shift_labels.view(-1)[active_loss_mask]
            sft_loss = loss_fct(active_current_logits, active_shift_labels)
        
        total_loss = sft_loss

        # 2. KL Divergence Loss (if enabled and model is PeftModel)
        if self.use_kl_loss and self.kl_beta > 0 and isinstance(model, PeftModel):
            base_model = model.get_base_model() # Get the original underlying model
            base_model.eval() # Ensure base model is in evaluation mode

            # Ensure base_model is on the same device as inputs
            # This might be tricky with DeepSpeed if base_model is not managed by it.
            # For simple DDP or single GPU, this should work.
            # inputs_device = next(iter(inputs.values())).device if isinstance(inputs, dict) else inputs['input_ids'].device
            # if base_model.device != inputs_device:
            # base_model.to(inputs_device) # This might cause issues with DeepSpeed ZeRO-3

            with torch.no_grad():
                # Make sure inputs for base_model don't include labels if it doesn't expect them
                base_model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                reference_outputs = base_model(**base_model_inputs)
                reference_logits = reference_outputs.logits
            
            # Shift reference logits similarly
            shift_reference_logits = reference_logits[..., :-1, :].contiguous()

            # Use the same active_loss_mask derived from labels for consistency
            active_current_log_probs = F.log_softmax(active_current_logits, dim=-1) # Already filtered
            
            # Filter reference logits with the same mask
            active_reference_logits = shift_reference_logits.view(-1, shift_reference_logits.size(-1))[active_loss_mask]
            active_reference_log_probs = F.log_softmax(active_reference_logits, dim=-1)

            if active_current_log_probs.numel() > 0 and active_reference_log_probs.numel() > 0:
                kl_div = F.kl_div(active_current_log_probs, active_reference_log_probs, 
                                  reduction='batchmean', log_target=True)
                total_loss = sft_loss + self.kl_beta * kl_div
            
            # Optional: Log individual losses
            # self.log({"sft_loss": sft_loss.item(), "kl_loss": kl_div.item() if 'kl_div' in locals() and kl_div is not None else 0.0})

        # Put labels back if they were popped, for return_outputs=True case
        if labels is not None:
            inputs["labels"] = labels
            
        # For return_outputs=True, the Trainer expects the original model output format
        # If you only return loss, it's fine. If you return (loss, model_outputs),
        # ensure model_outputs is what the Trainer expects (e.g., containing logits).
        # `outputs` here are from the LoRA model.
        return (total_loss, outputs) if return_outputs else total_loss