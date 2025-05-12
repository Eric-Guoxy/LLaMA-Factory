import torch
import torch.nn.functional as F
from transformers import Trainer
from peft import PeftModel # Important for LoRA
from .sft.trainer import CustomSeq2SeqTrainer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..hparams import FinetuningArguments
    from transformers import PreTrainedModel # For type hinting model

class SFTKLTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = getattr(self.finetuning_args, "kl_beta", 0.0)
        self.use_kl_loss = getattr(self.finetuning_args, "use_kl_loss", False)
        
        if self.use_kl_loss and self.kl_beta > 0:
            # Ensure the model is a PeftModel for LoRA KL divergence
            if not isinstance(self.model, PeftModel):
                raise ValueError("SFTKLTrainer expects a PeftModel when use_kl_loss is True for LoRA KL.")
            print(f"SFTKLTrainer initialized: KL loss enabled with beta={self.kl_beta}. "
                  "Reference model will be the base of the LoRA model.")
        elif self.use_kl_loss and self.kl_beta == 0:
            print("SFTKLTrainer: use_kl_loss is True, but kl_beta is 0. KL loss will not be applied.")

    def compute_loss(
        self, 
        model: "PeftTrainedModel", # This will be the PeftModel 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # 1. Standard SFT Loss (Cross-Entropy) from the LoRA-adapted model
        # The original Trainer's compute_loss for language modeling might need labels popped
        
        outputs = model(**inputs)

        sft_loss = outputs.loss # This is the primary SFT loss
        current_logits = outputs.logits # Logits from the LoRA-adapted model

        total_loss = sft_loss
        kl_divergence_raw = torch.tensor(0.0, device=sft_loss.device) # Initialize

        # 2. KL Divergence Loss (if enabled and model is PeftModel)
        if self.use_kl_loss and self.kl_beta > 0 and isinstance(model, PeftModel):
            base_model = model.base_model.model # Get the original underlying model
            is_base_training = base_model.training
            base_model.eval() # Ensure base model is in evaluation mode

            with torch.no_grad():
                # Make sure inputs for base_model don't include labels if it doesn't expect them
                base_model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                reference_outputs = base_model(**base_model_inputs)
                reference_logits = reference_outputs.logits
            
            # Restore base model's original training state if it was changed
            if is_base_training:
                base_model.train()
            
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

                if active_current_logits.numel() > 0 and active_current_logits.numel() > 0:
                    # Compute log probabilities for KL divergence 
                    current_log_probs = F.log_softmax(active_current_logits, dim=-1) 
                    reference_log_probs = F.log_softmax(active_reference_logits, dim=-1)
                    
                    kl_divergence_raw = F.kl_div(
                        current_log_probs,
                        reference_log_probs,
                        reduction='batchmean',
                        log_target=True
                    )
                    total_loss = sft_loss + self.kl_beta * kl_divergence_raw
                else:
                    if self.state.is_world_process_zero:
                        print("Warning: No active tokens found for KL divergence after masking. Skipping KL for this step.")

        # Logging
        if self.state.is_world_process_zero:
            log_data = {"sft_loss": sft_loss.detach().item()}
            if self.use_kl_loss and self.kl_beta > 0:
                log_data["kl_loss_raw"] = kl_divergence_raw.detach().item()
            self.log(log_data)

        return (total_loss, outputs) if return_outputs else total_loss