import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM
from peft import PeftModel # Important for LoRA
from .sft.trainer import CustomSeq2SeqTrainer
from typing_extensions import override

from typing import TYPE_CHECKING, Dict, Union, Tuple, Optional
if TYPE_CHECKING:
    from ..hparams import FinetuningArguments, ModelArguments, Seq2SeqTrainingArguments
    from transformers import PreTrainedModel # For type hinting model

class SFTKLTrainer(CustomSeq2SeqTrainer):
    def __init__(self, model_args: "ModelArguments", training_args: "Seq2SeqTrainingArguments", *args, **kwargs):
        super().__init__(model_args=model_args, *args, **kwargs) # Pass through model_args if super() expects it, or handle it here
        self.model_args = model_args # Store model_args
        self.training_args = training_args
        self.kl_beta = getattr(self.finetuning_args, "kl_beta", 0.0)
        self.use_kl_loss = getattr(self.finetuning_args, "use_kl_loss", False)
        self.reference_model = None # Initialize reference_model attribute
        self.max_txt_len = 0
        print(f"SFTKLTrainer __init__ called on rank {self.args.process_index}. use_kl_loss: {self.use_kl_loss}, kl_beta: {self.kl_beta}", flush=True)
        
        if self.use_kl_loss and self.kl_beta > 0:
            actual_model_for_check = self.model.module if hasattr(self.model, 'module') else self.model

            if self.finetuning_args.finetuning_type != "full":
                if not isinstance(actual_model_for_check, PeftModel):
                    raise ValueError(
                        f"SFTKLTrainer expects a PeftModel when use_kl_loss is True and finetuning_type is not 'full'. "
                        f"Got {type(actual_model_for_check)}"
                    )
                print(f"SFTKLTrainer initialized for PEFT: KL loss enabled with beta={self.kl_beta}. "
                      "Reference model will be the base of the LoRA model.")
            else: # Full finetuning with KL loss
                print(f"SFTKLTrainer initializing for FULL finetuning with KL loss (beta={self.kl_beta}). Loading reference model...")
                # For full fine-tuning, load a separate instance of the original model as reference
                # Ensure model_args is available, you might need to pass it to __init__
                if self.model_args is None:
                    raise ValueError("ModelArguments must be provided to SFTKLTrainer for loading a reference model in full finetuning.")

                # Determine the torch_dtype for the reference model
                if self.training_args.fp16:
                    ref_model_dtype = torch.float16
                elif self.training_args.bf16:
                    ref_model_dtype = torch.bfloat16
                else:
                    ref_model_dtype = torch.float32 
                
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    self.model_args.model_name_or_path,
                    trust_remote_code=self.model_args.trust_remote_code,
                    torch_dtype=ref_model_dtype,
                    low_cpu_mem_usage=False
                )
                
                self.reference_model.to(self.args.device)
                self.reference_model.eval() # Set to evaluation mode
                for param in self.reference_model.parameters():
                    param.requires_grad = False # Freeze all parameters
                
                #if self.state.is_world_process_zero:
                #    print(f"Reference model loaded and frozen: {self.model_args.model_name_or_path}")
                #    print(f"Model weights: {self.reference_model.get_input_embeddings().weight}")
                #    import pdb; pdb.set_trace() # <<< DEBUG

        elif self.use_kl_loss and self.kl_beta == 0:
            print("SFTKLTrainer: use_kl_loss is True, but kl_beta is 0. KL loss will not be applied.")

    @override
    def compute_loss(
        self, 
        model: "PreTrainedModel", 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Initialize the logging dict
        log_data = {}

        # 1. Standard SFT Loss (Cross-Entropy) from the LoRA-adapted model
        # The original Trainer's compute_loss for language modeling might need labels popped
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            sft_inputs = {**inputs, **loss_kwargs}
            outputs = model(**sft_inputs)
        else:
            outputs = model(**inputs)
            
        attention_mask = inputs["attention_mask"]
        token_lengths = attention_mask.sum(dim=1)  # shape: (batch_size,)
        max_token_length = token_lengths.max().item()
        if max_token_length > self.max_txt_len:
            self.max_txt_len = max_token_length
            if self.state.is_world_process_zero:
                print('Max token length updated:', self.max_txt_len)
                print('max_token_length:', max_token_length)
        

        sft_loss = outputs.loss # This is the primary SFT loss
        current_logits = outputs.logits # Logits from the LoRA-adapted model

        total_loss = sft_loss
        kl_divergence_raw = torch.tensor(0.0, device=sft_loss.device) # Initialize

        # 2. KL Divergence Loss (if enabled and model is PeftModel)
        if self.use_kl_loss and self.kl_beta > 0: # and isinstance(model, PeftModel):
            base_model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            reference_logits = None

            if self.finetuning_args.finetuning_type != "full":
                # PEFT/LoRA case: disable adapters to get reference logits
                peft_model_instance = model.module if hasattr(model, 'module') else model
                if not isinstance(peft_model_instance, PeftModel):
                     # This case should ideally be caught in __init__, but as a safeguard:
                    if self.state.is_world_process_zero:
                        print("Warning: KL loss enabled but model is not PeftModel in PEFT mode. Skipping KL.")
                    reference_logits = current_logits.detach() # Fallback: use current logits (KL will be 0)
                else:
                    original_training_state_hf_model = peft_model_instance.training
                    peft_model_instance.eval()
                    with torch.no_grad():
                        with peft_model_instance.disable_adapter():
                            reference_outputs = peft_model_instance(**base_model_inputs)
                            reference_logits = reference_outputs.logits
                    if original_training_state_hf_model:
                        peft_model_instance.train()
            else: # Full fine-tuning case
                if self.reference_model is not None:
                    with torch.no_grad():
                        # Ensure reference_model is on the same device as inputs
                        # This should have been handled in __init__ or needs to be ensured here if devices can change
                        if next(self.reference_model.parameters()).device != base_model_inputs['input_ids'].device:
                             self.reference_model.to(base_model_inputs['input_ids'].device)

                        reference_outputs = self.reference_model(**base_model_inputs)
                        reference_logits = reference_outputs.logits
                else:
                    # Should not happen if __init__ logic is correct and use_kl_loss is true
                    if self.state.is_world_process_zero:
                        print("Warning: Reference model not available for full finetuning KL. Skipping KL.")
                    reference_logits = current_logits.detach() # Fallback: use current logits (KL will be 0)
            
            if reference_logits is None: # Should be caught by earlier warnings, but as a final check
                if self.state.is_world_process_zero:
                    print("Error: Reference logits could not be obtained. Skipping KL loss.")
            else:
                labels = inputs.get("labels")
                if labels is None:
                    if self.state.is_world_process_zero:
                        print("Warning: Labels not found in inputs for KL divergence calculation. Skipping KL loss.")
                else:
                    shift_current_logits = current_logits[..., :-1, :].contiguous()
                    shift_reference_logits = reference_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss_mask = shift_labels.ne(self.label_ignore_index if hasattr(self, 'label_ignore_index') else -100)

                    active_current_logits = shift_current_logits.view(-1, shift_current_logits.size(-1))[loss_mask.view(-1)]
                    active_reference_logits = shift_reference_logits.view(-1, shift_reference_logits.size(-1))[loss_mask.view(-1)]

                    if active_current_logits.numel() > 0 and active_reference_logits.numel() > 0:
                        current_log_probs = F.log_softmax(active_current_logits, dim=-1) 
                        reference_log_probs = F.log_softmax(active_reference_logits, dim=-1)
                        
                        kl_divergence_raw = F.kl_div(
                            current_log_probs,
                            reference_log_probs,
                            reduction='batchmean',
                            log_target=True
                        )
                        log_data['kl_loss'] = kl_divergence_raw.detach().item()
                        total_loss = sft_loss + self.kl_beta * kl_divergence_raw

                        if self.state.is_world_process_zero:
                            num_elements_to_log = 10
                            actual_elements_current = min(num_elements_to_log, current_log_probs.numel())
                            actual_elements_reference = min(num_elements_to_log, reference_log_probs.numel())
                            log_data['current_log_probs_head'] = current_log_probs.view(-1)[:actual_elements_current].detach().cpu().tolist()
                            log_data['reference_log_probs_head'] = reference_log_probs.view(-1)[:actual_elements_reference].detach().cpu().tolist()
                    else:
                        if self.state.is_world_process_zero:
                            print("Warning: No active tokens found for KL divergence after masking. Skipping KL for this step.")
        
        # Logging
        if self.state.is_world_process_zero:
            log_data['sft_loss'] = sft_loss.detach().item()
            self.log(log_data)

        return (total_loss, outputs) if return_outputs else total_loss