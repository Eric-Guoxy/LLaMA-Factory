# LLaMA-Factory (Modified)

LLaMA-Factory is a collection of scripts to train and evaluate models based on the LLaMA architecture. It is designed to be easy to use and flexible, allowing users to customize their training and evaluation processes.

## Features
- Full/lora SFT training.
- Add KL loss to SFT training. (Optional)
- Automatically train and evaluate models.
- Support for multiple datasets.
- Support for multiple training configurations.

## Requirements
| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.45.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |

Note: Since we automate the training and evaluation process, you should also install `vllm` for evaluation. Also, `flash-attn` is also recommended for faster training.

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

Make sure to create a conda environment with Python 3.10 and activate it:

```bash
conda create -n llama python=3.10
conda activate llama
```

```bash
git clone git@github.com:Eric-Guoxy/LLaMA-Factory.git
git branch -M develop
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

> [!TIP]
> Use `pip install -e . --no-deps --no-build-isolation` to resolve package conflicts.

If you encounter import issues with `math_verify`, you can install it manually:

```bash
pip install math_verify
```

If you encounter build issues with `flash-attn`, you can install it manually using wheel files:

```bash
pip install flash_attn-xxx-linux_x86_64.whl
```

If you encounter issues with CUDA, especially correlated to `NVSwitch` or `NVLink`, you can try the following:

```bash
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
```

The `NCCL_DEBUG` environment variable can be set to `INFO`, `WARN`, or `ERROR` to control the verbosity of the NCCL library. Setting it to `INFO` will provide detailed information about the NCCL operations, which can help in debugging issues related to CUDA and NCCL.

If the CUDA error persists, it may be related to imcompatible cublas version (especially for GPUs using Hopper architecture, e.g. H20) you can try the following:

```bash
pip install nvidia-cublas-cu12==12.4.5.8
```

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can use datasets on HuggingFace / ModelScope / Modelers hub, load the dataset in local disk, or specify a path to s3/gcs cloud storage.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset.

You can also use **[Easy Dataset](https://github.com/ConardLi/easy-dataset)** or **[GraphGen](https://github.com/open-sciencelab/GraphGen)** to create synthetic data for fine-tuning.

### Quickstart

Use the following 3 commands to run LoRA **fine-tuning**, **inference** and **merging** of the Llama3-8B-Instruct model, respectively.

```bash
llamafactory-cli train examples/train_lora/xxx.yaml
llamafactory-cli chat examples/inference/xxx.yaml
llamafactory-cli export examples/merge_lora/xxx.yaml
```

See [examples/README.md](examples/README.md) for advanced usage (including distributed training).

> [!TIP]
> Use `llamafactory-cli help` to show help information.

We provide a few examples of training and evaluation scripts in the `examples`  and `evaluation` directory. You can use these scripts as a starting point for your own training and evaluation processes.

To run the training and evaluation scripts, you can use the following command (suppose you are in the root directory of the repository):

```bash
cd examples/train_full
bash run_7b_sft.sh
```
This will run the training and evaluation scripts for the Qwen2.5-Math-7B model. You can modify the scripts to suit your needs.

The scripts that ends with `_sft.sh` are for pure fine-tuning, while the scripts that ends with `_kl.sh` are for fine-tuning with KL loss.

We have also provided scripts for training the DeepSeek-R1-Distill-Qwen-1.5B model. Please make sure to overwrite the `model_name_or_path` to be the correct model path/name on your device.
