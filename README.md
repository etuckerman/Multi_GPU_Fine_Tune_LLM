# Multi_GPU_Fine_Tune_LLM

This repository contains code and configurations for fine-tuning language models using DeepSpeed, LoRA, and advanced quantization techniques. The primary components include configuration files, a training script, a Jupyter notebook for experimentation, and utility functions.

![GPU Utilization](https://github.com/etuckerman/Multi_GPU_Fine_Tune_LLM/blob/main/images/dual_gpu.png)


## Project Structure

- **`deepspeed_config_z3_qlora.yaml`**: Configuration file for DeepSpeed. It includes settings for mixed precision, distributed training, and optimizer offloading.
- **`notepad_finetune.ipynb`**: Jupyter notebook for running the fine-tuning process. It installs necessary libraries, sets up the environment, and launches the training script with specific parameters.
- **`train.py`**: Main training script that sets up and runs the fine-tuning process. It handles argument parsing, model and tokenizer creation, dataset preparation, and training loop.
- **`utils.py`**: Utility functions for dataset creation and model preparation. It includes functions for applying chat templates, loading datasets, and preparing models with LoRA and quantization configurations.

## Requirements

To run this project, you need the following Python packages:

- `accelerate`
- `transformers`
- `peft`
- `deepspeed`
- `trl`
- `bitsandbytes`
- `flash-attn`

You can install these packages using the provided Jupyter notebook:

```bash
pip install -q accelerate transformers peft deepspeed trl bitsandbytes flash-attn --no-build-isolation
