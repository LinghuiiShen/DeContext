# DeContext as Defense: Safe Image Editing in Diffusion Transformers

Linghui Shen, Mingyue Cui, [Xingyi Yang](https://adamdad.github.io/)

## Overview
<img width="1427" height="560" alt="image" src="https://github.com/user-attachments/assets/6b4b6749-ddc0-433c-91cb-f81cd69b40c9" />


## Environment Setup

Navigate to the project directory:
```bash
cd DeContext
```

Create and activate conda environment:
```bash
conda create -n decontext python=3.12
conda activate decontext
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Attack on Flux Kontext

Run the attack script:
```bash
bash ./scripts/attack_kontext.sh
```

Run inference:
```bash
python ./inference/kontext_inference.py
```

### 2. Attack on Step1X-Edit

#### Download Required Models

Download the following models and place them in `./attack/attack_Step1X_Edit/models`:

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)

For more details, refer to the [Step1X-Edit repository](https://github.com/stepfun-ai/Step1X-Edit).

#### Run Attack
```bash
bash ./scripts/attack_step1x.sh
```

#### Run Inference
```bash
python ./inference/step1x_inference.py
```

## Acknowledgement

Our work is built upon [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit). Thanks for their excellent work!
