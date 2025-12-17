# DeContext as Defense: Safe Image Editing in Diffusion Transformers

<div align="center">

### Linghui Shen<sup>1</sup>, Mingyue Cui<sup>1</sup>, [Xingyi Yang](https://adamdad.github.io/)<sup>1</sup>

<sup>1</sup>The Hong Kong Polytechnic University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/yourpaper)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://yourprojectpage.github.io)

</div>

---

## 📋 TL;DR

**DeContext** protects images from unauthorized manipulation by injecting targeted perturbations that disrupt multimodal attention pathways, effectively decoupling the link between input and output. 

## 🖼️ Overview

<img width="1427" height="560" alt="image" src="https://github.com/user-attachments/assets/294b468b-f0e0-43a7-9902-80e90bdb15f0" />


In-context diffusion models allow users to modify images with remarkable ease and realism. However, the same power raises serious privacy concerns: personal images can be easily manipulated for identity impersonation, misinformation, or other malicious uses, all without the owner's consent. While prior work has explored input perturbations to protect against misuse in personalized text-to-image generation, the robustness of modern, large-scale in-context DiT-based models remains largely unexamined. 

In this paper, we propose **DeContext**, a new method to safeguard input images from unauthorized in-context editing. Our key insight is that contextual information from the source image propagates to the output primarily through multimodal attention layers. By injecting small, targeted perturbations that weaken these cross-attention pathways, DeContext breaks this flow, effectively decoupling the link between input and output. This simple defense is both efficient and robust. We further show that early denoising steps and specific transformer blocks dominate context propagation, which allows us to concentrate perturbations where they matter most. Experiments on *Flux Kontext* and *Step1X-Edit* show that DeContext consistently blocks unwanted image edits while preserving visual quality.

## 🛠️ Environment Setup

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

## 🚀 How to Run

### 1️⃣ Attack on Flux Kontext

Run the attack script:
```bash
bash ./scripts/attack_kontext.sh
```

Run inference:
```bash
python ./inference/kontext_inference.py
```

### 2️⃣ Attack on Step1X-Edit

#### 📥 Download Required Models

Download the following models and place them in `./attack/attack_Step1X_Edit/models`:

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)

> **Note:** For more details, refer to the [Step1X-Edit repository](https://github.com/stepfun-ai/Step1X-Edit).

#### ⚔️ Run Attack
```bash
bash ./scripts/attack_step1x.sh
```

#### 🔍 Run Inference
```bash
python ./inference/step1x_inference.py
```

## 🙏 Acknowledgement

Our work is built upon [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit). Thanks for their excellent work!

## 📝 Citation

If you find this work useful, please consider citing:
```bibtex
@article{decontext2024,
  title={DeContext as Defense: Safe Image Editing in Diffusion Transformers},
  author={Shen, Linghui and Cui, Mingyue and Yang, Xingyi},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
