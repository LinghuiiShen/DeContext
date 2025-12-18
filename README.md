# DeContext as Defense: Safe Image Editing in Diffusion Transformers

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/yourpaper)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://linghuiishen.github.io/decontext_project_page/)

</div>

---

<p align="center">
  <img width="70%" alt="image" src="https://github.com/user-attachments/assets/8fb1504c-926e-462e-94dd-bee09f0c452f" />
</p>

## 🖼️ Overview
DeContext is a defense for DiT-based in-context image editing that effectively detaches the context from the input, safeguarding images against unauthorized manipulation through subtle perturbation injection.
<img width="1427" height="560" alt="image" src="https://github.com/user-attachments/assets/294b468b-f0e0-43a7-9902-80e90bdb15f0" />


> **DeContext as Defense: Safe Image Editing in Diffusion Transformers**  
> Linghui Shen, Mingyue Cui, [Xingyi Yang](https://adamdad.github.io/)  
> The Hong Kong Polytechnic University


## 🛠️ Environment Setup

```bash
cd DeContext
```

Create and activate conda environment (Optional):
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

#### Run Attack
```bash
bash ./scripts/attack_step1x.sh
```

#### Run Inference
```bash
python ./inference/step1x_inference.py
```

## 🙏 Acknowledgement

Our work is built upon [Diffusers](https://github.com/huggingface/diffusers) and [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit). Thanks for their excellent work!

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
