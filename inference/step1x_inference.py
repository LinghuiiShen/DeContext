import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent  # DeContext 目录
sys.path.insert(0, str(project_root))

from attack.attack_Step1X_Edit.inference import ImageGenerator

def main():
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./attack/attack_Step1X_Edit/models", help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to the output directory')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=512, type=int)
    parser.add_argument('--offload', action='store_true', help='Use offload for large models')
    parser.add_argument('--quantized', action='store_true', help='Use fp8 model weights')
    parser.add_argument('--version', type=str, default='v1.1', choices=['v1.0', 'v1.1'])
    
    args = parser.parse_args()

    image_path = "./example/170.jpg"
    prompt = "A photo of this person"
    seed = 88

    os.makedirs(args.output_dir, exist_ok=True)

    if args.version == 'v1.0':
        ckpt_name = 'step1x-edit-i1258.safetensors'
    elif args.version == 'v1.1':
        ckpt_name = 'step1x-edit-v1p1-official.safetensors'

    print("Loading model...")
    image_edit = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, ckpt_name),
        qwen2vl_model_path=os.path.join(args.model_path, 'Qwen2.5-VL-7B-Instruct'),
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        mode="torch",
        version=args.version,
    )
    print("Model loaded successfully!")

    ref_image = Image.open(image_path).convert("RGB")

    print(f"Generating image with prompt: '{prompt}', seed: {seed}")
    image = image_edit.generate_image(
        prompt=prompt,
        negative_prompt="",
        ref_images=ref_image,
        num_samples=1,
        num_steps=args.num_steps,
        cfg_guidance=args.cfg_guidance,
        seed=seed,
        show_progress=True,
        size_level=args.size_level,
    )[0]

    output_path = os.path.join(args.output_dir, "output.jpg")
    image.save(output_path, quality=95)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()