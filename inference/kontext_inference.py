#!/usr/bin/env python3
"""
Simplified Flux Kontext script - Single prompt generation
"""

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import argparse
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description="Flux Kontext - Single Image Generation")
    parser.add_argument("--context_image", type=str, 
                       default="perturbed/adversarial_170.png",
                    #    default="example/170.jpg",
                       help="Context image path")
    parser.add_argument("--prompt", type=str, 
                       default="A photo of this person",
                       help="Text prompt for generation")
    parser.add_argument("--model_path", type=str, 
                       default="black-forest-labs/FLUX.1-Kontext-dev", 
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default="./outputs",
                       help="Output directory")
    parser.add_argument("--guidance_scale", type=float, default=3.5, 
                       help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=20, 
                       help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--width", type=int, default=512, 
                       help="Output width")
    parser.add_argument("--height", type=int, default=512, 
                       help="Output height")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load pipeline
    print(f"Loading model: {args.model_path}")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    )
    pipe.to(args.device)
    print("Model loaded successfully")
    
    # Load context image
    # Convert to absolute path if relative
    context_image_path = args.context_image
    if not os.path.isabs(context_image_path):
        context_image_path = os.path.abspath(context_image_path)
    
    if not os.path.exists(context_image_path):
        raise FileNotFoundError(f"Context image not found: {context_image_path}")
    
    context_image = load_image(context_image_path)
    context_image_name = Path(args.context_image).stem
    
    print(f"\nContext image: {args.context_image}")
    print(f"Image size: {context_image.size}")
    print(f"Prompt: {args.prompt}")
    
    # Set random seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Generate image
    print("\nGenerating...")
    image = pipe(
        image=context_image,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        max_area=512*512,
        _auto_resize=False,
        num_inference_steps=args.num_inference_steps,
        generator=generator
    ).images[0]
    
    # Save output
    output_filename = f"{context_image_name}_output.png"
    output_path = os.path.join(args.output_dir, output_filename)
    image.save(output_path)
    
    print(f"\n✓ Generation complete!")
    print(f"Saved to: {output_path}")
    
    # Clean up
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()