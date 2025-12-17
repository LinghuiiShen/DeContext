#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Multi-Prompt Reconstruction Error
Combines multiple prompts with reconstruction error maximization.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

logger = get_logger(__name__)

import re

class MultiPromptDataset(Dataset):
    """Dataset with multiple prompts and corresponding targets."""
    
    def __init__(self, condition_images_dir, target_base_dir, prompts, resolution=512, condition_image_name=None):
        self.condition_images_dir = Path(condition_images_dir)
        self.target_base_dir = Path(target_base_dir)
        self.prompts = prompts
        self.resolution = resolution
        
        # Load condition images
        all_cond_files = sorted([
            f for f in self.condition_images_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ])

        if condition_image_name:
            self.cond_files = [f for f in all_cond_files if f.name == condition_image_name]
            if not self.cond_files:
                raise ValueError(f"Condition image '{condition_image_name}' not found in {self.condition_images_dir}")
            logger.info(f"Processing single condition image: {condition_image_name}")
        else:
            self.cond_files = all_cond_files
            logger.info(f"Processing all {len(self.cond_files)} condition images")
        
        # Build mapping: {cond_stem: {prompt_idx: target_path}}
        self.target_mapping = {}
        
        for cond_file in self.cond_files:
            cond_stem = cond_file.stem 
            self.target_mapping[cond_stem] = {}
            for prompt_idx in range(len(prompts)):
               # target_pattern = f"{cond_stem}_prompt_{prompt_idx:02d}.*"              
                target_pattern = f"prompt_{prompt_idx:02d}.*"
                target_files = list(self.target_base_dir.glob(target_pattern))
                
                if target_files:
                    self.target_mapping[cond_stem][prompt_idx] = target_files[0]
                else:
                    pass
            
            num_targets = len(self.target_mapping[cond_stem])
            if num_targets > 0:
                logger.info(f"✓ Loaded {num_targets} targets for {cond_stem}")
            else:
                logger.error(f"✗ No targets found for {cond_stem} in {self.target_base_dir}")
                logger.error(f"  Expected files like: {cond_stem}_prompt_00.png, {cond_stem}_prompt_01.png, ...")
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.cond_files)
    
    def __getitem__(self, idx):
        cond_path = self.cond_files[idx]
        cond_stem = cond_path.stem
        
        # Load condition image
        cond_image = Image.open(cond_path).convert('RGB')
        cond_image = self.image_transforms(cond_image)
        
        # Load all target images for this condition
        targets = {}
        for prompt_idx, target_path in self.target_mapping[cond_stem].items():
            target_image = Image.open(target_path).convert('RGB')
            target_image = self.image_transforms(target_image)
            targets[prompt_idx] = target_image
        
        return {
            'condition_image': cond_image,
            'targets': targets,
            'cond_path': str(cond_path),
            'cond_stem': cond_stem
        }
    


def encode_prompt_flux_kontext(text_encoders, tokenizers, prompt, max_sequence_length=512, device=None):
    """Encode text prompt using dual text encoders."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    # CLIP encoding
    text_inputs_clip = tokenizers[0](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        pooled_prompt_embeds = text_encoders[0](
            text_inputs_clip.input_ids.to(device)
        ).pooler_output
    
    # T5 encoding
    text_inputs_t5 = tokenizers[1](
        prompt, padding="max_length", max_length=max_sequence_length, 
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoders[1](text_inputs_t5.input_ids.to(device))[0]
    
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
    
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(timesteps, scheduler, n_dim=4, dtype=torch.float32, device=None):
    """Get noise sigmas for given timesteps."""
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if len(matches) > 0:
            step_indices.append(matches[0].item())
        else:
            step_indices.append(torch.argmin(torch.abs(schedule_timesteps - t)).item())
    
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def get_prompt_pool():
    """Returns the 20-prompt pool."""
    prompts = [
    # ===== 1. Expression Changes (10) =====
        
        "Make this person smile happily.",
        "Make this person look angry.",
        "Make this person look surprised.",
        "Make this person look sad.",
        "Give this person a serious expression.",
        "Make this person laugh joyfully.",
        "Make this person look worried.",
        "Make this person look excited.",
        "Make this person look tired.",
        "Make this person look confused.",
    

    # ===== 2. Accessory Addition (10) =====

        "Add glasses to this person.",
        "Add sunglasses to this person.",
        "Add a hat to this person.",
        "Add a scarf to this person.",
        "Add a necklace to this person.",
        "Add earrings to this person.",
        "Add a cap to this person.",
        "Add a headband to this person.",
        "Add a watch to this person.",
        "Add a tie to this person.",


    # ===== 3. Posture and Movement (10) =====

        "Make this person wave their hand.",
        "Make this person cross their arms.",
        "Make this person point forward.",
        "Make this person give a thumbs up.",
        "Make this person put hands on hips.",
        "Make this person cover their mouth.",
        "Make this person hold up a peace sign.",
        "Make this person shrug their shoulders.",
        "Make this person touch their chin.",
        "Make this person salute.",


    # ===== 4. Style Changes (10) =====

        "Make this person look older.",
        "Make this person look younger.",
        "Add a beard to this person.",
        "Add a mustache to this person.",
        "Make this person's hair longer.",
        "Make this person's hair shorter.",
        "Add makeup to this person.",
        "Add freckles to this person.",
        "Make this person's skin tanned.",
        "Add stubble to this person.",


    # ===== 5. Scene Change (10) =====
        "Place this person on a tropical beach with palm trees and ocean waves.",
        "Put this person in a snowy mountain landscape with pine trees.",
        "Move this person to a bustling city street with skyscrapers.",
        "Put this person in a beach setting.",
        "Put this person in a snowy mountain.",
        "Put this person in a city street.",
        "Put this person in a forest.",
        "Put this person in a coffee shop.",
        "Put this person in a library.",
        "Put this person in a park.",


    # ===== 6. Combined Prompts (10) =====
        "Make this person smile and wave.",
        "Add glasses and make this person look serious.",
        "Make this person look older with a beard.",
        "Add a hat and make this person point forward.",
        "Make this person look surprised with hands on face.",
        "Add sunglasses and make this person give a thumbs up.",
        "Make this person look younger and smile happily.",
        "Add a scarf and make this person cross their arms.",
        "Make this person look tired and rub their eyes.",
        "Add earrings and make this person look confident."
        
    ]

    return prompts
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                       default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./condition_images_Celeb")
    parser.add_argument("--target_base_dir", type=str, default="./clean_result_multi/27")
    parser.add_argument("--condition_image_name", type=str, default=None, help="Specific condition image to process, e.g., '27.jpg'")  
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=600)
    parser.add_argument("--output_dir", type=str, default="adversarial_01_sim_multi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                       datefmt="%H:%M:%S", level=logging.INFO)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get prompt pool
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts")
    
    # Load tokenizers and models
    tokenizer_clip = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    
    logger.info("Loading models...")
    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_t5 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Set precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    text_encoder_clip.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    text_encoder_clip.eval()
    text_encoder_t5.eval()
    vae.eval()
    transformer.eval()
    
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    transformer.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()
    
    # Load dataset
    dataset = MultiPromptDataset(
        condition_images_dir=args.condition_images_dir,
        target_base_dir=args.target_base_dir,
        prompts=prompt_pool,
        resolution=args.resolution,
        condition_image_name=args.condition_image_name  
    )
    
    logger.info(f"Loaded {len(dataset)} condition images")
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    
    text_encoders = [text_encoder_clip, text_encoder_t5]
    tokenizers = [tokenizer_clip, tokenizer_t5]
    
    # Pre-encode all prompts
    logger.info("Pre-encoding all prompts...")
    prompt_embeds_pool = []
    pooled_prompt_embeds_pool = []
    
    for prompt in prompt_pool:
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_flux_kontext(
                text_encoders, tokenizers, prompt, args.max_sequence_length, accelerator.device
            )
            prompt_embeds_pool.append(prompt_embeds)
            pooled_prompt_embeds_pool.append(pooled_prompt_embeds)
    
    logger.info(f"Pre-encoded {len(prompt_embeds_pool)} prompts")
    
    # Free text encoders
    del text_encoder_clip, text_encoder_t5, tokenizer_clip, tokenizer_t5
    torch.cuda.empty_cache()


    # Process each condition image
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        targets_dict = sample['targets']
        cond_path = sample['cond_path']
        cond_stem = sample['cond_stem']
        
        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")
        logger.info(f"Available targets: {len(targets_dict)}")
        
        # Pre-encode all target latents for this condition image
        target_latents_pool = {}
        for prompt_idx, target_image in targets_dict.items():
            target_image = target_image.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
            with torch.no_grad():
                target_latents = vae.encode(target_image).latent_dist.mode()
                target_latents = (target_latents - vae_config_shift_factor) * vae_config_scaling_factor
                target_latents_pool[prompt_idx] = target_latents
        
        # Initialize perturbed condition
        perturbed_condition = condition_image.clone().detach()
        perturbed_condition.requires_grad_(True)
        original_condition = condition_image.clone().detach()
        
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")
        
        guidance = torch.tensor([3.5], device=accelerator.device)
        
        for step in progress_bar:
            # Randomly sample a prompt (and corresponding target)
            available_prompts = list(targets_dict.keys())
            random_prompt_idx = available_prompts[torch.randint(0, len(available_prompts), (1,)).item()]
            
            prompt_embeds = prompt_embeds_pool[random_prompt_idx]
            pooled_prompt_embeds = pooled_prompt_embeds_pool[random_prompt_idx]
            target_latents = target_latents_pool[random_prompt_idx]
            
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                device=accelerator.device, dtype=prompt_embeds.dtype
            )
            
            perturbed_condition.requires_grad = True
            
            # Encode perturbed condition
            cond_latents = vae.encode(perturbed_condition).latent_dist.mode()
            cond_latents = (cond_latents - vae_config_shift_factor) * vae_config_scaling_factor
            
            # Sample timestep
            bsz = 1
            timesteps = torch.randint(
                980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
            ).long()
            
            # Add noise to target
            noise = torch.randn_like(target_latents)
            sigmas = get_sigmas(timesteps, scheduler, target_latents.ndim, target_latents.dtype, accelerator.device)
            noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise
            
            # Prepare latent IDs
            latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                target_latents.shape[0],
                target_latents.shape[2] // 2,
                target_latents.shape[3] // 2,
                accelerator.device,
                weight_dtype,
            )
            
            cond_latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
                cond_latents.shape[0],
                cond_latents.shape[2] // 2,
                cond_latents.shape[3] // 2,
                accelerator.device,
                weight_dtype,
            )
            cond_latent_ids[..., 0] = 1
            
            combined_latent_ids = torch.cat([latent_image_ids, cond_latent_ids], dim=0)
            
            # Pack latents
            packed_noisy_target = FluxKontextPipeline._pack_latents(
                noisy_target_latents,
                batch_size=target_latents.shape[0],
                num_channels_latents=target_latents.shape[1],
                height=target_latents.shape[2],
                width=target_latents.shape[3],
            )
            
            packed_cond_latents = FluxKontextPipeline._pack_latents(
                cond_latents,
                batch_size=cond_latents.shape[0],
                num_channels_latents=cond_latents.shape[1],
                height=cond_latents.shape[2],
                width=cond_latents.shape[3],
            )
            
            packed_input = torch.cat([packed_noisy_target, packed_cond_latents], dim=1)
            
            # Forward pass
            model_pred = transformer(
                hidden_states=packed_input,
                timestep=timesteps / 1000,
                guidance=guidance.expand(bsz),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_latent_ids,
                return_dict=False,
            )[0]
            
            # Extract target prediction
            model_pred = model_pred[:, :packed_noisy_target.shape[1]]
            
            # Unpack prediction
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            model_pred = FluxKontextPipeline._unpack_latents(
                model_pred,
                height=target_latents.shape[2] * vae_scale_factor,
                width=target_latents.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )
            
            # Flow matching loss
            target_velocity = noise - target_latents
            loss = F.mse_loss(model_pred.float(), target_velocity.float())
            
            # Backward
            transformer.zero_grad()
            vae.zero_grad()
            loss.backward()
            alpha = args.alpha
            eps = args.eps
            
            print(perturbed_condition.grad.shape)
            print(f"Grad max: {perturbed_condition.grad.max().item():.6f}")
            print(f"Grad min: {perturbed_condition.grad.min().item():.6f}")
 
           
            grad = perturbed_condition.grad


            # grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-8)
            # g = mu * g + grad_norm

            # Gradient ascent to maximize the flow matching loss
            adv_condition = perturbed_condition +  alpha * grad.sign()
            eta = torch.clamp(adv_condition - original_condition, min=-eps, max=+eps)
            perturbed_condition = torch.clamp(original_condition + eta, min=-1, max=+1).detach_()
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()

            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()
            
            progress_bar.set_postfix({
                'prompt_idx': random_prompt_idx,
                'loss': f'{loss.item():.4f}',
                'pert': f'{perturbation_norm:.4f}',
                'timestep': timesteps.item()
            })
            
            # Clear memory
            del loss, model_pred, cond_latents, packed_input, noisy_target_latents, noise
            torch.cuda.empty_cache()
            
            # Save intermediate results
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    perturbed_np = perturbed_condition.detach().squeeze(0).cpu().float().numpy()
                    perturbed_np = (perturbed_np + 1.0) * 127.5
                    perturbed_np = np.transpose(perturbed_np, (1, 2, 0)).astype(np.uint8)
                    perturbed_pil = Image.fromarray(perturbed_np)
                    
                    intermediate_path = os.path.join(args.output_dir, f"adversarial_{Path(cond_path).stem}_step_{step+1}.png")
                    perturbed_pil.save(intermediate_path)
                    
                    perturbation = perturbed_condition - original_condition
                    delta_vis = perturbation.detach().squeeze(0).cpu().float().numpy()
                    delta_vis = (delta_vis + args.eps) / (2 * args.eps)
                    delta_vis = (delta_vis * 255).astype(np.uint8)
                    delta_vis = np.transpose(delta_vis, (1, 2, 0))
                    delta_pil = Image.fromarray(delta_vis)
                    
                    delta_intermediate_path = os.path.join(args.output_dir, f"perturbation_{Path(cond_path).stem}_step_{step+1}.png")
                    delta_pil.save(delta_intermediate_path)
                    
                    logger.info(f"Saved intermediate result at step {step+1}: {intermediate_path}")
        
        # Clear memory for next image
        del perturbed_condition, original_condition, target_latents_pool
        torch.cuda.empty_cache()
    
    logger.info("Multi-prompt adversarial attack completed!")


if __name__ == "__main__":
    main()