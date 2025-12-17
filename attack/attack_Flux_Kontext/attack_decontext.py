#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Context Suppression
Minimizes attention weight proportion allocated to context image.
"""

import argparse
import logging
import math
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
from diffusers.models.embeddings import apply_rotary_emb
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
logger = get_logger(__name__)


class AdversarialDataset(Dataset):
    def __init__(self, condition_images_dir, prompts, resolution=512):
        self.condition_images_dir = Path(condition_images_dir)
        self.prompts = prompts 
        self.resolution = resolution

        self.cond_files = sorted([f for f in self.condition_images_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
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
        
        cond_image = Image.open(cond_path).convert('RGB')
        cond_image = self.image_transforms(cond_image)
        
        return {
            'condition_image': cond_image,
            'cond_path': str(cond_path)
        }


def encode_prompt_flux_kontext(text_encoders, tokenizers, prompt, max_sequence_length=512, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    text_inputs_clip = tokenizers[0](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        pooled_prompt_embeds = text_encoders[0](text_inputs_clip.input_ids.to(device)).pooler_output
    
    text_inputs_t5 = tokenizers[1](
        prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoders[1](text_inputs_t5.input_ids.to(device))[0]
    
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(timesteps, scheduler, n_dim=4, dtype=torch.float32, device=None):
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


class AttentionHook:
    def __init__(self):
        self.qkv_normalized = {}
        self.hooks = []
    
    def register_hooks(self, transformer, selected_layers=None):
        if selected_layers is None:
            selected_layers = {
                'single_blocks': list(range(0, 26))
            }
        
        def create_norm_hook(layer_name, proj_type):
            def hook_fn(module, input, output):
                if output is not None and hasattr(output, 'shape'):
                    self.qkv_normalized[f"{layer_name}_{proj_type}"] = output.clone()
            return hook_fn
        
        for i in selected_layers.get('double_blocks', []):
            if i < len(transformer.transformer_blocks):
                block = transformer.transformer_blocks[i]
                if hasattr(block, 'attn'):
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'double_{i}', proj_type)
                            )
                            self.hooks.append(hook)
        
        for i in selected_layers.get('single_blocks', []):
            if i < len(transformer.single_transformer_blocks):
                block = transformer.single_transformer_blocks[i]
                if hasattr(block, 'attn'):
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'single_{i}', proj_type)
                            )
                            self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.qkv_normalized = {}


# def compute_context_proportion(qkv_normalized_dict, image_rotary_emb):
#     """
#     Compute proportion of attention weight allocated to context image.
    
#     Returns:
#         context_prop: Average proportion across all layers [scalar]
#     """
#     freqs_cos, freqs_sin = image_rotary_emb
#     proportions = []
    
#     for layer_name in qkv_normalized_dict:
#         if not layer_name.endswith('_q'):
#             continue
        
#         layer_base = layer_name.replace('_q', '')
#         q_key = f"{layer_base}_q"
#         k_key = f"{layer_base}_k"
        
#         if q_key not in qkv_normalized_dict or k_key not in qkv_normalized_dict:
#             continue
        
#         q_norm = qkv_normalized_dict[q_key]
#         k_norm = qkv_normalized_dict[k_key]
        
#         batch_size, seq_len, heads, dim_head = q_norm.shape
        
#         # Determine sequence structure 
#         # (default with 512*512, the sequence length needs to be adjusted with context images of different resolutions)
#         if 'double_' in layer_base:
#             if seq_len != 2048:
#                 continue
#             target_start, target_end = 0, 1024
#             text_start, text_end = None, None
#             context_start, context_end = 1024, 2048
#         elif 'single_' in layer_base:
#             if seq_len != 2560:
#                 continue
#             target_start, target_end = 512, 1536
#             text_start, text_end = 0, 512
#             context_start, context_end = 1536, 2560
#         else:
#             continue
        
#         # Apply RoPE
#         q_with_rope = apply_rotary_emb(q_norm, image_rotary_emb, sequence_dim=1)
#         k_with_rope = apply_rotary_emb(k_norm, image_rotary_emb, sequence_dim=1)
        
#         # Compute full attention scores
#         q_with_rope = q_with_rope.permute(0, 2, 1, 3)  # [B, H, S, D]
#         k_with_rope = k_with_rope.permute(0, 2, 1, 3)
#         print(q_with_rope.shape)
#         attn_scores = q_with_rope @ k_with_rope.transpose(-2, -1) / math.sqrt(dim_head)  # [B, H, S, S]
#         print(attn_scores.shape)
#         # Extract target queries
#         target_attn_scores = attn_scores[:, :, target_start:target_end, :]  # [B, H, 1024, S]
        
#         # Softmax over all keys (text + target + context)
#         attn_weights = F.softmax(target_attn_scores, dim=-1)  # [B, H, 1024, S]

#         # Extract context weights
#         context_weights = attn_weights[:, :, :, context_start:context_end]  # [B, H, 1024, 1024]
          
#         # Compute proportion: sum over context keys / total (which is 1 after softmax)
#         context_prop_per_query = context_weights.sum(dim=-1)  # [B, H, 1024]
        
#         # Average over heads and queries
#         layer_prop = context_prop_per_query.mean()  # scalar

#         proportions.append(layer_prop)
    
#     if len(proportions) == 0:
#         return torch.tensor(0.0)
    
#     return torch.stack(proportions).mean()

def compute_context_proportion(qkv_normalized_dict, image_rotary_emb):
    """
    Compute proportion of attention weight allocated to context image.
    
    Returns:
        context_prop: Average proportion across all layers [scalar]
    """
    freqs_cos, freqs_sin = image_rotary_emb
    proportions = []
    
    for layer_name in qkv_normalized_dict:
        if not layer_name.endswith('_q'):
            continue
        
        layer_base = layer_name.replace('_q', '')
        q_key = f"{layer_base}_q"
        k_key = f"{layer_base}_k"
        
        if q_key not in qkv_normalized_dict or k_key not in qkv_normalized_dict:
            continue
        
        q_norm = qkv_normalized_dict[q_key]
        k_norm = qkv_normalized_dict[k_key]
        
        batch_size, seq_len, heads, dim_head = q_norm.shape
        
        # Determine sequence structure 
        # (default with 512*512, the sequence length needs to be adjusted with context images of different resolutions)
        if 'double_' in layer_base:
            if seq_len != 2048:
                continue
            target_start, target_end = 0, 1024
            text_start, text_end = None, None
            context_start, context_end = 1024, 2048
        elif 'single_' in layer_base:
            if seq_len != 2560:
                continue
            target_start, target_end = 512, 1536
            text_start, text_end = 0, 512
            context_start, context_end = 1536, 2560
        else:
            continue
        
        # Apply RoPE
        q_with_rope = apply_rotary_emb(q_norm, image_rotary_emb, sequence_dim=1)
        k_with_rope = apply_rotary_emb(k_norm, image_rotary_emb, sequence_dim=1)
        
        # Compute full attention scores
        q_target_with_rope = q_with_rope[:, target_start:target_end, :, :]  # [B, 1024, H, D]
        q_target_with_rope = q_target_with_rope.permute(0, 2, 1, 3)  # [B, H, 1024, D]
        k_with_rope = k_with_rope.permute(0, 2, 1, 3)
        
        # print(q_target_with_rope.shape)
        target_attn_scores = q_target_with_rope @ k_with_rope.transpose(-2, -1) / math.sqrt(dim_head)  # [B, H, S, S]
        # print(target_attn_scores.shape)
        
        # Softmax over all keys (text + target + context)
        attn_weights = F.softmax(target_attn_scores, dim=-1)  # [B, H, 1024, S]

        # Extract context weights
        context_weights = attn_weights[:, :, :, context_start:context_end]  # [B, H, 1024, 1024]
          
        # Compute proportion: sum over context keys / total (which is 1 after softmax)
        context_prop_per_query = context_weights.sum(dim=-1)  # [B, H, 1024]
        
        # Average over heads and queries
        layer_prop = context_prop_per_query.mean()  # scalar

        proportions.append(layer_prop)
    
    if len(proportions) == 0:
        return torch.tensor(0.0)
    
    return torch.stack(proportions).mean()



# def sample_timestep_with_bias(current_step, total_steps, num_train_timesteps, device, batch_size=1):
#     progress = current_step / total_steps  
#     min_timestep = int((1 - progress) * 0 + progress * 980) 
#     max_timestep = num_train_timesteps
    
#     timesteps = torch.randint(
#         min_timestep, 
#         max_timestep, 
#         (batch_size,), 
#         device=device
#     ).long()
    
#     return timesteps



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                       default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./example")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="./perturbed")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


def get_prompt_pool():
    """
    a prompt set containing 20 image editing commands
    """
    prompts = [
        
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


    # ===== 5. Scenes Changes (10) =====

        "Place this person on a tropical beach with palm trees and ocean waves.",
        "Put this person in a snowy mountain landscape with pine trees.",
        "Move this person to a bustling city street with skyscrapers.",
        "Put this person in a beach setting.",
        "Put this person in a snowy mountain.",
        "Put this person in a city street.",
        "Put this person in a forest.",
        "Put this person in a coffee shop.",
        "Put this person in a library.",
        "Put this person in a park."


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
        "Add earrings and make this person look confident.",

    ]


    return prompts



def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                       datefmt="%H:%M:%S", level=logging.INFO)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # get prompt pool
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts in the pool")
    
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
    dataset = AdversarialDataset(
        condition_images_dir=args.condition_images_dir,
        prompts=prompt_pool,
        resolution=args.resolution
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
    
    # release
    del text_encoder_clip, text_encoder_t5, tokenizer_clip, tokenizer_t5
    torch.cuda.empty_cache()
    
    attention_hook = AttentionHook()
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample['cond_path']
        
        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")
        
        perturbed_condition = condition_image.clone().detach()
        perturbed_condition.requires_grad_(True)
        original_condition = condition_image.clone().detach()
        
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")
        
        # Fixed values
        guidance = torch.tensor([3.5], device=accelerator.device)
        
        # g = torch.zeros_like(original_condition)
        # mu = 0.8
        for step in progress_bar:
            # randomly select a prompt at each step
            random_prompt_idx = torch.randint(0, len(prompt_pool), (1,)).item()
            current_prompt = prompt_pool[random_prompt_idx]
            prompt_embeds = prompt_embeds_pool[random_prompt_idx]
            pooled_prompt_embeds = pooled_prompt_embeds_pool[random_prompt_idx]
            
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=prompt_embeds.dtype)
            
            bsz = 1
            
            # Aample a high timestep
            timesteps = torch.randint(980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()

            # Alternative way
            # if step % 10 == 0:
            #     timesteps = torch.tensor([1000], device=accelerator.device).long()
            # else:
            #     timesteps = torch.randint(980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            
            # sample a noise as target
            dummy_target_shape = (bsz, 16, args.resolution // 8, args.resolution // 8)  
            noise = torch.randn(dummy_target_shape, device=accelerator.device, dtype=weight_dtype)
            
            t_normalized = (timesteps.float() / scheduler.config.num_train_timesteps).to(dtype=weight_dtype)
            t_normalized = t_normalized.view(-1, 1, 1, 1)
            
            # noisy latents
            noisy_latents = noise  
            
            perturbed_condition.requires_grad = True
            
            # Encode perturbed condition
            cond_latents_pert = vae.encode(perturbed_condition).latent_dist.mode()
            cond_latents_pert = (cond_latents_pert - vae_config_shift_factor) * vae_config_scaling_factor
            
            # Prepare latent IDs
            latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                bsz, cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype
            )
            
            cond_latent_ids_pert = FluxKontextPipeline._prepare_latent_image_ids(
                cond_latents_pert.shape[0], cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype
            )
            cond_latent_ids_pert[..., 0] = 1
            combined_latent_ids_pert = torch.cat([latent_image_ids, cond_latent_ids_pert], dim=0)
            
            # Pack latents
            packed_noisy_latents = FluxKontextPipeline._pack_latents(
                noisy_latents, bsz, noisy_latents.shape[1], noisy_latents.shape[2], noisy_latents.shape[3]
            )
            
            packed_cond_pert = FluxKontextPipeline._pack_latents(
                cond_latents_pert, bsz, cond_latents_pert.shape[1], cond_latents_pert.shape[2], cond_latents_pert.shape[3]
            )
            packed_input_pert = torch.cat([packed_noisy_latents, packed_cond_pert], dim=1)
            
            # Register hooks
            attention_hook.clear_hooks()
            attention_hook.register_hooks(transformer)
            
            timesteps_expanded = timesteps.expand(packed_noisy_latents.shape[0]).to(packed_noisy_latents.dtype)
            
            # Forward pass
            model_pred = transformer(
                hidden_states=packed_input_pert,
                timestep=timesteps_expanded / 1000,
                guidance=guidance.expand(bsz),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_latent_ids_pert,
                return_dict=False
            )[0]
            
            # Compute RoPE
            ids_pert = torch.cat([text_ids, latent_image_ids, cond_latent_ids_pert], dim=0)
            image_rotary_emb_pert = transformer.pos_embed(ids_pert)
            
            # Compute context proportion
            context_prop = compute_context_proportion(
                attention_hook.qkv_normalized,
                image_rotary_emb_pert
            )
            
            # Loss: minimize context proportion
            L_context = 1 - context_prop
            L_goal = L_context
            
            # Backward
            transformer.zero_grad()
            vae.zero_grad()
            L_goal.backward()

            
            # print(f"Grad max: {perturbed_condition.grad.max().item():.6f}")
            # print(f"Grad min: {perturbed_condition.grad.min().item():.6f}")

           # grad = perturbed_condition.grad
           # grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-8)
           # g = mu * g + grad_norm
            
            # PGD update
            adv_condition = perturbed_condition + args.alpha * perturbed_condition.grad.sign()
            eta = torch.clamp(adv_condition - original_condition, min=-args.eps, max=args.eps)
            perturbed_condition = torch.clamp(original_condition + eta, min=-1, max=1).detach_()
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()

            progress_bar.set_postfix({
                'prompt_idx': random_prompt_idx,
                'timestep': timesteps.item(),
                'ctx_prop': f'{context_prop.item():.4f}',
                # 'L_goal': f'{L_goal.item():.4f}',
                'pert': f'{perturbation_norm:.4f}'
            })
 
            del L_context, L_goal, model_pred, cond_latents_pert, packed_cond_pert
            del packed_input_pert, adv_condition, context_prop, noisy_latents, noise
            torch.cuda.empty_cache()
            
            # Save intermediate results every 200 steps
            if (step + 1) % 200 == 0:
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
        
        attention_hook.clear_hooks()
        del progress_bar, guidance, latent_image_ids
        torch.cuda.empty_cache()
        
        # # Save final results
        # output_path = os.path.join(args.output_dir, f"adversarial_{Path(cond_path).stem}.png")
        # perturbed_np = ((perturbed_condition.squeeze(0).cpu().float().numpy() + 1.0) * 127.5).astype(np.uint8)
        # Image.fromarray(np.transpose(perturbed_np, (1, 2, 0))).save(output_path)
        
        # logger.info(f"Saved: {output_path}")
    
    logger.info("Attack completed!")


if __name__ == "__main__":
    main()