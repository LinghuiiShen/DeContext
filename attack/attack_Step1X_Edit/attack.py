#!/usr/bin/env python
# coding=utf-8
"""
Step1X-Edit Adversarial Attack - Context Suppression
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import argparse
import logging
import sys
from pathlib import Path
import math
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
from modules.layers import EmbedND

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_models
from einops import rearrange, repeat

logger = get_logger(__name__)


class AdversarialDataset(Dataset):
    def __init__(self, condition_images_dir, prompts, resolution=512):
        self.condition_images_dir = Path(condition_images_dir)
        self.prompts = prompts
        self.resolution = resolution
        
        self.cond_files = sorted([f for f in self.condition_images_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        self.image_transforms = transforms.Compose([
            transforms.Lambda(lambda img: self._resize_to_multiple_of_16(img)),
            transforms.ToTensor(),  
        ])
    
    def _resize_to_multiple_of_16(self, img):
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(self.resolution * self.resolution * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(self.resolution * self.resolution / r))
            w_new = math.ceil(h_new * r)

        h_new = h_new // 16 * 16
        w_new = w_new // 16 * 16

        return img.resize((w_new, h_new), Image.LANCZOS)
    
    def __len__(self):
        return len(self.cond_files)
    
    def __getitem__(self, idx):
        cond_path = self.cond_files[idx]
        cond_image = Image.open(cond_path).convert('RGB')
        
        cond_image_tensor = self.image_transforms(cond_image)  # [C, H, W], 范围 [0,1]
        
        return {
            'condition_image': cond_image_tensor,
            'cond_path': str(cond_path),
        }
        
def create_pe_embedder(hidden_size, num_heads, device):
    pe_dim = hidden_size // num_heads
    axes_dim = [16, 56, 56]
    return EmbedND(dim=pe_dim, theta=10000, axes_dim=axes_dim).to(device)


def rope(pos, dim: int, theta: int):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def apply_rope(xq, xk, freqs_cis):
    xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    xk = xk.transpose(1, 2)

    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    xq_out = xq_out.reshape(*xq.shape).type_as(xq).transpose(1, 2)
    xk_out = xk_out.reshape(*xk.shape).type_as(xk).transpose(1, 2)

    return xq_out, xk_out

class AttentionHook:
    def __init__(self):
        self.qk_dict = {}
        self.hooks = []
    
    def register_hooks(self, dit, selected_layers):
        """
        Register hooks on SingleStreamBlock.norm (QKNorm)
        Captures Q/K after RMSNorm, before RoPE
        """
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                q_normed, k_normed = output
                self.qk_dict[layer_name] = {
                    'q': q_normed.clone(),  # [B, L, H, D]
                    'k': k_normed.clone()   # [B, L, H, D]
                }
            return hook_fn
        
        for layer_idx in selected_layers.get('single_blocks', []):
            if layer_idx < len(dit.single_blocks):
               
                block = dit.single_blocks[layer_idx]
                layer_name = f'single_{layer_idx}'
                
                # Hook on QKNorm's forward output
                hook = block.norm.register_forward_hook(create_hook(layer_name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.qk_dict = {}

def compute_context_proportion(qk_dict, img_ids, txt_ids, dit, version='v1.1'):
    proportions = []
    
    for layer_name, qk_data in qk_dict.items():
        if 'q' not in qk_data or 'k' not in qk_data:
            continue
        
        q = qk_data['q']  # [B, L, H, D]
        k = qk_data['k']  # [B, L, H, D]
        batch_size, seq_len, num_heads, head_dim = q.shape

        if seq_len != 2688:
            continue

        target_start, target_end = 640, 1664
        context_start, context_end = 1664, 2688

        all_positions = torch.cat([txt_ids, img_ids], dim=1) 
        pe = dit.pe_embedder(all_positions) # [B, L, H, D//2, 2, 2] 
        q_rope, k_rope = apply_rope(q, k, pe)

        # Extract only target queries
        q_target_rope = q_rope[:, target_start:target_end, :, :]  # [B, 1024, H, D]
        q_target_rope = q_target_rope.transpose(1, 2)  # [B, H, 1024, D]
        k_rope=k_rope.transpose(1,2)
        
        scale_factor = 1 / math.sqrt(q.size(-1))
        target_attn_scores = (q_target_rope @ k_rope.transpose(-2, -1)) * scale_factor

        attn_weights = F.softmax(target_attn_scores, dim=-1)

        condition_weights = attn_weights[:, :, :, context_start:context_end]
        context_prop_per_query = condition_weights.sum(dim=-1)
        
        layer_prop = context_prop_per_query.mean()
        proportions.append(layer_prop)
    
    if len(proportions) == 0:
        return torch.tensor(0.0, device=q.device, dtype=q.dtype)
    
    return torch.stack(proportions).mean()


def get_prompt_pool():
    return [
        # Expressions (10)
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
        
        # Accessories (10)
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
        
        # Posture (10)
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
        
        # Style (10)
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
        
        # Scenes (10)
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
        
        # Combined (10)
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--condition_images_dir", type=str, default="./example")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--attack_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="./stepx_attack_results2")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--version", type=str, default='v1.1', choices=['v1.0', 'v1.1'])
    return parser.parse_args()


def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                       datefmt="%H:%M:%S", level=logging.INFO)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompt pool
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts")
    
    # Set precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Load models
    logger.info("Loading Step1X models...")
    
    ae, dit, llm_encoder = load_models(
        dit_path=os.path.join(args.model_path, 'step1x-edit-v1p1-official.safetensors' if args.version == 'v1.1' else 'step1x-edit-i1258.safetensors'),
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        qwen2vl_model_path=os.path.join(args.model_path, 'Qwen2.5-VL-7B-Instruct'),
        mode="torch",
        device=accelerator.device,
        max_length=640,
        dtype=weight_dtype,
        version=args.version,
    )

    ae = ae.to(accelerator.device, dtype=torch.float32)
    dit = dit.to(accelerator.device, dtype=weight_dtype)
    
    dit.eval()
    ae.eval()
    dit.requires_grad_(False)
    ae.requires_grad_(False)
    
    # Enable gradient checkpointing
    dit.enable_gradient_checkpointing()
    # ae.enable_gradient_checkpointing()
    
    # Load dataset
    dataset = AdversarialDataset(
        args.condition_images_dir, prompt_pool, args.resolution
    )
    logger.info(f"Loaded {len(dataset)} condition images")

    # Initialize hook
    attention_hook = AttentionHook()
    
    # Attack each image
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample['cond_path']
        
        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")
        
        perturbed_condition = condition_image.clone().detach()
        perturbed_condition.requires_grad_(True)
        original_condition = condition_image.clone().detach()
        
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")
        
        for step in progress_bar:

            prompt_idx = torch.randint(0, len(prompt_pool), (1,)).item()
            current_prompt = prompt_pool[prompt_idx]

            # Re-encode with the updated adversarial sample because llm encoder need both text prompts and context image
            perturbed_float32_for_llm = perturbed_condition.detach().to(torch.float32)
            with torch.no_grad():
                txt, mask = llm_encoder([current_prompt], perturbed_float32_for_llm)

            # Sample timestep
            if step % 10 == 0:
                timestep_value = 1000 # enhance the effect
            else:
                timestep_value = torch.randint(980, 1000, (1,)).item()
            
            # Generate noise
            noise_shape = (1, 16, args.resolution // 8, args.resolution // 8)
            noise = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype)

            perturbed_condition_float32 = perturbed_condition.to(torch.float32)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                cond_latent = ae.encode(perturbed_condition_float32 * 2 - 1)

            # Pack latents
            bs, _, h, w = noise.shape
            img = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            ref_img = rearrange(cond_latent, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)

            # Position IDs
            img_ids = torch.zeros(h // 2, w // 2, 3, device=accelerator.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=accelerator.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=accelerator.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
            
            if args.version == 'v1.0':
                ref_img_ids = torch.zeros(h // 2, w // 2, 3, device=accelerator.device)
            else:
                ref_img_ids = torch.ones(h // 2, w // 2, 3, device=accelerator.device)
            ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(h // 2, device=accelerator.device)[:, None]
            ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(w // 2, device=accelerator.device)[None, :]
            ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)
            
            txt_ids = torch.zeros(bs, txt.shape[1], 3, device=accelerator.device)

            # Concatenate
            img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
            img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)
            
            # Register hooks
            attention_hook.clear_hooks()
            attention_hook.register_hooks(dit, selected_layers={'single_blocks': list(range(26))})

            # Forward pass
            t_vec = torch.full((bs,), timestep_value / 1000.0, dtype=weight_dtype, device=accelerator.device)
            pe_embedder = create_pe_embedder(3072, 24, accelerator.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_pred = dit(
                    img=img.to(accelerator.device),
                    img_ids=img_ids.to(accelerator.device),
                    txt_ids=txt_ids.to(accelerator.device),
                    timesteps=t_vec,
                    llm_embedding=txt.to(accelerator.device),
                    t_vec=t_vec,
                    mask=mask.to(accelerator.device),
                )
            
            current_img_ids = img_ids.clone()
            current_txt_ids = txt_ids.clone()

            context_prop = compute_context_proportion(
                attention_hook.qk_dict, 
                current_img_ids,
                current_txt_ids,
                dit,
                version=args.version
            )
            
            # Loss: minimize context proportion
            L_context = 1 - context_prop
            
            # Backward
            dit.zero_grad()
            ae.zero_grad()
            
            if isinstance(L_context, torch.Tensor) and L_context.requires_grad:
                L_context.backward()

            if step == 0:
                logger.info("=== Initial State Check ===")
                logger.info(f"original_condition range: [{original_condition.min():.4f}, {original_condition.max():.4f}]")
                logger.info(f"perturbed_condition range: [{perturbed_condition.min():.4f}, {perturbed_condition.max():.4f}]")
                logger.info(f"Expected range: [0.0, 1.0]")
                logger.info(f"eps: {args.eps}, alpha: {args.alpha}")

            # PGD update
            if perturbed_condition.grad is not None:
                grad_sign = perturbed_condition.grad.sign()
                adv_condition = perturbed_condition + args.alpha * grad_sign
                eta = torch.clamp(adv_condition - original_condition, min=-args.eps, max=args.eps)
                perturbed_condition = torch.clamp(original_condition + eta, min=0.0, max=1.0).detach_()
                perturbed_condition.requires_grad_(True)
            
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()
            context_prop_value = context_prop.item() if isinstance(context_prop, torch.Tensor) else 0.0

            progress_bar.set_postfix({
                'prompt_id': prompt_idx,
                'timestep': timestep_value,
                'ctx_prop': f'{context_prop_value:.4f}',
                'pert': f'{perturbation_norm:.4f}'
            })
            
            # Cleanup
            del model_pred, context_prop, L_context
            del img, ref_img, img_ids, ref_img_ids, txt_ids
            del cond_latent, perturbed_condition_float32, noise
            if perturbed_condition.grad is not None:
                del grad_sign, adv_condition, eta
            torch.cuda.empty_cache()
            
            # Save intermediate
            if (step + 1) % 200 == 0:
                with torch.no_grad():
                    perturbed_np = perturbed_condition.detach().squeeze(0).cpu().float().numpy()
                    perturbed_np = (perturbed_np * 255).astype(np.uint8)  
                    perturbed_np = np.transpose(perturbed_np, (1, 2, 0))
                    perturbed_pil = Image.fromarray(perturbed_np)
                    
                    intermediate_path = os.path.join(
                        args.output_dir, 
                        f"adversarial_{Path(cond_path).stem}_step_{step+1}.png"
                    )
                    perturbed_pil.save(intermediate_path)
                    perturbation = perturbed_condition - original_condition
                    delta_vis = perturbation.detach().squeeze(0).cpu().float().numpy()
                    
                    delta_vis = (delta_vis + args.eps) / (2 * args.eps) 
                    delta_vis = np.clip(delta_vis, 0, 1)  
                    delta_vis = (delta_vis * 255).astype(np.uint8)
                    delta_vis = np.transpose(delta_vis, (1, 2, 0))
                    delta_pil = Image.fromarray(delta_vis)
                    
                    delta_intermediate_path = os.path.join(
                        args.output_dir, 
                        f"perturbation_{Path(cond_path).stem}_step_{step+1}.png"
                    )
                    delta_pil.save(delta_intermediate_path)
                    
                    logger.info(f"Saved intermediate result at step {step+1}: {intermediate_path}")
                    logger.info(f"Perturbation range: [{perturbation.min().item():.4f}, {perturbation.max().item():.4f}]")
                    
        attention_hook.clear_hooks()
    
    logger.info("Attack completed!")


if __name__ == "__main__":
    main()