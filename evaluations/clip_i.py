import argparse
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)

def get_image_embeddings(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, type=str)
    parser.add_argument("--edit_dir", required=True, type=str)
    parser.add_argument("--persons", required=True, nargs="+")
    parser.add_argument("--prompts", default=[3, 4, 5, 6], nargs="+", type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--real_ids", required=True, nargs="+")
    args = parser.parse_args()

    result = []
    
    for prompt_id in args.prompts:
        print(f"Processing prompt {prompt_id}")
        clip_scores = []
        skipped_count = 0 
        
        for person_photo, real_id in zip(args.persons, args.real_ids):

            src_path = os.path.join(args.src_dir, f"image_{real_id}.jpg")
            if not os.path.exists(src_path):
                print(f"Warning: {src_path} not found, skipping")
                continue
            

            prompt_dir = os.path.join(args.edit_dir, person_photo, f"prompt{prompt_id}")
            if not os.path.exists(prompt_dir):
                print(f"Warning: {prompt_dir} not found, skipping")
                continue

            try:
                src_image = Image.open(src_path).convert("RGB")
                src_embeddings = get_image_embeddings(src_image, clip_processor, clip_model, device)
            except (OSError, IOError) as e:
                print(f"Warning: Corrupted source image {src_path}, skipping this person")
                skipped_count += 1
                continue
            

            edit_files = sorted(os.listdir(prompt_dir))[:args.top_k]
            
            for edit_file in edit_files:
                edit_path = os.path.join(prompt_dir, edit_file)

                try:
                    edit_image = Image.open(edit_path).convert("RGB")
                    edit_embeddings = get_image_embeddings(edit_image, clip_processor, clip_model, device)
                    
                    similarity = F.cosine_similarity(edit_embeddings, src_embeddings, dim=-1).mean().item()
                    clip_scores.append(similarity)
                    
                except (OSError, IOError, Exception) as e:
                    print(f"Warning: Corrupted or truncated image {edit_path}, skipping")
                    skipped_count += 1
                    continue

        mean_score = np.mean(clip_scores) if clip_scores else 0.0
        result.append({
            "prompt": prompt_id,
            "mean_clip_i": mean_score,
            "num_samples": len(clip_scores),
            "skipped": skipped_count
        })
        print(f"Prompt {prompt_id}: CLIP-I = {mean_score:.4f} (n={len(clip_scores)}, skipped={skipped_count})")

    df = pd.DataFrame(result)
    print("\n=== Final Results ===")
    print(df)
    df.to_csv("clip_i_results.csv", index=False)
    print("\nResults saved to clip_i_results.csv")