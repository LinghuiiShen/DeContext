#!/bin/bash

# Step1X-Edit Adversarial Attack Script
# Simple runner script

MODEL_PATH="./attack/attack_Step1X_Edit/models"
CONDITION_IMAGES_DIR="./example"
OUTPUT_DIR="./perturbed"
RESOLUTION=512
ALPHA=0.005
EPS=0.1
ATTACK_STEPS=800
SEED=8
MIXED_PRECISION="bf16"
VERSION="v1.1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run attack
python ./attack/attack_Step1X_Edit/attack.py \
    --model_path "$MODEL_PATH" \
    --condition_images_dir "$CONDITION_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution "$RESOLUTION" \
    --alpha "$ALPHA" \
    --eps "$EPS" \
    --attack_steps "$ATTACK_STEPS" \
    --seed "$SEED" \
    --mixed_precision "$MIXED_PRECISION" \
    --version "$VERSION"