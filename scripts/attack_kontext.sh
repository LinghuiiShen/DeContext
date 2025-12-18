#!/bin/bash

PRETRAINED_MODEL="black-forest-labs/FLUX.1-Kontext-dev"
CONDITION_IMAGES_DIR="./example"
OUTPUT_DIR="./perturbed"
RESOLUTION=512
ALPHA=0.005
EPS=0.1
ATTACK_STEPS=800
SEED=8
MIXED_PRECISION="bf16"

mkdir -p "$OUTPUT_DIR"

# 后台运行，输出重定向到日志文件
nohup python ./attack/attack_Flux_Kontext/attack_decontext.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --condition_images_dir "$CONDITION_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution "$RESOLUTION" \
    --alpha "$ALPHA" \
    --eps "$EPS" \
    --attack_steps "$ATTACK_STEPS" \
    --seed "$SEED" \
    --mixed_precision "$MIXED_PRECISION" \
    > "$OUTPUT_DIR/attack.log" 2>&1 &

echo "Attack started in background. PID: $!"
echo "Log file: $OUTPUT_DIR/attack.log"
echo "To check progress: tail -f $OUTPUT_DIR/attack.log"
echo "To stop: kill $!"