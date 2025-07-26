#!/bin/bash

# This is a simplified script for fine-tuning the VILA 8B model on a single GPU.
# It is adapted from the original m3-vila8b.sh script.

# --- Configuration ---
# TODO: Fill in the following paths before running the script.

# 1. Path to the VILA codebase
# This should point to the directory where you have cloned the VILA repository.
# Based on your project structure, this is likely 'thirdparty/VILA'.
VILA_CODE_PATH="thirdparty/VILA"

# 2. Path to your training data
# This should be the absolute path to your training JSON file.
TRAIN_JSON_PATH="/path/to/your/train.json"

# 3. Path to the base model on Hugging Face or local directory
# We are using the pretrained model from Hugging Face as you specified.
MODEL_NAME_OR_PATH="MONAI/Llama3-VILA-M3-8B"

# 4. Path to the output directory for checkpoints
# This is where the trained model checkpoints will be saved.
OUTPUT_DIR="/path/to/output/checkpoints"

# 5. (Optional) Batch size
# Adjust this based on your GPU's VRAM. Start with 1 and increase if possible.
# The effective batch size will be (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS).
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2

# --- End of Configuration ---

# --- Script Execution ---

# Activate Conda environment
# Make sure you have a conda environment named 'vila' as per the original scripts.
# You may need to adjust this if your environment has a different name.
# source /path/to/your/miniconda3/bin/activate
# conda activate vila

# Set PYTHONPATH to include the VILA code
export PYTHONPATH=$VILA_CODE_PATH
echo "PYTHONPATH is set to $PYTHONPATH"

# Run the training script
# We are using the python command directly for a single GPU setup.
python $VILA_CODE_PATH/llava/train/train_mem.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $TRAIN_JSON_PATH \
    --vision_tower google/siglip-so400m-patch14-384 \
    --version llama_3 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True

echo "Training script finished."
