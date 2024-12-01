#!/bin/bash

# Set default values for the arguments
BACKBONE="resnet18"
MODEL_PATH="../models/xDB_ResNet18_Unet_20241130-201241_best_model.pth"
ORIGIN_DIR="../data/xDB/tier3"
BATCH_SIZE=16
TTA=false
MIXED_PRECISION=false

# Execute the Python script
python3 ResNet_Unet_xDB_testing.py \
    --backbone $BACKBONE \
    --model_path $MODEL_PATH \
    --origin_dir $ORIGIN_DIR \
    --batch_size $BATCH_SIZE \
    --tta \
    --mixed_precision \
