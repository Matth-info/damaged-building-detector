#!/bin/bash
# scripts/run_training.sh

# Ensure the script is executed from the correct directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || exit

# Function to run training for a specified backbone
run_training() {
    local BACKBONE=$1
    local BATCH_SIZE=8
    local EXPERIMENT_NAME="xDB_${BACKBONE}_Unet"
    local ORIGIN_DIR="../data/xDB/tier3"

    echo "Running training for backbone: $BACKBONE"
    echo "Experiment Name: $EXPERIMENT_NAME"

    python3 ResNet_Unet_xDB_training.py \
        --experiment_name $EXPERIMENT_NAME \
        --backbone $BACKBONE \
        --batch_size $BATCH_SIZE \
        --origin_dir $ORIGIN_DIR \
        --num_epochs 30 \
        --mixed_precision \
        --pretrained
        
}

# Run training for ResNet34
#run_training "resnet34"

# Run training for ResNet50
run_training "resnet50"