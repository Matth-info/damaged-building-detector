#!/bin/bash
# scripts/run_cloud_detector_training.sh

# Exit script on any error
set -e

# Default parameter values
BATCH_SIZE=8
EXPERIMENT_NAME="Cloud_Data_autoencoder"
ORIGIN_DIR="../data/data_samples/Cloud_DrivenData/final/public"
LEARNING_RATE=0.001
MODEL_DIR="../models"
NUM_EPOCHS=30
MIXED_PRECISION=true  # Toggle mixed precision

# Log the experiment settings
echo "=========================================="
echo "Running Cloud Detector Training"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Model Directory: $MODEL_DIR"
echo "Dataset Directory: $ORIGIN_DIR"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Mixed Precision Enabled: $MIXED_PRECISION"
echo "=========================================="

# Run the training script with the provided parameters
python3 AutoEncoder_Cloud_training.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --batch_size "$BATCH_SIZE" \
    --origin_dir "$ORIGIN_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --model_dir "$MODEL_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    $( [ "$MIXED_PRECISION" = true ] && echo "--mixed_precision" )


$env:PYTHONPATH = "src"
python scripts/AutoEncoder_Cloud_training.py \ 
--experiment_name "Cloud_Data_autoencoder" \
 --batch_size 1 \
 --origin_dir "data\data_samples\Cloud_DrivenData\final\public" \
 --learning_rate 0.001 \
 --model_dir "models" \
 --num_epochs 1   