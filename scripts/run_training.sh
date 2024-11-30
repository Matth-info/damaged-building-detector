#!/bin/bash
# scripts/run_training.sh

# Run the training with specified configuration
python ResNet_Unet_xDB.py --experiment_name "xDB_ResNet18_Unet" --backbone "resnet18" --batch_size 16 --num_epochs 30 --mixed_precision
