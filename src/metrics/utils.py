import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
from training.augmentations import augmentation_test_time, augmentation_test_time_siamese
import albumentations as A
import pandas as pd 
from tqdm import tqdm 
import mlflow
import logging

def compute_model_class_performance(
        model,
        dataloader,
        num_classes,
        device='cuda',
        class_names=None, 
        siamese=False,
        image_key="image",
        mask_key="mask",
        average_mode="macro",
        tta=False,
        mlflow_bool=False
):
    """
    Computes and stores class-wise performance metrics for a segmentation model.

    Parameters:
        model (torch.nn.Module): The segmentation model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        num_classes (int): The number of classes in the segmentation task.
        device (str): Device to perform the evaluation on ('cuda' or 'cpu').
        class_names (list): Optional list of class names corresponding to label indices.
        siamese (bool): Whether the model uses a siamese network structure.
        image_key (str): Key for accessing images in the dataloader batch.
        mask_key (str): Key for accessing masks in the dataloader batch.
        average_mode (str): Averaging mode for overall metrics ('macro', 'micro', etc.).
        tta (bool): Run Test Time Augmentation 
        output_file (str): Path to the file where metrics will be stored.

    Returns:
        None
    """
    model.eval()
    all_preds = []
    all_targets = []
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        with tqdm(dataloader, desc=f"Testing", unit="batch") as t:
            for batch in t:
                if siamese:
                    pre_image = batch["pre_image"].to(device)
                    post_image = batch["post_image"].to(device)
                    targets = batch[mask_key].to(device)

                    if tta:
                        outputs = augmentation_test_time_siamese(
                            model=model, 
                            images_1=pre_image, 
                            images_2=post_image, 
                            list_augmentations=[
                                                A.HorizontalFlip(p=1.0),  # Horizontal flip
                                                A.VerticalFlip(p=1.0)    # Vertical flip
                                            ],
                            aggregation="mean", 
                            device=device
                            )
                    else:
                        outputs = model(pre_image, post_image)
                else:
                    images = batch[image_key].to(device)
                    targets = batch[mask_key].to(device)
                    if tta:
                        outputs = augmentation_test_time(
                            model=model, 
                            images=images, 
                            list_augmentations=[
                                                A.HorizontalFlip(p=1.0),  # Horizontal flip
                                                A.VerticalFlip(p=1.0)    # Vertical flip
                                            ],
                            aggregation="mean", 
                            device=device
                        )
                    else: 
                        outputs = model(images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = targets.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

                correct_pixels += np.sum(preds == targets)
                total_pixels += np.prod(targets.shape)

    all_preds = np.concatenate([p.ravel() for p in all_preds])
    all_targets = np.concatenate([t.ravel() for t in all_targets])

    unique_classes = np.arange(num_classes)
    mlcm = multilabel_confusion_matrix(all_targets, all_preds, labels=unique_classes)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Collect metrics into a DataFrame
    metrics_data = []

    for i in unique_classes:
        tn, fp, fn, tp = mlcm[i].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics_data.append({
            "Class": class_names[i],
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "IoU": iou,
            "Dice": dice
        })
    
    metrics_df = pd.DataFrame(metrics_data)

    # Compute overall metrics
    precision_overall = precision_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)
    recall_overall = recall_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)
    f1_overall = f1_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)

    overall_metrics_data = {
            "Overall Precision": [precision_overall],
            "Overall Recall": [recall_overall],
            "Overall F1 Score": [f1_overall]
        }
    overall_metrics_df = pd.DataFrame(overall_metrics_data)
    
    if mlflow_bool:
        mlflow.log_table(metrics_df, artifact_file="class_wise_test_performance.json")
        mlflow.log_table(overall_metrics_df, artifact_file="overall_test_performance.json")
        logging.info("Per-Class Performance and Overall Performance file logged as artifact")
    else:
        print("Class-wise Performance Metrics:")
        print(metrics_df)
        print("-" * 40)
        print("Overall Performance Metrics:")
        for metric, value in overall_metrics_data.items():
            print(f"{metric}: {value[0]:.4f}")
