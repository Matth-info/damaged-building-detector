import logging

import albumentations as A
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def compute_model_class_performance(
    model,
    dataloader,
    num_classes,
    device="cuda",
    class_names=None,
    siamese=False,
    image_key="image",
    mask_key="mask",
    average_mode="macro",
    saving_method=None,
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
        saving_method (str): Define the method for saving results ('mlflow', 'pandas', 'display').

    Returns:
        dict: A dictionary containing class-wise and overall metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="Testing", unit="batch") as t:
            for batch in t:
                if siamese:
                    pre_image = batch["pre_image"].to(device)
                    post_image = batch["post_image"].to(device)
                    targets = batch[mask_key].to(device)
                    outputs = model(pre_image, post_image)
                else:
                    images = batch[image_key].to(device)
                    targets = batch[mask_key].to(device)
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

        metrics_data.append(
            {
                "Class": class_names[i],
                "Precision": np.round(precision, 4),
                "Recall": np.round(recall, 4),
                "F1 Score": np.round(f1, 4),
                "IoU": np.round(iou, 4),
                "Dice": np.round(dice, 4),
            }
        )

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index("Class")  # Ensure "Class" is set as the index

    # remove background / zero change from the Overall Metrics
    labels = [i for i in unique_classes if i != 0]
    # Compute overall metrics
    precision_overall = precision_score(
        all_targets,
        all_preds,
        labels=labels,
        average=average_mode,
        zero_division=0,
    )
    recall_overall = recall_score(
        all_targets,
        all_preds,
        labels=labels,
        average=average_mode,
        zero_division=0,
    )
    f1_overall = f1_score(
        all_targets,
        all_preds,
        labels=labels,
        average=average_mode,
        zero_division=0,
    )

    overall_metrics_data = {
        f"Precision ({average_mode})": [np.round(precision_overall, 4)],
        f"Recall ({average_mode})": [np.round(recall_overall, 4)],
        f"F1 Score ({average_mode})": [np.round(f1_overall, 4)],
    }
    overall_metrics_df = pd.DataFrame(overall_metrics_data)  # No index is set here

    # Handle saving methods
    if saving_method == "mlflow":
        mlflow.log_table(
            metrics_df.reset_index(), artifact_file="class_wise_test_performance.json"
        )
        mlflow.log_table(overall_metrics_df, artifact_file="overall_test_performance.json")
        logging.info("Per-Class Performance and Overall Performance file logged as artifact")

    elif saving_method == "display":
        print("Class-wise Performance Metrics:\n")
        print(metrics_df)
        print("-" * 40)
        print("Overall Performance Metrics:\n")
        for metric, value in overall_metrics_data.items():
            print(f"{metric}: {value[0]:.3f}")

    elif saving_method == "pandas":
        metrics_df.to_csv(
            "class_wise_test_performance.csv", index=True
        )  # Class-wise metrics with index
        overall_metrics_df.to_csv(
            "overall_test_performance.csv", index=False
        )  # Overall metrics without index
        print("Metrics saved as CSV files.")

    elif not saving_method:
        print("Metrics will not be saved.")

    else:
        print("Unsupported saving method. Metrics will not be saved.")

    # Return metrics as a dictionary
    return {
        "class_wise_metrics": metrics_df,
        "overall_metrics": overall_metrics_df,
    }
