from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.training.utils import display_confusion_matrix

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from src.utils import BaseLogger

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)


def model_evaluation(
    model: torch.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    class_names: str | None = None,
    *,
    is_siamese: bool = False,
    image_key: str = "image",
    mask_key: str = "mask",
    average_mode: str = "macro",
    saving_method: Literal["logger", "pandas", "display", None] = None,
    logger: BaseLogger | None,
    decimals: int = 4,
) -> dict:
    """Compute and stores class-wise performance metrics for a segmentation model.

    Parameters
    ----------
        model (torch.nn.Module): The segmentation model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        num_classes (int): The number of classes in the segmentation task.
        device (str): Device to perform the evaluation on ('cuda' or 'cpu').
        class_names (list): Optional list of class names corresponding to label indices.
        is_siamese (bool): Whether the model uses a is_siamese network structure.
        image_key (str): Key for accessing images in the dataloader batch.
        mask_key (str): Key for accessing masks in the dataloader batch.
        average_mode (str): Averaging mode for overall metrics ('macro', 'micro', etc.).
        saving_method (str): Define the method for saving results ('mlflow', 'pandas', 'display').
        decimals (int): Number of decimals

    Returns:
    -------
        dict: A dictionary containing class-wise and overall metrics.

    """
    if not logger and saving_method == "logger":
        raise AttributeError("Require to pass a BaseLogger as argument.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad(), tqdm(dataloader, desc="Testing", unit="batch") as t:
        for batch in t:
            if is_siamese:
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

    all_preds = np.concatenate([p.ravel() for p in all_preds])
    all_targets = np.concatenate([t.ravel() for t in all_targets])

    unique_classes = np.arange(num_classes)
    conf_matrix = confusion_matrix(y_true=all_targets, y_pred=all_preds)
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
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        metrics_data.append(
            {
                "Class": class_names[i],
                "Precision": np.round(precision, 4),
                "Recall": np.round(recall, 4),
                "F1 Score": np.round(f1, 4),
                "IoU": np.round(iou, 4),
                "Dice": np.round(dice, 4),
                "Acc": np.round(acc, 4),
            },
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
        f"Precision ({average_mode})": [np.round(precision_overall, decimals)],
        f"Recall ({average_mode})": [np.round(recall_overall, decimals)],
        f"F1 Score ({average_mode})": [np.round(f1_overall, decimals)],
    }
    overall_metrics_df = pd.DataFrame(overall_metrics_data)  # No index is set here

    # save confusion matrix
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        save_path = pathlib.Path(tmpfile.name)
        fig, ax = plt.subplots()
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=f".{decimals}f",
            cmap="Blues",
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_title("Normalized Confusion Matrix")
        ax.set_label("Predicted Label")
        ax.set_ylabel("True Label")
        plt.close(fig)
        plt.tight_layout()
        plt.savefig(save_path)

        # Handle saving methods
        match saving_method:
            case "logger":
                logger.log_table(
                    metrics_df.reset_index(),
                    artifact_file="class_wise_test_performance.json",
                )
                logger.log_artifact(str(save_path))
                logger.log_table(overall_metrics_df, artifact_file="overall_test_performance.json")
                logging.info(
                    "Per-Class Performance and Overall Performance file logged as artifact"
                )

            case "display":
                _logger.info("Class-wise Performance Metrics:\n")
                _logger.info(metrics_df)
                _logger.info("-" * 40)
                _logger.info("Overall Performance Metrics:\n")
                for metric, value in overall_metrics_data.items():
                    _logger.info("%s : %f", metric, value[0])
                _logger.info("-" * 40)
                _logger.info("Normalized Confusion Matrix:\n")
                display_confusion_matrix(conf_matrix, class_names, decimals)

            case "pandas":
                metrics_df.to_csv(
                    "class_wise_test_performance.csv",
                    index=True,
                )  # Class-wise metrics with index
                overall_metrics_df.to_csv(
                    "overall_test_performance.csv",
                    index=False,
                )  # Overall metrics without index

                pd.DataFrame(
                    conf_matrix,
                    columns=[f"Pred {name}" for name in class_names],
                    index=[f"True {name}" for name in class_names],
                ).to_csv("normalized_confusion_matrix.csv", float_format=f"%.{decimals}f")

                _logger.info("Metrics saved as CSV files.")
            case _:
                _logger.info("Unsupported saving method. Metrics will not be saved.")

        # Return metrics as a dictionary
        return {
            "class_wise_metrics": metrics_df,
            "overall_metrics": overall_metrics_df,
        }
