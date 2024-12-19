import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
from prettytable import PrettyTable

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
        output_file="class_performance_metrics.txt"
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
        for batch in dataloader:
            if siamese:
                pre_image = batch["pre_image"].to(device)
                post_image = batch["post_image"].to(device)
                targets = batch["post_mask"].to(device)
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

    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "F1 Score", "IoU", "Dice"]

    for i in unique_classes:
        tn, fp, fn, tp = mlcm[i].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        table.add_row([class_names[i], f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{iou:.4f}", f"{dice:.4f}"])

    precision_overall = precision_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)
    recall_overall = recall_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)
    f1_overall = f1_score(all_targets, all_preds, labels=unique_classes, average=average_mode, zero_division=0)

    with open(output_file, "w") as f:
        f.write("Per-Class Performance Metrics:\n")
        f.write(str(table) + "\n")
        f.write("-" * 40 + "\n")
        f.write("Overall Performance Metrics:\n")
        f.write(f"  Precision ({average_mode}): {precision_overall:.4f}\n")
        f.write(f"  Recall ({average_mode}):    {recall_overall:.4f}\n")
        f.write(f"  F1 Score ({average_mode}):  {f1_overall:.4f}\n")

    print("Per-Class Performance Metrics:")
    print(table)
    print("-" * 40)
    print("Overall Performance Metrics:")
    print(f"  Precision ({average_mode}): {precision_overall:.4f}")
    print(f"  Recall ({average_mode}):    {recall_overall:.4f}")
    print(f"  F1 Score ({average_mode}):  {f1_overall:.4f}")
    print(f"Metrics have been saved to {output_file}")
