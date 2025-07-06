# Post-Training Pruning
# Unstructured Pruning
from __future__ import annotations

import logging

import pandas as pd
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset

import src.datasets as ds
from src import models
from src.augmentation import Augmentation_pipeline
from src.testing import model_evaluation

# simple post training pruning technique (global)
# test the performance on a validation set (see the degradation on several metrics)
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)


def apply_identity(parameters_to_prune: list) -> None:
    """Apply identity pruning to the given parameters, effectively leaving them unpruned.

    Args:
        parameters_to_prune (list): List of (module, parameter_name) tuples to apply identity pruning.

    """
    for module, name in parameters_to_prune:
        prune.identity(module, name)


def global_unstructured_pruining(model: nn.Module, amount: float) -> None:
    """Perform global unstructured pruning on the model's Conv2d layers.

    Args:
    ----
        model (nn.Module): The PyTorch model to prune.
        amount (float): The proportion of weights to prune globally.

    """
    # Collect parameters to prune with names
    parameters_to_prune_with_names = [
        (name, module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    parameters_to_prune = [
        (module, param_name) for (_, module, param_name) in parameters_to_prune_with_names
    ]

    # Apply global unstructured pruning
    if amount > 0:
        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        apply_identity(parameters_to_prune)

    if torch.nn.utils.prune.is_pruned(model):
        overall_nb_sparse_weights, overall_nb_weights = 0, 0

        # Display sparsity for each pruned module
        for name, module, param_name in parameters_to_prune_with_names:
            nb_sparse_weights = float(torch.sum(getattr(module, param_name) == 0))
            nb_weights = float(getattr(module, param_name).nelement())
            overall_nb_sparse_weights += nb_sparse_weights
            overall_nb_weights += nb_weights
            sparsity = 100.0 * nb_sparse_weights / nb_weights
            _logger.debug("Sparsity in %s.%s: %.2f%%", name, param_name, sparsity)

        _logger.debug(
            "Overall Sparsity : %.2f%%",
            100 * overall_nb_sparse_weights / overall_nb_weights,
        ) if overall_nb_weights > 0 else logging.debug("Overall Sparsity : 0%")
    else:
        _logger.debug("Error Pytorch does not recognize Pruning format !")


def apply_pruning_to_model(pruned_model: nn.Module) -> None:
    """Make pruning permanent by removing weight_orig and weight_mask, and remove the forward_pre_hook.

    Args:
    ----
        pruned_model (nn.Module): The PyTorch model that have been pruned.

    """
    if torch.nn.utils.prune.is_pruned(pruned_model):
        parameters_to_prune_with_names = [
            (name, module, "weight")
            for name, module in pruned_model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        parameters_to_prune = [
            (module, param_name) for (_, module, param_name) in parameters_to_prune_with_names
        ]

        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
    else:
        logging.error("This model is not in Pytorch pruned format")


def run_pruning_experiment(
    model_class: nn.Module,
    weights_path: str,
    dataset: Dataset,
    sparsity_levels: list[float],
    device: str,
    kwargs: dict,
) -> pd.DataFrame:
    """Run an experiment to test the model at different levels of sparsity.

    Args:
    ----
        model_class (nn.Module): The model class to initialize.
        weights_path (str): Path to the model's weights file.
        dataset (Dataset): The dataset for evaluation.
        sparsity_levels (list): List of sparsity levels to test.
        device (str): Device to run the experiment on ('cpu' or 'cuda').
        kwargs (dict): Additional arguments for model initialization.

    Returns:
    -------
        pd.DataFrame: Results of the experiment.

    """
    results = []

    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=5, num_workers=4, pin_memory=True)

    for sparsity in sparsity_levels:
        logging.info("Testing sparsity level: %.2f%%", sparsity * 100)

        # Clone the model to avoid modifying the original
        pruned_model = models.initialize_model(model_class, weights_path, kwargs, device=device)

        # Apply pruning
        global_unstructured_pruining(pruned_model, amount=sparsity)

        # Evaluate the model
        dict_metrics = model_evaluation(
            model=pruned_model,
            dataloader=dataloader,
            num_classes=2,
            class_names=["No Change", "Change"],
            device=device,
            saving_method="display",
            is_siamese=True,
        )

        # Record results
        record = {
            "Sparsity (%)": sparsity * 100,
        }

        # Extract scalar values from the overall_metrics DataFrame
        overall_metrics = dict_metrics["overall_metrics"].iloc[0].to_dict()
        record.update(overall_metrics)

        # Append the record to the results list
        results.append(record)

    # Convert results to a DataFrame for display
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model and dataset setup
    model_class = models.SiameseResNetUNet
    weights_path = "models/Levir_CD_Siamese_ResNet18_Unet_20250106-184502_best_model.pth"
    dataset = ds.LevirCDDataset(
        origin_dir="data/Levir-cd-v2",
        type="val",
        transform=Augmentation_pipeline(image_size=(256, 256), mean=None, std=None, mode="val"),
    )
    kwargs = {
        "in_channels": 3,
        "out_channels": 2,
        "mode": "conc",
        "backbone_name": "resnet18",
        "pretrained": False,
        "freeze_backbone": True,
    }

    # Sparsity levels to test
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Run the experiment
    results_df = run_pruning_experiment(
        model_class=model_class,
        weights_path=weights_path,
        dataset=dataset,
        sparsity_levels=sparsity_levels,
        device=device,
        kwargs=kwargs,
    )

    # Display the results
    results_df.to_csv("pruning_experiment.csv")

    # Display the results
    # logging.info(results_df)
