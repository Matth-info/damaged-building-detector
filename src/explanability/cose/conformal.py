from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.explanability.cose.coverage_losses import compute_loss
from src.explanability.cose.help_funcs import (
    compute_risk_bound,
    one_hot_encoding_of_gt,
    split_dataset_idxs,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

# Configure logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class Conformalizer:
    """Conformalizer class for managing conformal prediction calibration and test splits.

    Attributes:
        model : torch.nn.Module
            The PyTorch model used for predictions.
        dataset : Dataset
            The dataset to be split and used for calibration/testing.
        random_seed : int | None
            Random seed for reproducibility; if None, no shuffling is performed.
        n_calib : int
            Number of calibration samples.
        device : torch.device
            Device on which computations are performed.
        conformal_set : Literal["lac", "aps"]
            Type of conformal set to use.

    Methods:
        split_dataset_cal_test():
            Splits the dataset into calibration and test indices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        random_seed: int | None,
        n_calib: int,
        device: torch.device | None = None,
        conformal_set: Literal["lac", "aps"] = "aps",
        # loss_type: Literal["binary", "miscoverage"] = "binary",
    ):
        """Initialize a Conformalizer object."""
        self.conformal_set = conformal_set
        self.model = model
        self.dataset = dataset
        self.device = device if device else torch.device("cpu")
        self.random_seed = random_seed  # if None, no shuffling is performed
        self.n_calib = n_calib
        # self.loss_type = loss_type

    def split_dataset_cal_test(self) -> None:  # , random_seed: int):
        """Split dataset into a calibration and test sets."""
        self.calibration_indices, self.test_indices = split_dataset_idxs(
            len_dataset=len(self.dataset),
            n_calib=self.n_calib,
            random_seed=self.random_seed,
        )


def compute_empirical_risks(
    dataset: Dataset,
    model: torch.nn.Module,
    conformalizer: Conformalizer,
    samples_ids: list[int],
    lambdas: list[float],
    loss_prms: dict,  # = "miscoverage",
    mincov: float | None = None,
) -> list[float]:
    """Compute the empirical risks for a given set of lambda values using calibration samples.

    Args:
    ----
        dataset (Dataset): The dataset containing calibration samples.
        model (torch.nn.Module): The PyTorch model used for predictions.
        conformalizer (Conformalizer): The conformal prediction object.
        samples_ids (list[int]): List of indices for calibration samples.
        lambdas (list[float]): List of lambda values to evaluate.
        loss_prms (dict): Parameters for the loss function, including the loss name and attrib_method
        mincov (float | None): Minimum coverage ratio for binary loss. Required if using "binary_loss".

    Returns:
    -------
        list[float]: A list of average empirical risks for each lambda value.

    Raises:
    ------
        ValueError: If `lambdas` is empty.
        ValueError: If `mincov` is None when using "binary_loss".
        ValueError: If an unknown loss name is provided in `loss_prms`.
        Exception: If an error occurs during loss computation for a sample.

    Notes:
    -----
        - The function iterates over calibration samples and computes the loss for each sample.
        - Supports two loss types: "miscoverage_loss" and "binary_loss".
        - The empirical risk is computed as the mean of losses across all calibration samples.

    """
    if len(lambdas) == 0:
        msg = f"ERROR: expected [lambdas] to be non-empty but got length of: {len(lambdas)}"
        raise ValueError(
            msg,
        )
    _losses = []

    loss_name = loss_prms["loss_name"]
    attrib_method_name = loss_prms["attrib_method"]

    for idx in samples_ids:  # , disable=(not verbose)):
        inputs = dataset[idx]
        images = inputs["image"].unsqueeze(0).to(conformalizer.device)  # shape (1, C, H, W)
        gt_mask = inputs["mask"].long()

        logits = model(images)
        logits = logits.squeeze(0)  # shape (C, H, W)
        probs = torch.nn.functional.softmax(logits, dim=0)

        n_total_recoded_labels = dataset.n_classes
        one_hot_gt = one_hot_encoding_of_gt(gt_mask, n_total_recoded_labels)

        output_losses = compute_loss(
            loss_type=loss_name,
            one_hot_semantic_mask=one_hot_gt.to(conformalizer.device),
            output_softmaxes=probs.to(conformalizer.device),
            lbds=lambdas,
            n_labs=dataset.n_classes,
            minimum_coverage_ratio=mincov,
            multimask_type=attrib_method_name,
        )

        pred_losses = output_losses.losses
        _losses.append(pred_losses)

    losses = np.array(_losses)
    empirical_risks = np.mean(losses, axis=0)  # Compute the average loss

    return empirical_risks.tolist()


# Note : Lambda => Coverage parameter
# Note : alpha => Acceptable risk


def lambda_optimization(
    dataset: Dataset,
    model: torch.nn.Module,
    conformalizer: Conformalizer,
    calibration_ids: list[int],
    loss_parameters: dict,
    search_parameters: dict,
    alpha_risk: float,
    mincov: float | None,
) -> tuple[float, dict, float, bool]:
    """Optimizes lambda for conformal prediction using calibration data.

    Args:
    ----
        dataset: Dataset object.
        model: PyTorch model.
        conformalizer (Conformalizer): Conformal prediction object.
        calibration_ids: List of calibration sample IDs.
        loss_parameters (dict): Parameters for the loss function.
        search_parameters (dict): Parameters for lambda search.
        alpha_risk (float): Acceptable risk level.
        mincov (float | None): Minimum coverage ratio.

    Returns:
    -------
        tuple: (optimal_lambda, risks, risk_bound, early_stopped)

    """
    alpha_tolerance = alpha_risk
    b_loss_bound = loss_parameters["b_loss_bound"]
    risk_bound = compute_risk_bound(alpha_tolerance, b_loss_bound, len(calibration_ids))

    lbd_lower = search_parameters["lbd_lower"]
    lbd_upper = search_parameters["lbd_upper"]
    n_iter = search_parameters["n_iter"]
    n_mesh = search_parameters["n_mesh"]
    lbd_tolerance = search_parameters["lbd_tolerance"]

    risks = {"lambdas": [], "avg_risks": []}
    early_stopped = False
    lambda_zero = [0.0]

    _logger.debug("Before computing empirical risks ")
    # Compute empirical risks for lambda=0
    emp_risks = compute_empirical_risks(
        dataset=dataset,
        model=model,
        conformalizer=conformalizer,
        samples_ids=calibration_ids,
        lambdas=lambda_zero,
        loss_prms=loss_parameters,
        mincov=mincov,
    )
    _logger.debug("After computing empirical risks ")
    if len(emp_risks) == 0:
        raise ValueError("compute_empirical_risks returned an empty list.")

    empirical_risk_lambda_zero = emp_risks[0]

    _logger.info("\n ======= PRELIM CHECK FOR LAMBDA=0 =======")
    _logger.info("alpha_risk = %f", alpha_risk)
    _logger.info("empirical_risk_lambda_zero = %f", empirical_risk_lambda_zero)
    _logger.info("risk_bound = %f", risk_bound)

    # Early stop if lambda=0 is sufficient
    if empirical_risk_lambda_zero <= risk_bound:
        early_stopped = True
        risks["lambdas"].append(lambda_zero)
        risks["avg_risks"].append(list(emp_risks))
        optimal_lambda = lambda_zero[0]
        torch.cuda.empty_cache()
        _logger.info("EARLY STOPPED: lambda=0 is sufficient.")
        return optimal_lambda, risks, risk_bound, early_stopped

    _logger.info("Continuing optimization...")

    # Lambda optimization loop
    for _ in tqdm(range(n_iter)):
        if abs(lbd_upper - lbd_lower) < lbd_tolerance:
            break

        step_size = (lbd_upper - lbd_lower) / n_mesh
        lbds = np.concatenate(
            (np.arange(lbd_lower, lbd_upper, step_size), np.array([lbd_upper])),
        )

        emp_risks = compute_empirical_risks(
            dataset=dataset,
            model=model,
            conformalizer=conformalizer,
            samples_ids=calibration_ids,
            lambdas=lbds,
            loss_prms=loss_parameters,
            mincov=mincov,
        )
        if len(emp_risks) != len(lbds):
            error_msg = f"Mismatch between lambda values ({len(lbds)}) and empirical risks ({len(emp_risks)})."
            raise ValueError(error_msg)

        risks["lambdas"].append(list(lbds))
        risks["avg_risks"].append(list(emp_risks))

        # Update bounds based on risks
        if emp_risks[-1] > risk_bound:
            lbd_lower = lbds[-1]
        elif emp_risks[0] <= risk_bound:
            lbd_upper = lbds[0]
        else:
            for l_, risk in zip(lbds, emp_risks):
                if risk > risk_bound:
                    lbd_lower = l_
                elif risk <= risk_bound:
                    lbd_upper = l_
                    break

    optimal_lambda = lbd_upper
    torch.cuda.empty_cache()
    return optimal_lambda, risks, risk_bound, early_stopped


def compute_losses_on_test(
    dataset: Dataset,
    model: torch.nn.Module,
    conformalizer: Conformalizer,
    samples_ids: list[int],
    lbd: float,
    minimum_coverage: float,
    loss_name: str,
) -> np.ndarray:
    """Compute losses, activations, and coverage ratios for test samples using a conformal prediction model.

    Args:
    ----
        dataset (Dataset): The dataset containing test samples.
        model (torch.nn.Module): The PyTorch model used for predictions.
        conformalizer (Conformalizer): The conformal prediction object.
        samples_ids (list[int]): List of indices for test samples.
        lbd (float): Lambda value used for thresholding in conformal prediction.
        minimum_coverage (float): Minimum coverage ratio for binary loss.
        loss_name (callable): Loss function to compute metrics (e.g., `binary_loss` or `miscoverage_loss`).

    Returns:
    -------
        np.ndarray: A NumPy array containing losses, activations ratios, and empirical coverage ratios.

    Raises:
    ------
        ValueError: If `lbd` is not a float.
        Exception: If an error occurs during loss computation for a sample.

    Notes:
    -----
        - The function iterates over test samples and computes losses, activations, and coverage ratios.
        - The loss function must support the required arguments (e.g., `binary_loss` or `miscoverage_loss`).
        - The output is a NumPy array with three rows: losses, activations ratios, and coverage ratios.

    """
    if not isinstance(lbd, float):
        msg = f"ERROR: expected [lbd] to be a float, but got type: {type(lbd)}"
        raise TypeError(
            msg,
        )
    losses = []
    activations = []
    empirical_coverage_ratio = []

    for _, idx in tqdm(enumerate(samples_ids)):
        inputs = dataset[idx]
        image = inputs["image"].unsqueeze(0).to(conformalizer.device)
        gt_mask = inputs["mask"].long()

        logits = model(image)
        logits = logits.squeeze(0)
        softmax_prediction = torch.nn.functional.softmax(logits, dim=0)

        one_hot_gt = one_hot_encoding_of_gt(
            gt_mask,
            dataset.n_classes,
        )

        output_losses = compute_loss(
            loss_type=loss_name,
            one_hot_semantic_mask=one_hot_gt.to(conformalizer.device),
            output_softmaxes=softmax_prediction.to(conformalizer.device),
            lbds=[lbd],
            minimum_coverage_ratio=minimum_coverage,
            n_labs=dataset.n_classes,
        )

        losses.append(output_losses.losses[0])  # [0] because we only have one lambda
        activations.append(output_losses.activations_ratio[0])
        empirical_coverage_ratio.append(output_losses.coverage_ratio[0])

    return np.array((losses, activations, empirical_coverage_ratio))
