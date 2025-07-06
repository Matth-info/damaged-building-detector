from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.explanability.cose import Conformalizer, compute_losses_on_test, lambda_optimization
from src.explanability.cose.help_funcs import RoundDataset, TrivialModel
from src.explanability.cose.visual import plot_overlay

if TYPE_CHECKING:
    from torch.utils.data import Dataset

# Configure logging
_logger = logging.getLogger(__name__)
"""logging.basicConfig(
    filename="conformal_prediction_experiments.log", encoding="utf-8", level=logging.DEBUG,
)"""
_logger.setLevel(logging.INFO)  # configurable; setup Handlers in MLOps pipeline


def run_crc_experiment(
    conformal_model: Conformalizer,
    calib_dataset: Dataset,
    calib_ids: list[int],
    random_seed: int,
    alpha: float,
    loss_params: dict,
    search_params: dict,
    output_directory: str,
    experiment_name: str,
    mincov: float | None = None,
) -> float:
    """Run a Conformal Risk Control (CRC) experiment to find the optimal lambda for a given risk level and calibration dataset.

    Args:
        conformal_model (Conformalizer): The conformalizer model to use.
        calib_dataset (Dataset): The calibration dataset.
        calib_ids (list[int]): List of calibration sample indices.
        random_seed (int): Random seed for reproducibility.
        alpha (float): Risk level (smaller is better).
        loss_params (dict): Parameters for the loss function.
        search_params (dict): Parameters for lambda search.
        output_directory (str): Directory to save experiment results.
        experiment_name (str): Name of the experiment.
        mincov (float | None, optional): Minimum coverage threshold.

    Returns:
        float: The optimal lambda value found during the experiment.
    """
    with torch.no_grad():
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%Hh%Mm%Ss")
        loss_name = loss_params["loss_name"]
        filename_out = f"{timestamp}_{calib_dataset.name}__id_{random_seed}__alpha_{alpha}__mincov_{mincov}__{loss_name}.json"
        output_path = Path(output_directory) / f"{filename_out}"
        _logger.info("------ output path : %s", output_path)

        optimal_lambda, risks, risk_bound, early_stopped = lambda_optimization(
            dataset=calib_dataset,
            model=conformal_model.model,
            conformalizer=conformal_model,
            calibration_ids=calib_ids,
            loss_parameters=loss_params,
            search_parameters=search_params,
            alpha_risk=alpha,
            mincov=mincov,
        )

        _logger.info(" ----End of optim lambda with results:")
        _logger.info(" ----- Optimal lambda : %f", optimal_lambda)
        _logger.info(" ----- Risk : %f", risk_bound)
        _logger.info(" ----------------------------------------")
        results = {
            "exp_name": experiment_name,
            "alpha": alpha,  # risk level smaller is better
            "alpha_risk_bound": risk_bound,
            "early_stopped": early_stopped,
            "mincov": mincov,  # loss_params["minimum_coverage_threshold"],
            "dataset": calib_dataset.name,
            "experiment_id": random_seed,
            "optimal_lambda": optimal_lambda,
            "loss_function": loss_params["loss_name"],
            "risks": risks,
            "lbd_search_params": search_params,
            "n_calib": conformal_model.n_calib,
            "cal_id": conformal_model.calibration_indices,
        }

        with Path.open(output_path, "w") as f:
            json.dump(results, f)

        torch.cuda.empty_cache()

        return optimal_lambda


if __name__ == "__main__":
    # Parameters for testing
    num_samples = 200
    image_size = (64, 64)
    n_classes = 5
    random_seed = 42
    n_calib = 100
    alpha = 0.01
    min_cov = 0.8
    loss_params = {"loss_name": "binary_loss", "b_loss_bound": 1.0, "attrib_method": "lac"}
    search_params = {
        "lbd_lower": 0.0,
        "lbd_upper": 1.0,
        "n_iter": 5,
        "n_mesh": 10,
        "lbd_tolerance": 0.01,
    }
    output_directory = "./src/explanability/outputs"
    Path.mkdir(output_directory, exist_ok=True)
    experiment_name = "test_experiment"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)

    # Initialize dataset, model, and conformalizer
    dataset = RoundDataset(
        length=num_samples,
        is_siamese=False,
        image_size=image_size[0],
        num_classes=n_classes,
    )
    model = TrivialModel(num_classes=n_classes).to(device)

    conformalizer = Conformalizer(
        model=model,
        dataset=dataset,
        random_seed=random_seed,
        n_calib=n_calib,
        device=device,
        conformal_set=loss_params["attrib_method"],
    )
    _logger.info("conformalizer initialized successfully")

    # Split dataset into calibration and test sets
    conformalizer.split_dataset_cal_test()

    # Run CRC experiment
    optimal_lambda = run_crc_experiment(
        conformal_model=conformalizer,
        calib_dataset=dataset,
        calib_ids=conformalizer.calibration_indices,
        random_seed=random_seed,
        alpha=alpha,
        loss_params=loss_params,
        search_params=search_params,
        output_directory=output_directory,
        experiment_name=experiment_name,
        mincov=min_cov,
    )  # for a given risk alpha CRC experiment return optimal lambda

    test_losses_np_array = compute_losses_on_test(
        dataset=dataset,
        model=conformalizer.model,
        conformalizer=conformalizer,
        samples_ids=conformalizer.test_indices,
        lbd=optimal_lambda,
        minimum_coverage=min_cov,
        loss_name=loss_params["loss_name"],
    )  # each prediction gets its loss value, activations ratio, and empirical coverage ratio

    averages = test_losses_np_array.mean(axis=1)

    _logger.info("Optimal lambda : %f", optimal_lambda)
    _logger.info(
        "Average empirical risk : %f", averages[0]
    )  # average ratio non coverage pixels per image
    _logger.info("Average activation ratio : %d / %d classes", int(averages[1]), dataset.n_classes)
    # average size of the prediction set
    _logger.info(
        "average empirical coverage ratio : %f",
        averages[2],
    )  # average ratio of covered pixels per image

    # Test display methods
    plot_overlay(
        idx=rng.choice(conformalizer.test_indices),
        dataset=dataset,
        model=conformalizer.model,
        alpha_nominal=alpha,
        threshold=optimal_lambda,
        mode=loss_params["attrib_method"],
        figsize=(12, 6),
        colormap="turbo",
        device=device,
    )
