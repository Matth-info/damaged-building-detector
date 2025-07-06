from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union, namedtuple, Any

import numpy as np

from src.explanability.cose.attribution_methods import (
    CONFORMAL_SETS_REGISTER,
    aps_multimask,
    choose_attribution_method,
    lac_multimask,
)
from src.explanability.cose.help_funcs import (
    compute_activable_pixels,
    is_semantic_mask_in_multimask,
)

if TYPE_CHECKING:
    import torch

LossOutput = namedtuple(
    "LossOutput",
    ["losses", "activations_ratio", "coverage_ratio", "minimum_coverage_ratio"],
)


def compute_loss(
    loss_type: str,
    one_hot_semantic_mask: torch.Tensor,
    output_softmaxes: torch.Tensor,
    lbds: float | Sequence[float] | np.ndarray,
    n_labs: int,
    minimum_coverage_ratio: float | None = None,
    multimask_type: Literal["lac", "aps"] = "lac",
    **kwargs: object,
) -> LossOutput:  # type: ignore
    """Compute loss dynamically based on the specified loss type.

    Args:
    ----
        loss_type (str): Type of loss to compute ("binary_loss" or "miscoverage_loss").
        one_hot_semantic_mask (torch.Tensor): Ground truth mask in one-hot encoding.
        output_softmaxes (torch.Tensor): Predicted softmax probabilities.
        lbds (float | Sequence[float] | np.ndarray): Lambda values for thresholding.
        n_labs (int): Number of ground truth classes.
        minimum_coverage_ratio (float | None): Minimum coverage ratio for binary loss.
        multimask_type (Literal["lac", "aps"]): Type of multimask to use ("lac" or "aps").
        **kwargs: Additional arguments for the loss functions.

    Returns:
    -------
        LossOutput: Named tuple containing losses, activations ratio, coverage ratio, and minimum coverage ratio.

    Raises:
    ------
        ValueError: If an invalid loss type is provided.

    """
    # Dynamically select the loss function
    if loss_type == "binary_loss":
        return binary_loss(
            one_hot_semantic_mask=one_hot_semantic_mask,
            output_softmaxes=output_softmaxes,
            lbds=lbds,
            n_labs=n_labs,
            minimum_coverage_ratio=minimum_coverage_ratio,
            multimask_type=multimask_type,
            **kwargs,
        )
    if loss_type == "miscoverage_loss":
        if minimum_coverage_ratio is None:
            raise ValueError(
                "ERROR: with [binary_loss], expected [minimum_coverage_ratio] cannot be [None]",
            )
        return miscoverage_loss(
            one_hot_semantic_mask=one_hot_semantic_mask,
            output_softmaxes=output_softmaxes,
            lbds=lbds,
            n_labs=n_labs,
            multimask_type=multimask_type,
            minimum_coverage_ratio=minimum_coverage_ratio,
            **kwargs,
        )
    msg = f"Invalid loss type: {loss_type}. Must be 'binary_loss' or 'miscoverage_loss'."
    raise ValueError(
        msg,
    )


def binary_loss(
    one_hot_semantic_mask: torch.Tensor,  # can contain extra channel for void pixels
    output_softmaxes: torch.Tensor,
    lbds: float | Sequence[float] | np.ndarray,
    n_labs: int,  # <-- [dataset.n_classes], being the num of ground-truth classes
    minimum_coverage_ratio: float,
    multimask_type: Literal["lac", "aps"] = "lac",
    **kwargs: object,
) -> Any:
    """Compute the binary loss for coverage evaluation, determining if the coverage ratio of the predicted mask is below the minimum required threshold.

    Args:
        one_hot_semantic_mask (torch.Tensor): Ground truth mask in one-hot encoding, may include void pixels.
        output_softmaxes (torch.Tensor): Predicted softmax probabilities.
        lbds (float | Sequence[float] | np.ndarray): Lambda values for thresholding.
        n_labs (int): Number of ground truth classes.
        minimum_coverage_ratio (float): Minimum required coverage ratio.
        multimask_type (Literal["lac", "aps"]): Type of multimask to use.
        **kwargs: Additional arguments.

    Returns:
        LossOutput: Named tuple containing losses, activations ratio, coverage ratio, and minimum coverage ratio.

    Raises:
        ValueError: If input dimensions or types are invalid.
    """
    delta_labs = one_hot_semantic_mask.shape[0] - n_labs

    if delta_labs >= 0:
        msg = f" [{one_hot_semantic_mask.shape[0] = }] must be >= to n_labs [{n_labs}]"
        raise ValueError(msg)

    multimask_func = choose_attribution_method(name=multimask_type)

    num_activable_pixels = compute_activable_pixels(
        one_hot_semantic_mask=one_hot_semantic_mask,
        n_labs=n_labs,
    )

    if num_activable_pixels == 0:
        msg = f"num_activable_pixels [{num_activable_pixels}] must be > 0 for n_labs [{n_labs}]"
        raise ValueError(
            msg,
        )

    if isinstance(lbds, float):
        lbds = [lbds]
    elif isinstance(lbds, (Sequence, np.ndarray)):
        pass
    else:
        msg = f"lbds must be float or sequence of floats, not {type(lbds)}"
        raise TypeError(msg)

    output = LossOutput(
        losses=[],
        activations_ratio=[],
        coverage_ratio=[],
        minimum_coverage_ratio=[],
    )
    for lbd in lbds:
        thresholded_one_hot_multimask = multimask_func(
            threshold=lbd,
            predicted_softmax=output_softmaxes,
            n_labels=n_labs,
        )

        binary_test = is_semantic_mask_in_multimask(
            one_hot_semantic_mask=one_hot_semantic_mask,
            one_hot_multimask=thresholded_one_hot_multimask,
        )  ## pixel_ij == 1: ground truth mask_ij is covered by multimask_ij

        if binary_test.shape[0] == n_labs:
            msg = f" [{binary_test.shape[0] = }] must be equal to n_labs [{n_labs}]"
            raise ValueError(msg)
        cov_ratio = binary_test.cpu().numpy().sum() / num_activable_pixels

        if np.isnan(cov_ratio):
            msg = f"[COSE] ERROR: got NaN in miscoverage_ratio for lbd {lbd} and n_labs {n_labs}"
            raise ValueError(
                msg,
            )

        binary_loss_value = int(cov_ratio < minimum_coverage_ratio)

        cov_ratio = binary_test.sum().cpu().numpy() / num_activable_pixels

        activable_pixels = thresholded_one_hot_multimask

        n_activations = activable_pixels.sum().cpu().numpy()
        activations_ratio = n_activations / num_activable_pixels

        output.losses.append(binary_loss_value)
        output.activations_ratio.append(float(activations_ratio))
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(minimum_coverage_ratio)

    return output


def miscoverage_loss(
    one_hot_semantic_mask: torch.Tensor,  # can contain extra channel for void pixels
    output_softmaxes: torch.Tensor,
    lbds: float | Sequence[float] | np.ndarray,
    n_labs: int,  # <-- [dataset.n_classes], being the num of non-void classes
    multimask_type: Literal["lac", "aps"] = "lac",
    *,
    minimum_coverage_ratio: bool = False,
    **kwargs: object,
) -> Any:
    """Miscoverage loss for CRC. If ground truth mask is entirely contained in multimask parametrized by [lbd] then return 1, else 0.

    Args:
    ----
        one_hot_semantic_mask (torch.Tensor): Ground truth mask in one-hot encoding, may include void pixels.
        output_softmaxes (torch.Tensor): Predicted softmax probabilities.
        lbds (float | Sequence[float] | np.ndarray): Lambda values for thresholding.
        n_labs (int): Number of ground truth classes.
        multimask_type (Literal["lac", "aps"]): Type of multimask to use ("lac" or "aps").
        minimum_coverage_ratio (bool, optional): Minimum required coverage ratio. Defaults to False.
        **kwargs: Additional arguments.

    Returns:
    -------
        LossOutput: Named tuple containing losses, activations ratio, coverage ratio, and minimum coverage ratio.

    """
    # [delta_labs]: how many "excess" extra labs there are.
    #               Assumes [n_labs] is num of ground truth labs
    if multimask_type == "lac":
        multimask_func = lac_multimask
    elif multimask_type == "aps":
        multimask_func = aps_multimask
    else:
        msg = f"multimask_type must be in {CONFORMAL_SETS_REGISTER}, not {multimask_type}"
        raise ValueError(
            msg,
        )

    num_activable_pixels = compute_activable_pixels(
        one_hot_semantic_mask=one_hot_semantic_mask,
        n_labs=n_labs,
    )

    if num_activable_pixels == 0:
        msg = f"num_activable_pixels [{num_activable_pixels}] must be > 0 for n_labs [{n_labs}]"
        raise ValueError(
            msg,
        )

    if isinstance(lbds, float):
        lbds = [lbds]
    elif isinstance(lbds, (Sequence, np.ndarray)):
        pass
    else:
        msg = f"lbds must be float or sequence of floats, not {type(lbds)}"
        raise TypeError(msg)

    for lbd in lbds:
        output = LossOutput(
            losses=[],
            activations_ratio=[],
            coverage_ratio=[],
            minimum_coverage_ratio=[],
        )
        thresholded_one_hot_multimask = multimask_func(
            threshold=lbd,
            predicted_softmax=output_softmaxes,
            n_labels=n_labs,
        )

        binary_test = is_semantic_mask_in_multimask(
            one_hot_semantic_mask=one_hot_semantic_mask,
            one_hot_multimask=thresholded_one_hot_multimask,
        )  ## Binary Covering Test such that pixel_ij == 1: ground truth mask_ij is covered by multimask_ij
        n_captured = (
            binary_test.sum().cpu().numpy().astype(int)
        )  # number of activated pixels (covered by the set-prediction mask)

        if binary_test.shape[0] == n_labs:
            msg = f" [{binary_test.shape[0] = }] must be equal to n_labs [{n_labs}]"
            raise ValueError(msg)

        n_pixels = one_hot_semantic_mask.shape[1] * one_hot_semantic_mask.shape[2]

        cov_ratio = (
            binary_test.sum() / num_activable_pixels
        )  # share of activated pixels in the image
        cov_ratio = cov_ratio.cpu().numpy()
        miscoverage_ratio = 1 - cov_ratio  # share of miscovered pixels in the image

        activable_pixels = thresholded_one_hot_multimask

        n_activations = activable_pixels.sum().cpu().numpy()
        activations_ratio = n_activations / num_activable_pixels

        # add data to the output
        output.losses.append(miscoverage_ratio)  # the ratio of error (in CRC)
        output.activations_ratio.append(float(activations_ratio))
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(minimum_coverage_ratio)

        if np.isnan(miscoverage_ratio):
            msg = f"[COSE] ERROR: got NaN in miscoverage_ratio for lbd {lbd} and n_labs {n_labs}"
            raise ValueError(
                msg,
            )

        ## as many appends as there are lbds
        output.losses.append(miscoverage_ratio)
        output.activations_ratio.append(activations_ratio)
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(None)

    return output
