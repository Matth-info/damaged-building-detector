from pathlib import Path

import torch

CONFORMAL_SETS_REGISTER = ["lac", "aps"]


def choose_attribution_method(name: str = "lac") -> callable:
    """Select and returns the attribution method function based on the provided name.

    Parameters
    ----------
    name : str, optional
        The name of the attribution method to use ("lac" or "aps"). Default is "lac".

    Returns:
    -------
    callable
        The selected attribution method function.

    Raises:
    ------
    ValueError
        If the provided name is not in CONFORMAL_SETS_REGISTER.
    """
    name = name.lower()
    if name == "lac":
        multimask_func = lac_multimask
    elif name == "aps":
        multimask_func = aps_multimask
    else:
        msg = f"multimask_type must be in {CONFORMAL_SETS_REGISTER}, not {name}"
        raise ValueError(
            msg,
        )
    return multimask_func


@torch.no_grad()
def aps_multimask(
    threshold: float,
    predicted_softmax: torch.Tensor,
    n_labels: int,
    *,
    always_include_top1: bool = True,
) -> torch.Tensor:
    """Compute multimask using Adaptative Predict Set (APS) conformal method.

    Always includes the top-1 class to avoid empty pixels.

    Args:
    ----
        threshold (float): Threshold for cumulative softmax values.
        predicted_softmax (torch.Tensor): Softmax predictions with shape (n_classes, H, W).
        n_labels (int): Number of classes in the dataset.
        always_include_top1 (bool): Whether to always include the top-1 class.

    Returns:
    -------
        torch.Tensor: A one-hot encoded tensor representing the multimask.

    """
    ordered_softmax, indices = torch.topk(predicted_softmax, k=n_labels, dim=0)
    test_over_threshold = ordered_softmax.cumsum(0) > threshold
    up_to_first_over = test_over_threshold.cumsum(0)
    test_up_to_first = up_to_first_over <= 1

    if always_include_top1:
        test_up_to_first[0, :, :] = 1  # Always include top-1 class

    reverse_indices = torch.argsort(indices, dim=0, descending=False)
    one_hot_attribution = test_up_to_first.gather(0, reverse_indices)

    del test_up_to_first, reverse_indices, ordered_softmax, indices
    torch.cuda.empty_cache()

    return one_hot_attribution.long()


@torch.no_grad()
def lac_multimask(
    threshold: float,
    predicted_softmax: torch.Tensor,
    n_labels: int,
    *,
    always_include_top1: bool = True,
) -> torch.Tensor:
    """Compute multimask using Least Ambiguous Set-Valued Classifiers (LAC) method.

    Includes all classes whose softmax value is higher than a given threshold.

    Args:
    ----
        threshold (float): Threshold for softmax values.
        predicted_softmax (torch.Tensor): Softmax predictions with shape (n_classes, H, W).
        n_labels (int): Number of classes in the dataset.
        always_include_top1 (bool): Whether to always include the top-1 class.

    Returns:
    -------
        torch.Tensor: A one-hot encoded tensor representing the multimask.

    """
    if n_labels > predicted_softmax.shape[0]:
        msg = f"n_labels [{n_labels}] cannot exceed the number of classes in predicted_softmax [{predicted_softmax.shape[0]}]"
        raise ValueError(
            msg,
        )

    inverse_threshold = 1 - threshold  # Classes above this value are included
    ordered_softmaxes, indices = torch.topk(predicted_softmax, k=n_labels, dim=0)
    test_over_threshold = ordered_softmaxes >= inverse_threshold

    if always_include_top1:
        test_over_threshold[0, :, :] = 1  # Always include top-1 class

    reverse_indices = torch.argsort(indices, dim=0, descending=False)
    one_hot_multimask = test_over_threshold.gather(0, reverse_indices)

    del ordered_softmaxes, reverse_indices, test_over_threshold, indices

    return one_hot_multimask.long()  # Multilabel mask with shape (n_labels, H, W)
