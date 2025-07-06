"""Package breaking down model training phase."""

import torch

from .train import Trainer
from .utils import define_weighted_random_sampler, initialize_optimizer_scheduler

OPTIMIZER_MAP = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
}

SCHEDULER_MAP = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}
