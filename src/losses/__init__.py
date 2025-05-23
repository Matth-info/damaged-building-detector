import torch.nn as nn

from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss
from .ensemble import Ensemble
from .focal import FocalLoss
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss
from .mcc import MCCLoss
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss
from .tversky import TverskyLoss

__all__ = [
    "BINARY_MODE",
    "MULTICLASS_MODE",
    "MULTILABEL_MODE",
    "JaccardLoss",
    "DiceLoss",
    "FocalLoss",
    "LovaszLoss",
    "SoftBCEWithLogitsLoss",
    "SoftCrossEntropyLoss",
    "TverskyLoss",
    "MCCLoss",
    "Ensemble",
]

LOSSES_MAP = {
    "JaccardLoss": JaccardLoss,
    "DiceLoss": DiceLoss,
    "FocalLoss": FocalLoss,
    "LovaszLoss": LovaszLoss,
    "SoftBCEWithLogitsLoss": SoftBCEWithLogitsLoss,
    "SoftCrossEntropyLoss": SoftCrossEntropyLoss,
    "TverskyLoss": TverskyLoss,
    "MCCLoss": MCCLoss,
    "Ensemble": Ensemble,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}
