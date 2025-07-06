"""Loss implementations from https://github.com/qubvel-org/segmentation_models.pytorch."""

from torch import nn

from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss
from .ensemble import Ensemble
from .focal import FocalLoss
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss
from .mcc import MCCLoss
from .ordinal import O2Loss, OrdinalCrossEntropy, WeightedCategoricalCrossEntropy
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss
from .tversky import TverskyLoss

__all__ = [
    "BINARY_MODE",
    "MULTICLASS_MODE",
    "MULTILABEL_MODE",
    "DiceLoss",
    "Ensemble",
    "FocalLoss",
    "JaccardLoss",
    "LovaszLoss",
    "MCCLoss",
    "O2Loss",
    "OrdinalCrossEntropy",
    "SoftBCEWithLogitsLoss",
    "SoftCrossEntropyLoss",
    "TverskyLoss",
    "WeightedCategoricalCrossEntropy",
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
    "Weighted_Categorical_CrossEntropy": WeightedCategoricalCrossEntropy,
    "Ordinal_CrossEntropy": OrdinalCrossEntropy,
    "O2_Loss": O2Loss,
}
