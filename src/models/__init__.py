"""Custom implementation of Pytorch Semantic Segmentation Models."""

from .Auto_Encoder import AutoEncoder
from .bit.network import BiT
from .changeformer.ChangeFormer import ChangeFormer
from .Ensemble import EnsembleModel
from .foundational.models.clay_seg import ClaySegmentor
from .foundational.models.prithvi_seg import PrithviSeg
from .foundational.models.scale_mae_seg import ScaleMaeSeg
from .MaskRCNN import Maskrcnn
from .resnet_unet.Resnet_Unet import ResNet_UNET
from .resnet_unet.Siamese_Resnet_Unet import SiameseResNetUNet
from .Segformer import Segformer
from .tiny_cd.change_classifier import TinyCD
from .unet.network import UNet
from .utils import initialize_model

MODELS_MAP = {
    "AutoEncoder": AutoEncoder,
    "ResNet_UNET": ResNet_UNET,
    "Segformer": Segformer,
    "BiT": BiT,
    "UNet": UNet,
    "Maskrcnn": Maskrcnn,
    "SiameseResNetUNet": SiameseResNetUNet,
    "EnsembleModel": EnsembleModel,
    "TinyCD": TinyCD,
    "ChangeFormer": ChangeFormer,
    "ClaySegmentor": ClaySegmentor,
    "PrithviSeg": PrithviSeg,
    "ScaleMaeSeg": ScaleMaeSeg,
}
