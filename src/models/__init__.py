from .Auto_Encoder import AutoEncoder
from .bit.network import BiT
from .changeformer.ChangeFormer import ChangeFormerV6 as ChangeFormer
from .Ensemble import EnsembleModel
from .MaskRCNN import Maskrcnn
from .resnet_unet.Resnet_Unet import ResNet_UNET
from .resnet_unet.Siamese_Resnet_Unet import SiameseResNetUNet
from .Segformer import Segformer
from .tiny_cd.change_classifier import TinyCD
from .unet.network import UNet
from .utils import initialize_model

__all__ = [
    "AutoEncoder",
    "ResNet_UNET",
    "Segformer",
    "BiT",
    "UNet",
    "Maskrcnn",
    "SiameseResNetUNet",
    "EnsembleModel",
    "TinyCD",
    "ChangeFormer",
    "initialize_model",
]

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
}
