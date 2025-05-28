import yaml

from src.datasets import Levir_cd_dataset
from src.models import SiameseResNetUNet


def read_config(path: str) -> dict[str, str]:
    """load YAML file"""
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


MODEL_MAPPER = {
    "siamese_resnet_unet": SiameseResNetUNet,
}

DATASET_MAPPER = {"levir_cd": Levir_cd_dataset}
