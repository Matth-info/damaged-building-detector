"""Gather several Pytorch Dataset implementation."""

from .cloud_DrivenData import CloudDrivenDataDataset
from .inference_dataset import DatasetInference, DatasetInferenceSiamese
from .levir_cd import LevirCDDataset
from .open_cities import OpenCities_Building_Dataset
from .puerto_rico import Puerto_Rico_Building_Dataset
from .xDB import xDB_Instance_Building, xDB_Siamese_Dataset, xDBDamagedBuilding

DATASETS_MAP = {
    "CloudDrivenDataDataset": CloudDrivenDataDataset,
    "Puerto_Rico_Building_Dataset": Puerto_Rico_Building_Dataset,
    "OpenCities_Building_Dataset": OpenCities_Building_Dataset,
    "xDBDamagedBuilding": xDBDamagedBuilding,
    "xDB_Instance_Building": xDB_Instance_Building,
    "xDB_Siamese_Dataset": xDB_Siamese_Dataset,
    "LevirCDDataset": LevirCDDataset,
}
