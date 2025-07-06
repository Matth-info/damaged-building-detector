from typing import Optional, Union
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from src.datasets import DATASETS_MAP
from src.models import MODELS_MAP


class ExperimentConfig(BaseModel):
    """Experiment Configuration."""

    name: str
    log_dir: str
    model_dir: str
    training_log_interval: int
    checkpoint_interval: int
    debug: bool
    verbose: bool
    is_siamese: bool


ModelType = Literal.__getitem__(tuple(MODELS_MAP.keys()))


class ModelConfig(BaseModel):
    """Model Configuration."""

    name: ModelType = Field(description="Available model type.")  # type: ignore
    in_channels: int = Field(gt=1)
    out_channels: int = Field(gt=2)
    freeze_backbone: bool = True
    backbone: str
    pretrained: bool = True
    mode: Literal["conc", "diff", None] = None
    checkpoint_path: Optional[str] = None


# extract available Dataset type
DatasetType = Literal.__getitem__(tuple(DATASETS_MAP.keys()))


class DataConfig(BaseModel):
    """Data Configuration."""

    dataset: DatasetType = Field(description="Available Dataset type.")  # type: ignore
    origin_dir: str
    type: str


class AugmentationConfig(BaseModel):
    """Augmentation Pipeline Configuration."""

    config_path: Optional[str]
    image_size: list[int]
    normalize_mean: list[float]
    normalize_std: list[float]
    mode: str


class DataLoaderConfig(BaseModel):
    """DataLoader Configuration."""

    batch_size: int
    num_workers: int
    sampler: Optional[str]
    mask_key: str
    image_key: str


class OptimizerParams(BaseModel):
    """Optimizer Configuration."""

    lr: float
    weight_decay: float


class SchedulerParams(BaseModel):
    """Scheduler Configuration."""

    step_size: int
    gamma: float


class EarlyStoppingConfig(BaseModel):
    """Earlystopping Configuration."""

    patience: int
    trigger_times: int


class TrainingConfig(BaseModel):
    """Training Configuration."""

    nb_epochs: int
    optimizer: dict
    scheduler: dict
    loss_fn: str
    is_mixed_precision: bool
    reduction: Literal[
        "weighted",
        "macro",
        "micro",
        "micro-imagewise",
        "macro-imagesize",
        "weighted-imagewise",
        "none",
        None,
    ]
    class_weights: list[float]
    gradient_accumulation_steps: int
    max_norm: float
    early_stopping: EarlyStoppingConfig


from src.metrics import METRICS_MAP

MetricsType = Literal.__getitem__(tuple(METRICS_MAP.keys()))


class TestingConfig(BaseModel):
    """Testing Configuration."""

    metrics: list[MetricsType] = Field(
        description="list of evaluation metrics to use. Must be from the predefined metric set."
    )
    tta: bool
    class_names: list[str]


class LoggerConfig(BaseModel):
    """Training Logger Configuration."""

    type: Literal["mlflow", "tensorboard"]
    tracking_uri: str
    enable_system_metrics: bool
    task: str


# Root config
class Config(BaseModel):
    """Overall Training Configuration."""

    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    data_loader: DataLoaderConfig
    training: TrainingConfig
    testing: TestingConfig
    logger: LoggerConfig

    @model_validator(mode="after")
    def validate_class_match(self) -> "Config":
        """Validate consistency of class counts across model, training, and testing config."""
        n_classes = self.model.out_channels
        class_names = self.testing.class_names
        class_weights = self.training.class_weights

        if class_names and len(class_names) != n_classes:
            raise ValueError(
                f"Mismatch: model has {n_classes} classes, but {len(class_names)} class names provided."
            )

        if self.training.reduction == "weighted":
            if class_weights is None or len(class_weights) != n_classes:
                raise ValueError(
                    f"Weighted reduction requires {n_classes} class weights, but got {class_weights}."
                )

        return self
