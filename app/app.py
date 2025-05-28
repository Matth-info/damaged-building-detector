import albumentations as A
import gradio as gr
import torch
from albumentations.pytorch import ToTensorV2

from src.augmentation import Augmentation_pipeline
from src.datasets import Levir_cd_dataset
from src.models import SiameseResNetUNet
from src.training.utils import apply_color_map
from src.utils.visualization import DEFAULT_MAPPING


def get_augmentation_pipeline(image_size=None, max_pixel_value=255, mean=None, std=None):
    transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]) if image_size is not None else A.NoOp(),
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0)
            if mean and std
            else A.NoOp(),
            ToTensorV2(),
        ],
        additional_targets={
            "post_image": "image",
        },
    )
    return transform


def predict(x1, x2):
    """Runs change detection on input images and returns the colorized prediction mask.

    Parameters:
        x1, x2 : np.ndarray of shape (H, W, 3) - Pre and post images

    Returns:
        np.ndarray of shape (H, W, 3) - Colorized change detection mask
    """

    # Apply transformations
    transformed = transform(image=x1, post_image=x2)
    x1, x2 = transformed["image"], transformed["post_image"]  # Shape: (C, H, W)

    # Debugging
    print("Input Tensor Type:", type(x1))

    # Add batch dimension
    x1 = x1.unsqueeze(0).to(device, dtype=torch.float32)  # Shape: (1, C, H, W)
    x2 = x2.unsqueeze(0).to(device, dtype=torch.float32)

    # Perform inference
    pred_mask = model(x1, x2)  # Output shape: (1, 2, H, W)

    # Convert logits to class labels
    predictions = torch.argmax(pred_mask, dim=1).cpu().float()  # Shape: (1, H, W)

    # Apply color mapping
    colorized_pred = apply_color_map(
        mask=predictions, color_dict=DEFAULT_MAPPING, with_transparency=False
    )  # Shape: (1, 3, H, W)

    # Remove batch dimension
    colorized_pred = colorized_pred.squeeze(0)  # Shape: (3, H, W)

    # Convert from (3, H, W) to (H, W, 3) for Gradio
    colorized_pred = colorized_pred.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 3)

    return colorized_pred


# Load Model
model_path = "../models/Levir_CD_Siamese_ResNet18_Unet_20250106-184502_best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = Augmentation_pipeline(
    image_size=(256, 256),
    mean=Levir_cd_dataset.MEAN,
    std=Levir_cd_dataset.STD,
    mode="infer",
)

model = SiameseResNetUNet(
    in_channels=3,
    out_channels=2,
    backbone_name="resnet18",
    pretrained=False,
    mode="conc",
).to(device)

model = model.load(model_path).eval()  # Fix: Assign loaded model

# Define Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Pre Image", type="numpy"),
        gr.Image(label="Post Image", type="numpy"),
    ],
    outputs="image",
    flagging_mode="manual",
    flagging_dir="./flagged",
    flagging_options=["Incorrect", "Error"],
    examples=[
        [
            "../data/data_samples/Levir-cd/test/A/test_101.png",
            "../data/data_samples/Levir-cd/test/B/test_101.png",
        ],
        [
            "../data/data_samples/xDB/tier3/images/joplin-tornado_00000000_pre_disaster.png",
            "../data/data_samples/xDB/tier3/images/joplin-tornado_00000000_post_disaster.png",
        ],
    ],
)

if __name__ == "__main__":
    demo.launch(allowed_paths=["../data/data_samples/"])
