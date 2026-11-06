
from pathlib import Path

import cv2
import torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
SAM2_DIR = PROJECT_DIR / "sam2"

IMAGE_PATH = PROJECT_DIR / "Test" / "test_image.jpg"
CHECKPOINT_PATH = SAM2_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# Select CPU for inference
DEVICE = "cpu"


# Load the image
image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# Build the SAM 2 model
model = build_sam2(
    MODEL_CONFIG,
    str(CHECKPOINT_PATH),
    device=DEVICE
)

# Create the image predictor
predictor = SAM2ImagePredictor(model)


# Prepare the image for prediction
predictor.set_image(image_rgb)

input_point = np.array([[318.0, 514.5]])
input_label = np.array([1])


print("SAM 2 model loaded successfully!")
print("Image loaded successfully!")
print("Image dimensions:", image_rgb.shape)
print("Device:", DEVICE)