
from pathlib import Path
import torch

from sam2.build_sam import build_sam2_video_predictor

import cv2
import matplotlib.pyplot as plt

import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
SAM2_DIR = PROJECT_DIR / "sam2"

CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Select CPU
DEVICE = "cpu"

print("Loading SAM 2 video predictor...")

predictor = build_sam2_video_predictor(
    MODEL_CONFIG,
    str(CHECKPOINT),
    device=DEVICE
)

print("SAM 2 video predictor loaded successfully!")
print(f"Device: {DEVICE}")


# Video frame directory
VIDEO_FRAMES_DIR = PROJECT_DIR / "Test" / "SAM2_Test_Frames"

print("\nInitializing video frames...")

inference_state = predictor.init_state(
    video_path=str(VIDEO_FRAMES_DIR)
)

print("Video frames initialized successfully!")
print(f"Total frames: {inference_state['num_frames']}")
print(
    f"Video dimensions: "
    f"{inference_state['video_width']} x "
    f"{inference_state['video_height']}"
)


# Load the first video frame
first_frame_path = VIDEO_FRAMES_DIR / "00000.jpg"

first_frame = cv2.imread(str(first_frame_path))

if first_frame is None:
    raise RuntimeError("Could not load the first video frame.")

# Convert BGR to RGB for Matplotlib
first_frame_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)

# Display the frame and collect one click
plt.figure(figsize=(8, 8))
plt.imshow(first_frame_rgb)
plt.title("Click once on the subject")
plt.axis("on")

print("\nClick once on the subject in the image window.")

click = plt.ginput(1, timeout=0)

plt.close()

if not click:
    raise RuntimeError("No click was received.")

x, y = click[0]

print(f"Selected point: ({x:.1f}, {y:.1f})")

# Prepare the selected point
points = np.array(
    [[236.9, 243.5]],
    dtype=np.float32
)

labels = np.array([1], dtype=np.int32)

# Give the selected point to SAM 2
print("\nApplying the initial subject selection...")

frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=points,
    labels=labels
)

print("Subject selection applied successfully!")
print(f"Frame index: {frame_idx}")
print(f"Object IDs: {object_ids}")
print(f"Mask shape: {masks.shape}")


# Extract the predicted mask
mask = masks[0, 0].cpu().numpy()

# Display original frame and predicted mask
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(first_frame_rgb)
plt.title("Original Video Frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(first_frame_rgb)
plt.imshow(mask > 0, alpha=0.5, cmap="jet")
plt.title("SAM 2 Initial Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()


# Store the predicted masks for each frame
video_masks = {}

print("\nStarting video propagation...")

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    # Convert the predicted mask into a binary NumPy array
    mask = (
        (out_mask_logits[0] > 0.0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Store the mask using its frame index
    video_masks[out_frame_idx] = mask

    # Display progress every 10 frames
    if (out_frame_idx + 1) % 10 == 0:
        print(
            f"Processed {out_frame_idx + 1}/"
            f"{inference_state['num_frames']} frames"
        )

print("\nVideo propagation completed!")
print(f"Total processed frames: {len(video_masks)}")


# Frames to inspect
frames_to_check = [0, 15, 30, 45, 59]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, frame_idx in enumerate(frames_to_check):

    # Load the corresponding original frame
    frame_path = VIDEO_FRAMES_DIR / f"{frame_idx:05d}.jpg"
    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(f"Could not load frame {frame_idx}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Retrieve the predicted mask
    mask = video_masks[frame_idx]

    # Match image dimensions to the mask
    frame_rgb = cv2.resize(
        frame_rgb,
        (mask.shape[1], mask.shape[0])
    )

    # Display original frame
    axes[i].imshow(frame_rgb)

    # Overlay segmentation mask
    visible_mask = np.ma.masked_where(mask == 0, mask)

    axes[i].imshow(
        visible_mask,
        alpha=0.5,
        cmap="jet"
    )

    axes[i].set_title(f"Frame {frame_idx}")
    axes[i].axis("off")

# Hide the unused sixth subplot
axes[5].axis("off")

plt.tight_layout()
plt.show()