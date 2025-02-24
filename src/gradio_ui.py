import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from cv_model import BoxDataset

CURR_DIR = Path(__file__).parent

# Define device for computation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For this example, we'll assume Option 1:
model = torch.load("path/to/model.pth", map_location=device)

model.eval()  # Disable dropout, etc.
model = model.to(device)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Define a helper function to unnormalize and display an image.
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Undo the normalization (assuming the following normalization in your transforms)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")


# Mapping from integer label to class name. Adjust if your mapping is different.
class_names = {0: "empty", 1: "full", 2: "partial"}

# Create the test dataset instance using your CSV file and image directory
test_dataset = BoxDataset(
    csv_file=CURR_DIR.parent / "data" / "box_classification" / "test" / "_classes.csv",
    img_dir=CURR_DIR.parent / "data" / "box_classification" / "test",
    transform=val_transforms,
)

# Create the DataLoader for the test dataset.
# Batch size is often set to 1 for visualization purposes.
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

# Number of images to display
num_images_to_plot = 4


def generate_plot():
    # Create a new figure for each call.
    plt.figure(figsize=(15, 4))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= num_images_to_plot:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions.
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Move the input back to CPU for visualization.
            img = inputs.cpu().squeeze(0)  # Remove the batch dimension.

            # Prepare title with true and predicted labels.
            true_label = class_names[labels.item()]
            pred_label = class_names[preds.item()]
            title = f"True: {true_label}\nPred: {pred_label}"

            # Plot the image.
            plt.subplot(1, num_images_to_plot, i + 1)
            imshow(img, title=title)
    plt.tight_layout()
    return plt.gcf()  # Return the current figure


# Create a Gradio interface with no inputs and a Plot output.
iface = gr.Interface(fn=generate_plot, inputs=[], outputs=gr.Plot())
iface.launch()
