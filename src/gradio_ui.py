import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cv_model import BoxDataset, SimpleCNN

CURR_DIR = Path(__file__).parent

# Define device for computation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()

# Load previously trained model
state_dict = torch.load(CURR_DIR.parent / "models" / "model.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)


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


# Define transforms.
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Mapping from integer label to class name. Adjust if your mapping is different.
class_names = {0: "empty", 1: "full", 2: "partial"}

# Create the test dataset instance.
test_dataset = BoxDataset(
    csv_file=CURR_DIR.parent / "data" / "box_classification" / "test" / "_classes.csv",
    img_dir=CURR_DIR.parent / "data" / "box_classification" / "test",
    transform=test_transforms,
)

# Create the DataLoader for the test dataset.
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

# Number of images to display.
num_images_to_plot = 4


def generate_plot():
    # Set a larger figure size.
    plt.figure(figsize=(15, 6))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= num_images_to_plot:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            img = inputs.cpu().squeeze(0)
            true_label = class_names[labels.item()]
            pred_label = class_names[preds.item()]
            title = f"True: {true_label}\nPred: {pred_label}"

            plt.subplot(1, num_images_to_plot, i + 1)
            imshow(img, title=title)
    plt.tight_layout()
    return plt.gcf()


# Use gr.Blocks for a custom layout.
with gr.Blocks() as demo:
    predict_btn = gr.Button("Predict")  # Custom button labeled "Predict"
    output_plot = gr.Plot()  # Output component for the plot

    # Bind the button click to the generate_plot function.
    predict_btn.click(fn=generate_plot, inputs=[], outputs=output_plot)

demo.launch()
