import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CURR_DIR = Path(__file__).parent


class BoxDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Construct full image path
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        # Convert one-hot encoding to a single label:
        if row[" EMPTY"] == 1:
            label = 0
        elif row[" FULL"] == 1:
            label = 1
        elif row[" PARTIAL"] == 1:
            label = 2
        else:
            raise ValueError("Invalid label in CSV")

        if self.transform:
            image = self.transform(image)

        return image, label


# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 16 x 112 x 112
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 32 x 56 x 56
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 64 x 28 x 28
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # Instantiate the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Define data transforms (include augmentation for training)
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Assume BoxDataset is defined as before
    # Create DataLoaders for training and validation
    train_dataset = BoxDataset(
        csv_file=CURR_DIR.parent
        / "data"
        / "box_classification"
        / "train"
        / "_classes.csv",
        img_dir=CURR_DIR.parent / "data" / "box_classification" / "train",
        transform=train_transforms,
    )
    val_dataset = BoxDataset(
        csv_file=CURR_DIR.parent
        / "data"
        / "box_classification"
        / "valid"
        / "_classes.csv",
        img_dir=CURR_DIR.parent / "data" / "box_classification" / "valid",
        transform=val_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        )

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_running_corrects.double() / len(val_dataset)
        print(f"Validation - Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), CURR_DIR.parent / "models" / "model.pth")

    # Ensure the model is in evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create the test dataset instance using your CSV file and image directory
    test_dataset = BoxDataset(
        csv_file=CURR_DIR.parent
        / "data"
        / "box_classification"
        / "test"
        / "_classes.csv",
        img_dir=CURR_DIR.parent / "data" / "box_classification" / "test",
        transform=val_transforms,
    )

    # Create the DataLoader for the test dataset.
    # Batch size is often set to 1 for visualization purposes.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Lists to collect all true labels and predictions
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Generate classification report
    # Ensure that the target names match your class mapping
    target_names = ["empty", "full", "partial"]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("Classification Report:")
    print(report)
