# kiwibot-case-study

# Report on Box Image Classification Code

This report explains the functionality of the provided Python script and describes how it addresses the problem of classifying box images into three categories: empty, full, and partial.

---

## Overview

The code implements an end-to-end deep learning pipeline using PyTorch. It performs the following tasks:

- **Data Loading and Preprocessing:** Reads image data and associated labels from CSV files.
- **Model Definition:** Implements a simple Convolutional Neural Network (CNN) for image classification.
- **Training and Validation:** Trains the model on a training set and validates its performance on a separate validation set.
- **Testing and Evaluation:** Evaluates the model on a test set using accuracy, a confusion matrix, and a detailed classification report.

---

## Key Components

### 1. Data Preparation and Custom Dataset

- **BoxDataset Class:**
  - **CSV Handling:** Reads a CSV file containing image filenames and one-hot encoded labels for the classes " EMPTY", " FULL", and " PARTIAL".
  - **Label Conversion:** Converts one-hot encoded labels to a single integer value:
    - `0` for empty
    - `1` for full
    - `2` for partial
  - **Image Loading:** Opens images using PIL and converts them to RGB.
  - **Transformations:** Applies transformations (data augmentation for training and normalization for validation/testing) if provided.

### 2. Model Architecture: SimpleCNN

- **CNN Structure:**
  - **Convolutional Blocks:** Three blocks, each consisting of:
    - A convolutional layer (kernel size 3, padding 1)
    - Batch normalization to stabilize training
    - ReLU activation for non-linearity
    - Max pooling for spatial downsampling
  - **Classifier:**
    - Two fully connected layers with dropout for regularization.
    - The final layer outputs logits for three classes.
- **Forward Pass:**
  - Processes the input image through the convolutional layers, flattens the output, and passes it through the fully connected classifier to produce the final predictions.

### 3. Training, Validation, and Testing Pipeline

- **Device Configuration:**
  - Uses a GPU if available; otherwise, defaults to CPU.

- **Loss Function and Optimizer:**
  - **Loss:** CrossEntropyLoss for multi-class classification.
  - **Optimizer:** Adam optimizer with a low learning rate and weight decay for regularization.

- **Data Transforms:**
  - **Training Transforms:**
    - Resizes images to 224×224.
    - Applies random horizontal flips, rotations, and color jitter for data augmentation.
    - Converts images to tensors and normalizes them using ImageNet statistics.
  - **Validation/Test Transforms:**
    - Resizes and normalizes images without augmentation to maintain consistency during evaluation.

- **DataLoaders:**
  - Creates loaders for training, validation, and testing datasets.
  - Uses a batch size of 32 for training/validation and 1 for testing (to facilitate individual predictions).

- **Training Loop:**
  - Runs for a set number of epochs (10).
  - For each batch:
    - Performs a forward pass, computes the loss, and backpropagates to update the model parameters.
    - Accumulates running loss and accuracy for performance monitoring.
  - Prints training loss and accuracy after each epoch.

- **Validation Phase:**
  - Evaluates model performance on the validation dataset after each training epoch.
  - Computes and prints validation loss and accuracy.

- **Model Saving:**
  - Saves the trained model's state dictionary to disk for future inference.

- **Testing and Evaluation:**
  - Evaluates the model on a separate test dataset.
  - Collects predictions and true labels to compute:
    - Overall accuracy using `accuracy_score`
    - Confusion matrix using `confusion_matrix`
    - Detailed classification report (precision, recall, F1-score) using `classification_report`

---

## Problem Addressed and Approach

### The Problem

The primary challenge was to develop an automated system capable of classifying images of boxes into three distinct categories (empty, full, and partial) by:
- Efficiently loading and preprocessing image data with associated CSV labels.
- Converting one-hot encoded labels to a single integer label for model compatibility.
- Designing and training a CNN capable of learning discriminative features from images.
- Evaluating the model using robust metrics to ensure high classification accuracy.

### The Approach

- **Custom Dataset Class:**  
  The `BoxDataset` class streamlines data loading and label conversion, ensuring that images and labels are processed correctly and efficiently.

- **CNN Architecture:**  
  The `SimpleCNN` model is a straightforward yet effective network architecture that leverages multiple convolutional layers to extract features and fully connected layers for classification.

- **Data Augmentation:**  
  Applying transformations during training (e.g., random flips, rotations, color adjustments) increases the robustness of the model by simulating a variety of conditions.

- **Structured Training Loop:**  
  A clear separation between training, validation, and testing phases allows for continuous monitoring of performance and prevents overfitting.

- **Comprehensive Evaluation:**  
  The use of accuracy, confusion matrix, and classification report provides a detailed understanding of the model’s performance across different classes.

---

## Running the code

### Visualizer UI

This module provides a Gradio-based interface to visually evaluate the trained box classification model. It:
- Loads a pre-trained `SimpleCNN` model from `models/model.pth`.
- Applies test-time image transformations and displays test images with true vs. predicted labels.
- Launches an interactive Gradio UI for quick inspection of model performance.

To run the visualizer, execute the following command:


```bash
conda create --name kiwibot-case-study python=3.10
```

Then activate the environment, and install the required packages:

```bash
conda activate kiwibot-case-study
pip install -r requirements.txt
```

Finally, run the visualizer script:

```bash
python gradio_ui.py
```

Then open the provided URL in a web browser to interact with the model.

```bash
http://127.0.0.1:7860
```

---

## Next Steps


- **Transfer Learning:** Consider leveraging pre-trained models (e.g., ResNet, VGG) to improve feature extraction and reduce training time.
- **Model Optimization:** Explore techniques such as quantization or pruning to deploy the model on edge devices with limited resources.
- **Integration and Deployment:** Develop a deployment strategy (e.g., a web service or mobile app) to integrate the model into a production environment.
