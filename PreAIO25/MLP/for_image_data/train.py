# 1.Imports libs
from dataset import ImageDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from PreAIO25.MLP.utils import compute_accuracy
from model import MLP

# 2.Random Seed Initialization & set device
random_state = 59
np.random.seed(random_state)
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 3.Load and Preprocess Dataset using class ImageDataset(load image data, preprocess, train-validation splitting and Py
train_dir = "/Users/thachha/Desktop/data/FER-2013/train"
test_dir = "/Users/thachha/Desktop/data/FER-2013/test"
classes = os.listdir(train_dir)
label2idx = {cls:idx for idx, cls in enumerate(classes)} #create dict để ánh xạ tên và chỉ số lớp
idx2label = {idx:cls for cls, idx in label2idx.items()}

batch_size = 256


# Create sub dataset using ImageDataset class (help to normalize, preprocess, ...) and load image data using  pytorch DataLoader utils.
train_dataset = ImageDataset(
    img_dir=train_dir,
    norm=True,
    label2idx=label2idx,
    split='train'
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Create validation dataset and DataLoader
val_dataset = ImageDataset(
    img_dir=train_dir,
    norm=True,
    label2idx=label2idx,
    split='val'
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Create test dataset and DataLoader
test_dataset = ImageDataset(
    img_dir=test_dir,
    norm=True,
    label2idx=label2idx,
    split='test'
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# # Take out 1 image and check its size after loading
# # Get one batch from the train_loader
# for images, labels in train_loader:
#     # Get the first image from the batch
#     sample_image = images[0]  # assuming images is a batch of shape [batch_size, channels, height, width]
#     print(f"Image size: {sample_image.shape}")
#     break  # exit after the first batch to avoid unnecessary looping #Image size: torch.Size([1, 224, 224])
# exit()
# Define the test image path
test_img_path = "/Users/thachha/Desktop/data/FER-2013/train/angry/Training_10118481.jpg"
# Read the image
img = cv2.imread(test_img_path)
#print(img.shape) (48, 48, 3)
# Define image dimensions
img_height, img_width = (224, 224)

# 8.Model
input_dims = img_height * img_width  # Calculate input dimensions based on image size
hidden_dims = 64  # Number of hidden dimensions
output_dims = len(classes)  # Number of output classes
lr = 1e-2  # Learning rate

# Instantiate the model and move it to the specified device
model = MLP(
    input_dims=input_dims,
    hidden_dims=hidden_dims,
    output_dims=output_dims
).to(device)

# 9.Define Optimizer and Loss

# Loss function
criterion = nn.CrossEntropyLoss()  # Phân loại đa lớp => Cross Entropy

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

# Training
epochs = 40
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(epochs):
    # Initialize variables for this epoch
    train_loss = 0.0
    train_target = []
    val_target = []
    train_predict = []
    val_predict = []

    # Training phase
    model.train()
    for X_samples, y_samples in train_loader:
        X_samples = X_samples.to(device)
        y_samples = y_samples.to(device)

        optimizer.zero_grad()
        outputs = model(X_samples)

        loss = criterion(outputs, y_samples)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_predict.append(outputs.detach().cpu())
        train_target.append(y_samples.cpu())

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    train_predict = torch.cat(train_predict)
    train_target = torch.cat(train_target)
    train_acc = compute_accuracy(train_predict, train_target)
    train_accs.append(train_acc)

    # Validation phase
    val_loss = 0.0
    val_target = []
    val_predict = []

    model.eval()
    with torch.no_grad():
        for X_samples, y_samples in val_loader:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device)

            outputs = model(X_samples)
            loss = criterion(outputs, y_samples)
            val_loss += loss.item()
            val_predict.append(outputs.cpu())
            val_target.append(y_samples.cpu())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_predict = torch.cat(val_predict)
    val_target = torch.cat(val_target)
    val_acc = compute_accuracy(val_predict, val_target)
    val_accs.append(val_acc)

    # Print progress
    print(
        f"\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}"
    )

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses, color='green')
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training Loss')

ax[0, 1].plot(val_losses, color='orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
ax[0, 1].set_title('Validation Loss')

ax[1, 0].plot(train_accs, color='green')
ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 0].set_title('Training Accuracy')

ax[1, 1].plot(val_accs, color='orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 1].set_title('Validation Accuracy')
plt.show(0)


# Evaluate the model on the test set
test_target = []
test_predict = []
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Disable gradient computation for efficiency
    for X_samples, y_samples in test_loader:
        # Move data to the appropriate device (e.g., GPU if available)
        X_samples = X_samples.to(device)
        y_samples = y_samples.to(device)

        # Make predictions using the model
        outputs = model(X_samples)

        # Store predictions and true targets for evaluation
        test_predict.append(outputs.cpu())
        test_target.append(y_samples.cpu())

# Concatenate predictions and targets
test_predict = torch.cat(test_predict)
test_target = torch.cat(test_target)

# Compute test accuracy
test_acc = compute_accuracy(test_predict, test_target)

# Print evaluation results
print('Evaluation on test set:')
print(f'Accuracy: {test_acc:.2f}')
