import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch .nn. functional as F
from dataset import CustomDataset
from torch . utils . data import Dataset , DataLoader
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import StandardScaler
from MLP_model import MLP
from utils import r_squared

random_state = 59
np.random.seed(random_state)
torch.manual_seed(random_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/Users/thachha/Desktop/data/Auto_MPG_data.csv"
dataset = pd.read_csv(dataset_path)
print(dataset.info)



X = dataset.drop(columns ="MPG_linear_data").values
y = dataset["MPG_linear_data"].values

val_size = 0.2
test_size = 0.125
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

# Standardize the data
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_val = normalizer.transform(X_val)
X_test = normalizer.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                          )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
                        )

input_dims = X_train . shape [1]
output_dims = 1
hidden_dims = 64

model = MLP(input_dims=input_dims , hidden_dims=hidden_dims , output_dims= output_dims).to(device)

# Learning rate
lr = 1e-2  # Learning rate (0.01)

# Loss function
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

epochs = 100
train_losses = []
val_losses = []
train_r2 = []
val_r2 = []

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
        train_predict += outputs.tolist()
        train_target += y_samples.tolist()

        loss = criterion(outputs, y_samples)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_r2.append(r_squared(train_target, train_predict))

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_samples, y_samples in val_loader:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device)

            outputs = model(X_samples)
            val_predict += outputs.tolist()
            val_target += y_samples.tolist()

            loss = criterion(outputs, y_samples)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_r2.append(r_squared(val_target, val_predict))

    # Print progress
    print(
        f"\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}"
        )

# Set the model to evaluation mode
model.eval()

# Perform evaluation without computing gradients
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    y_hat = model(X_test)
    # Compute the R-squared value for the test set
    test_set_r2 = r_squared(y_test.tolist(), y_hat.tolist())

# Print the evaluation results
print('Evaluation on test set:')
print(f'R2: {test_set_r2:.3f}')

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')

plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.grid(True)
plt.show()
