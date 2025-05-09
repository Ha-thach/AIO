import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset import CustomDataset
from MLP_model import MLP
from PreAIO25.MLP.utils import compute_accuracy

random_state = 59
np.random.seed(random_state)
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/Users/thachha/Desktop/data/NonLinear_data.npy"
dataset = np.load(dataset_path, allow_pickle=True).item()
# print(dataset)
X, y = dataset["X"], dataset["labels"]
print(X.shape, y.shape)  # (300, 2) (300,)

# Split dataset 7:2:1
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

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                          )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
                        )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False
                        )
input_dims = X_train.shape[1]
output_dims = torch.unique(y_train).shape[0]  # calculate how many classes
hidden_dims = 128

model = MLP(input_dims=input_dims, hidden_dims=hidden_dims, output_dims=output_dims).to(device)

# Learning rate
lr = 1e-2  # Learning rate (0.01)

# Loss function
criterion = nn.CrossEntropyLoss()  # Phân loại đa lớp => Cross Entropy

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

# Training
epochs = 300
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
plt.show()


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
