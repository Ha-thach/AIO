import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = "/Users/thachha/Desktop/data/creditcard.csv"
df = pd.read_csv(dataset_path)
print(df)

dataset_arr = df.to_numpy()
X, y = dataset_arr[:, :-1].astype(np.float64), dataset_arr[:, -1].astype(np.uint8)

# Add bias
intercept = np.ones((X.shape[0], 1))
X_b = np.concatenate((intercept, X), axis=1)

# One-hot encoding
n_classes = np.unique(y, axis=0).shape[0]
n_samples = y.shape[0]

y_encoded = np.zeros((n_samples, n_classes))
y_encoded[np.arange(n_samples), y] = 1

# Train, validation, and test split
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X_b, y_encoded,
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

print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of validation samples: {X_val.shape[0]}')
print(f'Number of test samples: {X_test.shape[0]}')

# Normalize features
normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# Prediction function
def predict(X, theta):
    z = np.dot(X, theta)
    return softmax(z)

# Compute loss function
def compute_loss(y_hat, y):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-7), axis=1))  # Add epsilon for stability

# Compute gradient
def compute_gradient(X, y, y_hat):
    return np.dot(X.T, (y_hat - y)) / X.shape[0]

# Update parameters
def update_theta(theta, gradient, lr):
    return theta - lr * gradient

# Compute accuracy
def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta)
    return (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()

# Hyperparameters
lr = 0.01
epochs = 30
batch_size = 1024
n_features = X_train.shape[1]

# Initialize parameters
np.random.seed(random_state)
theta = np.random.uniform(size=(n_features, n_classes))

# Metrics tracking
train_accs = []
train_losses = []
val_accs = []
val_losses = []

# Training loop
for epoch in range(epochs):
    train_batch_losses = []
    train_batch_accs = []
    val_batch_losses = []
    val_batch_accs = []

    for i in range(0, X_train.shape[0], batch_size):
        X_i = X_train[i:i + batch_size]
        y_i = y_train[i:i + batch_size]

        y_hat = predict(X_i, theta)
        train_loss = compute_loss(y_hat, y_i)
        gradient = compute_gradient(X_i, y_i, y_hat)
        theta = update_theta(theta, gradient, lr)

        train_batch_losses.append(train_loss)
        train_acc = compute_accuracy(X_i, y_i, theta)
        train_batch_accs.append(train_acc)

    y_val_hat = predict(X_val, theta)
    val_loss = compute_loss(y_val_hat, y_val)
    val_acc = compute_accuracy(X_val, y_val, theta)

    train_losses.append(np.mean(train_batch_losses))
    val_losses.append(val_loss)
    train_accs.append(np.mean(train_batch_accs))
    val_accs.append(val_acc)

    print(f'EPOCH {epoch + 1}:	 Training loss: {train_losses[-1]:.3f}	 Validation loss: {val_losses[-1]:.3f}')
