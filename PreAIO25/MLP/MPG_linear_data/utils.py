import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def r_squared(y_true, y_pred):
    """
    Compute the R-squared (coefficient of determination) metric.

    Args:
    - y_true (list or array-like): Ground truth values.
    - y_pred (list or array-like): Predicted values.

    Returns:
    - torch.Tensor: The R-squared value.
    """
    # Convert input to tensors and move to the specified device
    y_true = torch.tensor(y_true, dtype=torch.float32).to(device)
    y_pred = torch.tensor(y_pred, dtype=torch.float32).to(device)

    # Compute the mean of the true values
    mean_true = torch.mean(y_true)

    # Total sum of squares
    ss_tot = torch.sum((y_true - mean_true) ** 2)

    # Residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # R-squared calculation
    r2 = 1 - (ss_res / ss_tot)
    return r2
