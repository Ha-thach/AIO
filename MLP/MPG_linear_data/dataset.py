from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset with input features (X) and labels (y).
        """
        self.X = X
        self.y = y

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label based on the index.
        """
        return self.X[idx], self.y[idx]
