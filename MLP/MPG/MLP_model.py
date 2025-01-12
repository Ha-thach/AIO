import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        """
        Initialize the Multi-Layer Perceptron (MLP) with one hidden layer.

        Args:
        - input_dims (int): Number of input features.
        - hidden_dims (int): Number of neurons in the hidden layers.
        - output_dims (int): Number of output features.
        """
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = self.output(x)
        return out.squeeze(1)
