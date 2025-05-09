import torch
import torch.nn.functional as F

# Define the input matrix (add batch and channel dimensions)
input_matrix = torch.tensor([
    [2, 4, 2],
    [1, 3, 2],
    [3, 2, 1],
    [0, 0, 1],
    [0, 0, 1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 4, 6)

print(input_matrix)
# Define the kernel matrix (add out_channels and in_channels dimensions)
kernel_matrix = torch.tensor([
    [1, 1],
    [1, 0],
    [0, 0]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
#Define bias
bias = torch.tensor([1.0], dtype=torch.float32)  # Size: (1,) for 1 output channel
print(bias)
# Perform 2D convolution
output = F.conv2d(input_matrix, kernel_matrix, bias=bias)
output= F.max_pool2d(output, kernel_size=(1,2))

# Print the output
print(output)
