import numpy as np

def compute_gram_matrix(feature_map: np.ndarray) -> np.ndarray:
    """
    Tính Gram Matrix từ feature map.
    Args :
        feature_map (np. ndarray ): Ma trận đặc trưng có kích thước (C, H, W).
    Returns :
        np. ndarray : Gram Matrix có kích thước (C, C).
    """

    C, H, W = feature_map.shape # Matrix input
    F = feature_map.reshape(C, H * W) # Convert matrix to (C, H*W)
    G = np.dot(F, F.T) # Compute Gram Matrix G = F @ F.T

    G /= (H * W) # Normalization based on number of pixels
    return G

    # Example of the initial data have size (3, 4, 4)
np.random.seed(42)
feature_map = np.random.rand(3, 4, 4)
#print(feature_map)
gram_matrix = compute_gram_matrix(feature_map)
#print("Gram Matrix:\n", gram_matrix)

import torch
feature_map_torch = torch.tensor(feature_map, dtype= torch.float32)
F_torch = feature_map_torch.view(3, -1)
gram_matrix_torch = torch.mm(F_torch, F_torch  .t()) / (4 * 4)
#print("\ nGram Matrix PyTorch :\n", gram_matrix_torch.numpy())

#Example 1
feature_map = np.ndarray([[[1 , 2], [3, 4]] , # Kênh 1
                         [[5 , 6], [7, 8]] , # Kênh 2
                         [[9 , 10] , [11 , 12]]]) # Kênh 3

gram_matrix = compute_gram_matrix(feature_map)
print("Gram Matrix:\n", gram_matrix)

def compute_similarity(gram1:np.ndarray , gram2:np.ndarray) -> float:
    """
    3 Tính độ tương đồng giữa hai Gram Matrix .
    4
    5 Args :
    6 gram1 (np. ndarray ): Gram Matrix ảnh 1.
    7 gram2 (np. ndarray ): Gram Matrix ảnh 2.
    8
    9 Returns :
    10 float : Độ tương đồng trong khoảng [0 ,1].
    11 """
    12  # Your code here #
    13
    14  # Feature Map của hai ảnh
    15
    feature_map1 = np.array([
        16[[1, 2], [3, 4]],
        17[[5, 6], [7, 8]],
        18[[9, 10], [11, 12]]
        19])
    20
    21
    feature_map2 = np.array([
        22[[2, 4], [6, 8]],
        23[[1, 3], [5, 7]],
        24[[0, 2], [4, 6]]
        25])
    26
    27  # Tính Gram Matrix của hai ảnh
    28
    gram1 = compute_gram_matrix(feature_map1)
    29
    gram2 = compute_gram_matrix(feature_map2)