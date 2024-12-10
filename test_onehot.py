import torch
from torch.nn import functional as F

x = torch.tensor([[1, 2, 3], [2, 3, 0]])
print(F.one_hot(x, num_classes=4))
