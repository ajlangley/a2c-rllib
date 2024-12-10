import torch

x = torch.arange(8)
x = torch.cat([x, torch.zeros(3)])
y = x.unfold(-1, 4, 1)
print(x)
print(y)
