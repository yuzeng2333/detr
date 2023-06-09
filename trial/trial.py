import torch
import torch.nn.functional as F

t1 = torch.tensor([[1.0, 0.2, 0.3],
                   [0.1, 0.5, 1.0],
                   [1.0, 0.8, 0.7]])

t3 = torch.tensor([[1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0]])


t2 = torch.tensor([0, 2, 0])

loss1 = F.cross_entropy(t1, t2)
print("loss1: ", loss1)

loss2 = F.cross_entropy(t3, t2)
print("loss2: ", loss2)

