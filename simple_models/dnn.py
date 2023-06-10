import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.var_num = 5
        self.sol_len = 512
        self.max_degree = 3
        self.fc1 = nn.Linear(self.var_num * self.sol_len, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        # Output is 5*3 = 15 because each of the 5 labels is one-hot encoded to length 3
        self.fc4 = nn.Linear(64, (self.max_degree+1)*self.var_num)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # Reshape to (-1, 5, 3) and apply softmax along the last dimension
        return F.softmax(x.view(-1, self.var_num, (self.max_degree+1)), dim=2)  

