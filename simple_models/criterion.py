import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_CROSS_ENTROPY(nn.module):
    def forward(self, outputs, targets):
        degree_labels = targets["max_degree"]
        return F.cross_entropy(outputs, degree_labels)