import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_CROSS_ENTROPY(nn.Module):
    def __init__(self, max_var_num):
        self.max_var_num = max_var_num

    def forward(self, outputs, targets):
        # flatten the first dimension of outputs
        outputs = outputs.view(-1, outputs.shape[-1])
        degrees = []
        for target in targets:
            degrees.append(target['max_degree'])
        # flatten the first dimension of degrees
        degrees = torch.tensor(degrees).view(-1)
        # if the length of degrees is less than max_var_num, pad it with 0
        if degrees.shape[0] < self.max_var_num:
            degrees = torch.nn.functional.pad(degrees, (0, self.max_var_num - degrees.shape[0]))
        weights = torch.tensor([0.4, 0.2, 0.4])
        loss= F.cross_entropy(outputs, degrees, weights)
        # return a dictionary
        return {'loss': loss}