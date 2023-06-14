import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_CROSS_ENTROPY(nn.Module):
    def forward(self, outputs, targets):
        # flatten the first dimension of outputs
        outputs = outputs.view(-1, outputs.shape[-1])
        degrees = []
        for target in targets:
            degrees.append(target['max_degree'])
        # flatten the first dimension of degrees
        degrees = torch.tensor(degrees).view(-1)
        loss= F.cross_entropy(outputs, degrees)
        # return a dictionary
        return {'loss': loss}