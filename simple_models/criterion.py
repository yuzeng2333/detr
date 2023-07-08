import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_CROSS_ENTROPY(nn.Module):
    #def __init__(self, max_var_num):
    #    self.max_var_num = max_var_num

    def forward(self, outputs, targets):
        # flatten the first dimension of outputs
        outputs = outputs.view(-1, outputs.shape[-1])
        #max_var_num = self.max_var_num
        #FIXME: hard code the d_model
        d_model = 5
        degrees = []
        for target in targets:
            single_degree = target['max_degree']
            length = len(single_degree)
            if length < d_model:
                # pad the list with 0
                single_degree = single_degree + [0] * (d_model - length)
            degrees.append(single_degree)
        # flatten the first dimension of degrees
        degrees = torch.tensor(degrees).view(-1)
        weights = torch.tensor([0.4, 0.2, 0.4])
        loss= F.cross_entropy(outputs, degrees, weights)
        # return a dictionary
        return {'loss': loss}