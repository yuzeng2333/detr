import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_CROSS_ENTROPY(nn.Module):
    #def __init__(self, max_var_num):
    #    self.max_var_num = max_var_num

    def forward(self, args, outputs, targets):
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
        weights = torch.tensor([0.25, 0.25, 0.5])
        degrees = degrees.to(args.device)
        outputs = outputs.to(args.device)
        avg_output = torch.mean(outputs, dim=0)
        avg_output = avg_output.unsqueeze(0)
        weights = weights.to(args.device)
        loss= F.cross_entropy(outputs, degrees, weights) - avg_output[0][0] * 0.1
        # return a dictionary
        return {'loss': loss}