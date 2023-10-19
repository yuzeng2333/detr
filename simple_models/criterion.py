import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import get_degrees

class DNN_CROSS_ENTROPY(nn.Module):
    def __init__(self, args):
        super(DNN_CROSS_ENTROPY, self).__init__()
        self.d_model = args.d_model

    def forward(self, args, outputs, targets):
        # flatten the first dimension of outputs
        if args.device[0:4] != 'cuda':
            print ("Warning: the device str does not contain 'cuda'")
        outputs = outputs.view(-1, outputs.shape[-1])
        var_num = args.max_var_num
        d_model = self.d_model
        degrees = []
        for target in targets:
            single_degree = target['max_degree']
            length = len(single_degree)
            if length < var_num:
                # pad the list with 0
                single_degree = single_degree + [0] * (var_num - length)
            degrees.append(single_degree)
        # flatten the first dimension of degrees
        degrees = torch.tensor(degrees, device=args.device).view(-1)
        weights = torch.tensor([1.0, 2.0, 4.0], device=args.device)
        loss= F.cross_entropy(outputs, degrees, weights)
        # return a dictionary
        return {'loss': loss}


# in this loss, the operator types (2-degree, 1-degree, etc.)
class OP_TYPE_LOSS(nn.Module):
    def __init__(self):
        super(OP_TYPE_LOSS, self).__init__()

    def forward(self, args, outputs, targets):
        # flatten the first dimension of outputs
        outputs = outputs.view(-1, outputs.shape[-1])
        #max_var_num = self.max_var_num
        #FIXME: hard code the d_model
        d_model = 5
        degrees = get_degrees(targets, d_model)
        # convert list to tensor
        degrees = torch.tensor(degrees)
        degrees = degrees.to(args.device)
        outputs = outputs.to(args.device)
        criterion = nn.BCEWithLogitsLoss()
        loss= criterion(outputs, degrees)
        # return a dictionary
        return {'loss': loss}