from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class InvarNet(nn.Module):
    def __init__(self, args):
        super(InvarNet, self).__init__()

    def forward(self, x, masks):
