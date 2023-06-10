from .dnn import DNN
from .accuracy import count_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dnn_model(args):
    model = DNN(args.max_var_num)
    criterion = F.cross_entropy
    return model, criterion, count_accuracy