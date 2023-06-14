from .dnn import DNN
from .accuracy import count_accuracy
from .criterion import DNN_CROSS_ENTROPY
from .transformer_encoder import MyTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dnn_model(args):
    model = DNN(args.max_var_num)
    criterion = DNN_CROSS_ENTROPY()
    return model, criterion, count_accuracy

def build_transformer_model(args):
    model = MyTransformer()
    criterion = DNN_CROSS_ENTROPY()
    return model, criterion, count_accuracy