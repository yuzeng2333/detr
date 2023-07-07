from .dnn import DNN
from .accuracy import count_accuracy
from .criterion import DNN_CROSS_ENTROPY
from .transformer_encoder import MyTransformer
from .transformer_512 import TransformerV2
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dnn_model(args):
    model = DNN(args.max_var_num)
    #criterion = DNN_CROSS_ENTROPY(args.max_var_num)
    criterion = DNN_CROSS_ENTROPY()
    return model, criterion, count_accuracy

def build_transformer_model(args):
    model = TransformerV2()
    #model = MyTransformer()
    #criterion = DNN_CROSS_ENTROPY(args.max_var_num)
    criterion = DNN_CROSS_ENTROPY()
    return model, criterion, count_accuracy