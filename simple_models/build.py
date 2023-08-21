from .dnn import DNN
from .accuracy import count_accuracy
from .criterion import DNN_CROSS_ENTROPY, OP_TYPE_LOSS
from .transformer_encoder import MyTransformer
from .transformer_512 import TransformerV2
from .pointnet import PointNetCls
from .double_transformer import DoubleTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dnn_model(args):
    model = DNN(args.d_model)
    #criterion = DNN_CROSS_ENTROPY(args.max_var_num)
    criterion = DNN_CROSS_ENTROPY(args.d_model)
    return model, criterion, count_accuracy

def build_transformer_model(args):
    model = TransformerV2(args.d_model)
    #model = MyTransformer()
    #criterion = DNN_CROSS_ENTROPY(args.max_var_num)
    criterion = DNN_CROSS_ENTROPY(args.d_model)
    return model, criterion, count_accuracy

def build_pointnet(args):
    model = PointNetCls(args)
    criterion = OP_TYPE_LOSS()
    return model, criterion, count_accuracy

def build_double_transformer(args):
    model = DoubleTransformer(args.d_model)
    criterion = DNN_CROSS_ENTROPY(args.d_model)
    return model, criterion, count_accuracy