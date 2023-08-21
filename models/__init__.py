# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .invar_detr import build_detr
from simple_models.build import build_dnn_model, build_transformer_model, build_pointnet, build_double_transformer


def build_model(args):
    if args.sel_model == 'detr':
        return build_detr(args)
    elif args.sel_model == 'dnn':
        return build_dnn_model(args)
    elif args.sel_model == 'transformer':
        return build_transformer_model(args)
    elif args.sel_model == 'pointnet':
        return build_pointnet(args)
    elif args.sel_model == 'double':
        return build_double_transformer(args)
    else:
        raise ValueError(f"Unrecognized model '{args.sel_model}'")
