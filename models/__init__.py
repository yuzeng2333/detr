# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .invar_detr import build_detr
from simple_models.build import build_dnn_model, build_transformer_model


def build_model(args):
    if args.sel_model == 'detr':
        return build_detr(args)
    elif args.sel_model == 'dnn':
        return build_dnn_model(args)
    elif args.sel_model == 'transformer':
        return build_transformer_model(args)
    else:
        raise ValueError(f"Unrecognized model '{args.sel_model}'")
