# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .invar_1detr import build


def build_model(args):
    return build(args)
