# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .invariance import build as build_invariance

def build_dataset(image_set, args):
    if args.dataset_file == 'invariance':
        # return two things: data and labels
        return build_invariance(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
