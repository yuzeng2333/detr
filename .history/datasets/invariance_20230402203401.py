# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision

def InvarianceDataset


def build(image_set, args):
    root = Path(args.invar_path)
    assert root.exists(), f'provided invariance data path {root} does not exist'
    PATHS = {
        "train": (root / "data", root / "label"),
        "val": (root / "val_data", root / "val_label"),
    }

    data_folder, label_folder = PATHS[image_set]
    # currently only load the data for ps2
    dataset = InvarianceDataset(data_folder, label_folder, "ps2")
    return dataset
