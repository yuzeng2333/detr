# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import pandas as pd

def InvarianceDataset(data_folder, label_folder, filenames):
    for filename in filenames:
        # check if "filename.csv" exists in the data_folder
        if (data_folder / filename).exists():
            data = pd.read_csv(data_folder / filename)
        else: # add .npy later
            raise ValueError(f'file {filename} does not exist')
        # data preprocessing
        # remove the colume for "trace_idx" if it exists
        if "trace_idx" in data.columns:
            data = data.drop(columns=["trace_idx"])
        if "while_counter" in data.columns:
            data = data.drop(columns=["while_counter"])
        if "1" in data.columns:
            data = data.drop(columns=["1"])
        if "run_id" in data.columns:
            data = data.drop(columns=["1"])             

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
