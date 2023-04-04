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
from torch.utils.data import Dataset


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract data from the DataFrame
        # Assuming columns: "input_data" and "labels"
        input_data = self.data.iloc[idx]["input_data"]
        labels = self.data.iloc[idx]["labels"]

        # Convert the data to the appropriate format (e.g., Tensors)
        # This depends on your specific data and requirements
        input_data = torch.tensor(input_data)
        labels = torch.tensor(labels)

        # Apply transforms if provided
        if self.transforms:
            input_data, labels = self.transforms(input_data, labels)

        return input_data, labels


def InvarianceDataset(data_folder, label_folder, filenames):
    # map from file name to the list of iloc
    MAX_VAR_NUM = 6
    batch_mask = torch.tensor([]) # shape: (num_files, MAX_VAR_NUM)
    var_names = {}
    batch_data = torch.tensor([]) # shape: (num_files, MAX_VAR_NUM, 512)
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
            data = data.drop(columns=["run_id"])
        # get the list of variable names
        var_names[filename] = list(data.columns)
        # column number
        col_num = len(data.columns)
        assert col_num <= MAX_VAR_NUM, f'number of columns in {filename} must be at most {MAX_VAR_NUM}'
        # convert the data to tensor
        data = torch.tensor(data.values)
        # transpose
        data = data.transpose(0, 1)
        # pad to MAX_VAR_NUM
        data = torch.nn.functional.pad(data, (0, MAX_VAR_NUM - col_num))
        # assert number of rows must be at least 512
        assert data.shape[1] >= 512, f'number of rows in {filename} must be at least 512'
        # get the first 512 rows
        data = data[:, :512]
        # append to batch_data
        batch_data = torch.cat((batch_data, data.unsqueeze(0)), dim=0)
        # get the mask
        mask = torch.zeros(MAX_VAR_NUM)
        mask[:col_num] = 1
        batch_mask = torch.cat((batch_mask, mask.unsqueeze(0)), dim=0)
        # begin working on the lables
        

                    

def build(image_set, args):
    root = Path(args.invar_path)
    assert root.exists(), f'provided invariance data path {root} does not exist'
    PATHS = {
        "train": (root / "data", root / "label"),
        "val": (root / "val_data", root / "val_label"),
    }

    data_folder, label_folder = PATHS[image_set]
    # currently only load the data for ps2
    dataset = InvarianceDataset(data_folder, label_folder, ["ps2", "ps3"])
    return dataset
