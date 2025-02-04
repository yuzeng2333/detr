# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from numpy import int64

import torch
import torch.utils.data
import torchvision
import pandas as pd
from torch.utils.data import Dataset
import json
import numpy as np

def ReadInvarianceData(args, data_folder, label_folder, filenames):
    # if filenames is None, read all files in data_folder
    # find all the file names in data_folder
    loop_iter = args.loop_iter
    if not filenames:
        for file in data_folder.iterdir():
            if file.is_file():
                # remove the suffix in file name
                filename = file.name
                filename_without_suffix = filename.split('.')[0]
                #print("add file: ", filename_without_suffix)
                filenames.append(filename_without_suffix)
    # map from file name to the list of iloc
    MAX_VAR_NUM = args.max_var_num
    batch_mask = torch.tensor([]) # shape: (num_files, MAX_VAR_NUM)
    var_names = {}
    batch_data = [] # shape: (num_files, MAX_VAR_NUM, loop_iter )
    batch_label = []
    for filename in filenames:
        # convert filename to int
        filename_int = int(filename)
        if args.train_num > 0 and filename_int > args.train_num:
            continue
        # check if "filename.csv" exists in the data_folder
        full_filename = filename + ".csv"
        if (data_folder / full_filename).exists():
            data = pd.read_csv(data_folder / full_filename)
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
        if data.values.dtype == np.object_:
            data_values = np.array([list(map(int, item[0].split())) for item in data.values])
        else:
            data_values = data.values
        data = torch.tensor(data_values)
        # transpose
        #print(data.shape)
        data = data.transpose(0, 1)
        data = data.float()
        # normalization for each row: substract by mean, and divide by std
        data = data - data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        std_nonzero = torch.where(std != 0, std, torch.ones_like(std))
        data = data / std_nonzero
        # pad to MAX_VAR_NUM
        data = data.transpose(0, 1)
        data = torch.nn.functional.pad(data, (0, MAX_VAR_NUM - col_num))
        data = data.transpose(0, 1)
        # assert number of rows must be at least loop_iter
        feature_size = data.shape[1]
        if feature_size >= loop_iter:
            # get the first loop_iter items
            data = data[:, :loop_iter]
        else:
            # pad to loop_iter items
            data = torch.nn.functional.pad(data, (0, loop_iter - data.shape[1]))
        # append to batch_data
        batch_data.append(data)
        # get the mask
        mask = torch.zeros(MAX_VAR_NUM, loop_iter)
        mask[:col_num, :feature_size] = 1
        batch_mask = torch.cat((batch_mask, mask.unsqueeze(0)), dim=0)
        # read the label
        json_filename = filename + ".json"
        #print("open file: ", json_filename)
        with open(label_folder / json_filename) as f:
            label = json.load(f)
        batch_label.append(label)
    return batch_data, batch_mask, batch_label
                    

class InvarianceDateSet(Dataset):
    def __init__(self, args, data_folder, label_folder, filenames, max_var_num):
        data, mask, label = ReadInvarianceData(args, data_folder, label_folder, filenames)
        self.data = data
        self.mask = mask
        self.label = label

    def __getitem__(self, idx):
        #sample = {'image': self.data[idx], 'landmarks': self.label[idx]}
        #return sample
        return self.data[idx], self.label[idx], self.mask[idx]

    def __len__(self):
        return len(self.label)


def build(image_set, args):
    root = Path(args.invar_path)
    max_var_num = args.max_var_num
    assert root.exists(), f'provided invariance data path {root} does not exist'
    PATHS = {
        "train": (root / "data", root / "label"),
        "val": (root / "val_data", root / "val_label"),
        "trial": (root / "trial_data", root / "trial_label"),
    }

    data_folder, label_folder = PATHS[image_set]
    # currently only load the data for ps2
    #file_names = ["ps2", "ps3", "ps4_1", "ps5_1", "ps6_1", "sqrt1_1"]
    file_names = []
    dataset = InvarianceDateSet(args, data_folder, label_folder, file_names, args.d_model)
    return dataset
