# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.panoptic_eval import PanopticEvaluator
import random
from itertools import permutations
from collections import Counter

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_old(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


def train_invar(model, dataloader, eval_dataloader, count_accuracy, criterion, optimizer, device, args):
    param_file = args.save_path
    # if the saved parameters exist, load it
    if os.path.isfile(param_file):
        print("Loading saved parameters...")
        model.load_state_dict(torch.load(param_file))
    model.train()
    if args.early_stop:
        iteration = 1
    else:
        iteration = args.num_iterations
    d_model = args.d_model
    var_num = args.max_var_num
    # convert the string of args.enable_perm to int
    enable_perm = int(args.enable_perm)
    if enable_perm > 0:
        permute_num = args.perm_num
    else:
        permute_num = 1

    reference_perm = list(range(0, var_num))

    print_loss = 1
    print_outputs = 0
    print_weights = 0            
    print("{:<10} {:<10} {:<10}".format('eq', 'op', 'total'))  # printing the headers
    print("-"*30)  # print line for separation
    # print the length of the batches
    print("len(dataloader): ", len(dataloader) )
    for i in range(iteration):
    #for i in range(2):
        print("Iteration: ", i)
        permutations = []
        #if permute_num == 1:
        #    permutations.append(reference_perm)
        #    #permutations.append([2, 3, 4, 1, 0])
        #else:
        while len(permutations) < permute_num:
            perm = random.sample(reference_perm, len(reference_perm))
            permutations.append(perm)   
        #batch_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            if args.early_stop and batch_idx == args.stop_batch_num:
                break
            loss_list = []
            for perm_idx in range(permute_num):
                #print("perm_idx: ", perm_idx)
                #dim_size = max_var_num
                perm_list = permutations[perm_idx]
                print("batch: ", batch_idx, " perm: ", perm_idx, " perm_list: ", perm_list)
                #if perm_idx == 0:
                #    batch_idx += 1
                inputs, targets, masks = batch
                local_batch_size = inputs.shape[0]
                #print("batch_size: ", local_batch_size)
                # permute the inputs, masks and targets
                inputs, masks, targets = perm_data(inputs, masks, targets, perm_list, d_model)    
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, masks)
                outputs = outputs.to('cpu')
                #loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                losses = criterion(args, outputs, targets)
                # print loss and iteration numbers
                if(print_loss == 1):
                    print("Loss: ", losses['loss'].item())
                if print_outputs == 1:
                    print("outputs: ", outputs)
                # print the weights
                if print_weights == 1:
                    print("weights: ", model.fc.weight)
                # stop if loss is nan
                #if torch.isnan(outputs['eq']).any() or torch.isnan(outputs['op']).any():        
                #    print("Found NaN at index")
                #    return
                total_loss = sum(loss for loss in losses.values())
                # print the all kinds of losses
                #print("loss: ", loss)
                # print loss_eq, loss_op and total_loss in a form
                #print("{:<10.2f} {:<10.2f} {:<10.2f}".format(loss['loss_eq'].item(), loss['loss_op'].item(), total_loss.item()))
                #print("{:<10.2f}".format(losses['loss'].item()))
                #print("Eq loss: ", loss['loss_eq'].item())
                #print("Op loss: ", loss['loss_op'].item())
                #print("Total loss: ", total_loss.item())
                loss_list.append(total_loss.item())
                total_loss.backward()
                optimizer.step()

            average_loss = sum(loss_list) / len(loss_list)
            print("Average loss: ", average_loss)
        if i % 10 == 0:
        #if i % 1 == 0:
            #evaluate_max_degree(args, model, eval_dataloader, count_accuracy, device, False)
        # save the parameters
        torch.save(model.state_dict(), param_file)


def evaluate_op_eq(model, dataloader, count_accuracy, device):
    model.eval()  # Put the model in evaluation mode
    all_eq_accuracy = []
    all_op_accuracy = []

    # We don't need to update the model parameters, so we use torch.no_grad() 
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, masks = batch
            inputs = inputs.to(device)

            outputs = model(inputs, masks)
            eq_accuracy, op_accuracy = count_accuracy(outputs, targets)
            all_eq_accuracy.append(eq_accuracy)
            all_op_accuracy.append(op_accuracy)
            # print the two accuracies
            print("eq_accuracy: ", eq_accuracy)
            print("op_accuracy: ", op_accuracy)
    
    avg_eq_accuracy = sum(all_eq_accuracy) / len(all_eq_accuracy)
    avg_op_accuracy = sum(all_op_accuracy) / len(all_op_accuracy)
    print("avg_eq_accuracy: ", avg_eq_accuracy)
    print("avg_op_accuracy: ", avg_op_accuracy)


def merge_wrong_positions(all_wrong_positions, wrong_positions):
    all_wrong_positions.append(wrong_positions)


def merge_wrong_values(all_wrong_values, wrong_values):
    for key in wrong_values.keys():
        if key in all_wrong_values.keys():
            all_wrong_values[key].append(wrong_values[key])
        else:
            all_wrong_values[key] = wrong_values[key]


def print_analysis_results(all_wrong_positions, all_wrong_values):
    # flatten all_wrong_positions
    all_wrong_positions = [item for sublist in all_wrong_positions for item in sublist]
    # apply mod 5 to all elements in all_wrong_positions
    all_wrong_positions = [x % 5 for x in all_wrong_positions]
    count = Counter(all_wrong_positions)
    print("Wrong positions: ")
    for number, frequency in sorted(count.items()):
        print(f"{number}: {'*' * frequency}")
    print("Wrong values: ")
    for key in all_wrong_values.keys():
        print("== key: ", key)
        all_wrong_value_list = [item for sublist in all_wrong_values[key] for item in (sublist if isinstance(sublist, list) else [sublist])]
        count = Counter(all_wrong_value_list)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")


def evaluate_max_degree(args, model, dataloader, count_accuracy, device, verbose=False):
    model.eval()  # Put the model in evaluation mode
    all_degree_accuracy = []
    all_wrong_positions = []
    all_wrong_values = {}
    # We don't need to update the model parameters, so we use torch.no_grad()
    idx = 0
    var_num = args.max_var_num
    reference_perm = list(range(0, var_num))

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, masks = batch
            perm_list = random.sample(reference_perm, len(reference_perm))
            if args.enable_perm > 0:
                inputs, masks, targets = perm_data(inputs, masks, targets, perm_list, args.d_model)
            inputs = inputs.to(device)
            outputs = model(inputs, masks)
            outputs = outputs.to('cpu')
            """output is of shape (batch_size, n_classes)"""        
            outputs_flatten = outputs.view(-1, outputs.shape[-1])
            #pred = outputs_flatten.argmax(dim=1, keepdim=True)
            pred = (torch.sign(outputs_flatten) + 1) / 2
            # flatten pred
            pred = pred.view(-1)
            print_result = idx < 10
            correct_num, wrong_positions, wrong_values = count_accuracy(args, pred, targets, print_result)
            correct_ratio = correct_num / pred.shape[0]
            idx = idx+1
            merge_wrong_positions(all_wrong_positions, wrong_positions)
            merge_wrong_values(all_wrong_values, wrong_values)
            all_degree_accuracy.append(correct_ratio)
            # print the two accuracies
            if verbose:
                print("correct_degree_ratio: ", correct_ratio)
    
    avg_degree_accuracy = sum(all_degree_accuracy) / len(all_degree_accuracy)
    print("avg_degree_correct_ratio: ", avg_degree_accuracy)
    if verbose:
        print_analysis_results(all_wrong_positions, all_wrong_values)


def perm_data(inputs, masks, targets, perm_list, d_model):
    perm = torch.tensor(perm_list)
    inputs = inputs[:, perm]
    for target in targets:
        degree_list = target['max_degree']
        # pad degree_list with 0s
        degree_list += [0] * (d_model - len(degree_list))
        rearranged_list = [degree_list[i] for i in perm_list]
        target['max_degree'] = rearranged_list
    masks = masks[:, perm]
    return inputs, masks, targets