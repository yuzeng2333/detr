import torch
from .util import get_degrees


def analyze_result(pred, degrees):
    """analyze which positions get wrong, and which values get wrong"""
    pred = pred.view(-1)
    degrees = degrees.view(-1)
    result_list = pred.eq(degrees.view_as(pred))
    wrong_positions = []
    # wrong_values is a map from the correct values to the list of wrong values
    wrong_values = {}
    for idx, result in enumerate(result_list):
        if not result:
            wrong_positions.append(idx)
            correct_value = degrees[idx].item()
            wrong_value = pred[idx].item()
            if correct_value not in wrong_values:
                wrong_values[correct_value] = []
            wrong_values[correct_value].append(wrong_value)
    return wrong_positions, wrong_values


def count_accuracy(args, pred, targets, do_print):
    max_var_degree = args.max_var_degree
    degrees = get_degrees(targets, max_var_degree+1)
    # flatten the first dimension of degrees
    degrees = torch.tensor(degrees).view(-1)
    result_list = pred.eq(degrees.view_as(pred))
    correct_num = result_list.sum().item()
    # print the targets and the pred
    if do_print:
        print('targets:', degrees)
        print('pred   :', pred)
    wrong_positions, wrong_values = analyze_result(pred, degrees)
    return correct_num, wrong_positions, wrong_values