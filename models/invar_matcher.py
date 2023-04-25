# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from datasets.invar_spec import op_idx


class InvarHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_eq: float = 1, cost_op: float = 0.2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_eq = cost_eq
        self.cost_op = cost_op
        assert cost_eq != 0 or cost_op != 0, "all costs cant be 0"

    def convert_op_to_idx(self, op_query_list):
        one_hot = torch.zeros(len(op_idx))
        for op_list in op_query_list:
            for op in op_list:
                # assert the op should be inside op_idx
                assert op in op_idx
                one_hot[op_idx[op]] = 1
        return one_hot

    def convert_eq_to_idx(self, eq_query_list):
        # idx is a tensor with the same length as eq_query_list
        idx = torch.zeros(len(eq_query_list)) 
        i = 0
        for eq in eq_query_list:
            if eq == 'eq':
                idx[i] = 0
            elif eq == 'ineq':
                idx[i] = 1
            else:
                idx[i] = 2
            i += 1
        return idx
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["op"].shape[:2]

        """
        If we predict both the vars and corresponding ops, the output shape would be like:
        # the shape for predicted output vars: [batch_size, num_queries, num_items, (num_vars*num_vars-1*num_vars-2)]
        # the shape for predicted output op:   [batch_size, num_queries, num_items, 3, num_ops]
        
        But Even predicting the vars and corresponding ops seems quite challenging. 
        Let's start with sth simple. So for the first step, we only predicts the ops.
        Then the output shape would be like:
        # the shape for predicted output op:   [batch_size, num_queries, num_ops]
        # the shape for predicted eq/ineq:     [batch_size, num_queries, bool]
        """        

        # We flatten to compute the cost matrices in a batch
        out_op = outputs["op"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_op], num_op is the max types of op
        out_eq = outputs["eq"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 3]

        # Also concat the target labels and boxes
        #tgt_op = torch.cat([self.convert_op_to_idx(v["op"]) for v in targets]) # [all_num_ops]
        tgt_eq = torch.cat([self.convert_eq_to_idx(v["eq"]) for v in targets]) # [all_num_eq]
        # convert to long type
        tgt_eq = tgt_eq.long()
        # construct tgt_op explicitly
        num_eqs = tgt_eq.shape[0]
        num_total_queries = out_eq.shape[0]
        cost_op = torch.zeros(num_total_queries, num_eqs)
        col = 0
        for v in targets:
            op_query_list = v["op"]
            for op_list in op_query_list:
                # every op_list has a column in tgt_op
                for row in range(num_total_queries):
                    total_prob = 0
                    for op in op_list:
                        total_prob += out_op[row, op_idx[op]]
                    cost_op[row, col] = total_prob
                col += 1
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_eq = -out_eq[:, tgt_eq]
        
        cost_op = -cost_op

        # Final cost matrix
        # batch_size = 2 (2 images), num_queries = 5 
        # (upper limit of number of predicted equations in each image),
        # true number of equations in image 1 = 2, image 2 = 2, num_total_eqs is 4
        # shape of C: [batch_size * num_queries, num_total_eqs] = [10, 4]
        C = self.cost_eq * cost_eq + self.cost_op * cost_op
        C = C.view(bs, num_queries, -1).cpu() # shape: [batch_size, num_queries, num_total_eqs] = [2, 5, 4]

        sizes = [len(v["op"]) for v in targets] # [2, 2]: true number of equations in each image
        #indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        C_split = C.split(sizes, -1) # two tensors, each has shape [2, 5, 2]
        for i, c in enumerate(C_split): # i is indices [0, 1]; c is tensor of shape [2, 5, 2]
            # c has tensors for two images, so we need to index it
            # c[i] is the pedict for one image [5, 2]
            # col_indices are always [0, 1] since all the true labels are always selected
            # row_indices are the indices of the selected predictions in 5 queries
            row_indices, col_indices = linear_sum_assignment(c[i]) 
            current_indices = (row_indices, col_indices)
            indices.append(current_indices)
        ## the indices is a list of tuple, each tuple contains two lists, one for row indices, one for col indices
        ## the row list at index i corresponds to the col list at index i
        ## row[i] = j means the i-th query is matched to the j-th target
        #ret = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        ret = []
        for i, j in indices:
            i_tensor = torch.as_tensor(i, dtype=torch.int64)
            j_tensor = torch.as_tensor(j, dtype=torch.int64)
            ret.append((i_tensor, j_tensor))
        return ret

def build_matcher(args):
    return InvarHungarianMatcher()
