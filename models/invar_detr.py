# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .invar_matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .invar_transformer import build_transformer
from datasets.invar_spec import op_idx
from datasets.eq_spec import eq_idx


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, num_eq=2, num_classes=0, num_items=10, num_op=6, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.op_num = len(op_idx)
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.eq_embed = nn.Linear(hidden_dim, num_eq + 1)
        self.op_embed = MLP(hidden_dim, hidden_dim, self.op_num, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, data, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # pos is a tensor with the same shape as data, but is all zero

        hs = self.transformer(data, mask)

        outputs_eq = self.eq_embed(hs)
        outputs_op = self.op_embed(hs).sigmoid()
        out = {'eq': outputs_eq, 'op': outputs_op}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_eq, outputs_op)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def convert_eq_to_idx(self, eq):
        if eq == 'none':
            one_hot = torch.zeros(len(eq_idx)+1)
            one_hot[-1] = 1
            return one_hot
        assert eq in eq_idx
        one_hot = torch.zeros(len(eq_idx)+1)
        one_hot[eq_idx[eq]] = 1
        return one_hot

    def convert_op_to_idx(self, op_list):
        one_hot = torch.zeros(len(op_idx))
        for op in op_list:
            # assert the op should be inside op_idx
            assert op in op_idx
            one_hot[op_idx[op]] = 1
        return one_hot

    def get_eq_outputs_and_labels(self, outputs, targets, indices):
        assert 'eq' in outputs
        outputs_eq = outputs['eq'] # [batch_size x num_queries, (num_eq + 1)] num_eq == 2
        # num_queries is the max number of equations/inequalities to be detected for one data (# iter x # vars)
        # num_eq is the number of eq types. There are only two types of eqs: eq and ineq

        # idx has two parts, the first part is the batch index, the second part is the index of the query
        idx = self._get_src_permutation_idx(indices)
        target_classes_o_list = []
        for t, (_, J) in zip(targets, indices):
            # convert J from tensor to a list
            J = J.tolist()
            for j in J:
                target_classes_o_i = t["eq"][j]
                # convert to one-hot
                num_label = torch.tensor(eq_idx[target_classes_o_i])
                target_classes_o_list.append(num_label)
        # convert the list to a tensor
        # ground truth labels for each matched object
        target_classes_o = torch.stack(target_classes_o_list)

        # initialize the target classes with the no-object class
        sizes = list(outputs_eq.shape[:2])
        target_classes = torch.full(sizes, self.num_classes,
                                    dtype=torch.int64, device=outputs_eq.device)
        # set the last dimension as the one-hot encodings of the non-empty classes
        # transform idx: the inner dimension should be pairs from the first and the second dimension
        # of the idx
        pos = torch.stack(idx, dim=1)
        # assign target_classes_o to the corresponding positions in target_classes
        # use zip to iterate each pos and target_classes_o
        for p, t in zip(pos, target_classes_o):
            # convert t to the same type as target_classes
            t = t.type_as(target_classes)
            # convert p to a tuple
            p = tuple(p.tolist())
            target_classes[p[0], p[1]] = t

        # flatten the logits
        # shape of outputs_eq: [batch_size x num_queries, (num_eq + 1)], 
        # -- each row is the probability of each eq type (including the no-object class)
        # shape of target_classes: [batch_size x num_queries]
        # -- each row is the ground truth eq type for each query: 0, 1, 2 (2 is the no-object class)
        outputs_eq = outputs_eq.flatten(0, 1)
        # flatten the target classes
        target_classes = target_classes.flatten(0, 1)
        return outputs_eq, target_classes


    # uses cross_entropy loss for eqs
    def loss_eq(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        outputs_eq, target_classes = self.get_eq_outputs_and_labels(outputs, targets, indices)
        loss_ce = F.cross_entropy(outputs_eq, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['eq']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["eq"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def get_op_outputs_and_labels(self, outputs, targets, indices):
        #assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        # shape of src_boxes: [true_total_eq_num, len(op_idx)] = 4 x 6
        # true_total_eq_num is the number of eq/ineq from the labels
        selected_output_ops = outputs['op'][idx]
        # target_boxes should have the same shape as src_boxes
        #target_boxes = torch.cat([t['op'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        label_op_list = []
        for t, (_, i) in zip(targets, indices):
            # i is a 1-d tensor
            i = i.tolist()
            for j in i:
                op_list = t['op'][j]
                encoding = self.convert_op_to_idx(op_list)
                label_op_list.append(encoding)
        label_ops = torch.stack(label_op_list)
        return selected_output_ops, label_ops

    # use l1 loss for ops. But this is not a good idea, since the ops are not categorical
    def loss_op(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        selected_output_ops, label_ops = self.get_op_outputs_and_labels(outputs, targets, indices)
        BCE_loss = torch.nn.BCEWithLogitsLoss()
        loss_bbox = BCE_loss(selected_output_ops, label_ops)

        losses = {}
        losses['loss_op'] = loss_bbox.sum() / num_boxes
        return losses

    
    #def loss_masks(self, outputs, targets, indices, num_boxes):
    #    """Compute the losses related to the masks: the focal loss and the dice loss.
    #       targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #    """
    #    assert "pred_masks" in outputs

    #    src_idx = self._get_src_permutation_idx(indices)
    #    tgt_idx = self._get_tgt_permutation_idx(indices)
    #    src_masks = outputs["pred_masks"]
    #    src_masks = src_masks[src_idx]
    #    masks = [t["masks"] for t in targets]
    #    # TODO use valid to mask invalid areas due to padding in loss
    #    target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #    target_masks = target_masks.to(src_masks)
    #    target_masks = target_masks[tgt_idx]

    #    # upsample predictions to the target size
    #    src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
    #                            mode="bilinear", align_corners=False)
    #    src_masks = src_masks[:, 0].flatten(1)

    #    target_masks = target_masks.flatten(1)
    #    target_masks = target_masks.view(src_masks.shape)
    #    losses = {
    #        "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
    #        "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
    #    }
    #    return losses
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'eq': self.loss_eq,
            'cardinality': self.loss_cardinality,
            'op': self.loss_op
            #'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["eq"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        #if 'aux_outputs' in outputs:
        #    for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #        indices = self.matcher(aux_outputs, targets)
        #        for loss in self.losses:
        #            if loss == 'masks':
        #                # Intermediate masks losses are too costly to compute, we ignore them.
        #                continue
        #            kwargs = {}
        #            if loss == 'labels':
        #                # Logging is enabled only for the last layer
        #                kwargs = {'log': False}
        #            l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
        #            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #            losses.update(l_dict)

        return losses


# this class uses many functions in SetCriterion
class CountAccuracy(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.set_criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef, losses)

    def get_eq_accuracy(self, outputs, targets):
        # shape of output_eq: [batch_size x num_eq, num_classes+1]
        # shape of label_eq: [batch_size x num_eq, 1]
        output_eq, label_eq = self.set_criterion.get_eq_outputs_and_labels(outputs, targets)
        # get the index of the max probability for each row in output_eq
        pred_eq = output_eq.argmax(dim=-1)
        # compare the prediction with the label
        correct = pred_eq == label_eq
        # count the number of correct predictions
        num_correct = correct.sum().item()
        # count the number of equations
        num_eq = label_eq.shape[0]
        return num_correct, num_eq

    def get_op_accuracy(self, outputs, targets):
        # shape of output_op: [batch_size x num_op, 4]
        # shape of label_op: [batch_size x num_op, 1]
        output_op, label_op = self.set_criterion.get_op_outputs_and_labels(outputs, targets)
        # for each element of output_op, set it to 1 if it is greater than 0.5, otherwise set it to 0
        pred_op = (output_op > 0.5).float()
        # compare the prediction with the label
        correct = pred_op == label_op
        # count the number of correct predictions
        num_correct = correct.sum().item()
        # count the number of operations
        num_op = label_op.shape[0]
        return num_correct, num_op

    def forward(self, outputs, targets):
        num_correct_eq, num_eq = self.get_eq_accuracy(outputs, targets)
        num_correct_op, num_op = self.get_op_accuracy(outputs, targets)
        eq_accuracy = num_correct_eq / num_eq
        op_accuracy = num_correct_op / num_op
        return eq_accuracy, op_accuracy


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer()

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['eq', 'op']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    count_accuracy = CountAccuracy(num_classes, matcher=matcher, weight_dict=weight_dict,
                                   eos_coef=args.eos_coef, losses=losses)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, count_accuracy
