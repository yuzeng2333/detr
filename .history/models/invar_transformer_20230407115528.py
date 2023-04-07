# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torch
import torch.nn as nn
from torch.nn import Transformer

class InvarTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

        # Transformer model
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)

        # Query embeddings for the decoder
        self.query_embed = nn.Embedding(num_queries, d_model)

    def forward(self, src, src_mask):
        # Input shape: (batch_size, seq_length, d_model)
        bs, seq_length, _ = src.shape

        # Positional encoding is not needed, so set to 0
        pos_encoding = torch.zeros(src.shape, device=src.device)

        # Add positional encoding to the input
        src_with_pos = src + pos_encoding

        # Transpose the input to match the transformer's expected input shape (seq_length, batch_size, d_model)
        src_with_pos = src_with_pos.transpose(0, 1)

        # Generate initial target (query) embeddings
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # Pass the input and target through the transformer
        output = self.transformer(src_with_pos, tgt, src_key_padding_mask=src_mask)

        # Transpose the output back to the shape (batch_size, seq_length, d_model)
        output = output.transpose(0, 1)

        return output

# Example usage
#src = torch.randn(batch_size, 6, 512)  # Random input sentence (batch_size, seq_length, d_model)
#src_mask = torch.ones(batch_size, 6, 512).bool()  # Assuming no padding in this example


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return InvarTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
