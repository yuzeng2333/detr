# the model here is generated from GPT. It should be revised

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, nhead, nhid, nlayers, dropout):
        super(MyTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, input_dim)
        self.pos_encoder = PositionalEncoding(input_dim, dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout),
            num_layers=nlayers
        )

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1))
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = self.fc(x[-1, :, :])

        return x

# Initialize the model
model = MyTransformer(input_dim=512, num_classes=3, nhead=4, nhid=2048, nlayers=2, dropout=0.2)
