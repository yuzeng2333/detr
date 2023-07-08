import torch
from torch import nn

class TransformerV2(nn.Module):
    def __init__(self, d_model=5, nhead=1, num_layers=6, num_classes=3):
        super(TransformerV2, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )

        self.classifier = nn.Linear(d_model, d_model * num_classes)

    def forward(self, src, masks):
        # exchange the 2rd and 3rd dimension
        src = src.transpose(1, 2)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.classifier(output)
        # reshape the output to (batch_size, d_model, num_classes)
        output = output.reshape(output.shape[0], self.d_model, self.num_classes)
        return output
