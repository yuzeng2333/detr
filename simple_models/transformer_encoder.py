import torch
from torch import nn

class MyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, num_classes=3):
        super(MyTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, masks):
        output = self.transformer_encoder(src)
        output = self.classifier(output)
        return output


# Initialize the model
model = MyTransformer()
