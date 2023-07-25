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

        # TODO: fix this hard coding
        length = 512
        self.linear1 = nn.Linear(d_model * length, length)
        self.linear2 = nn.Linear(length, length)
        self.linear3 = nn.Linear(length, d_model * num_classes)

    def forward(self, src, masks):
        # exchange the 2rd and 3rd dimension
        trans = src.transpose(1, 2)
        encoder_output = self.transformer_encoder(trans)
        # for each row of output, sort it separately
        trans2 = encoder_output.transpose(1, 2)
        sorted = torch.sort(trans2, dim=2, descending=True)[0]
        # flatten the last two dimensions
        sorted = sorted.reshape(sorted.shape[0], -1)
        linear1 = self.linear1(sorted)
        relu1 = nn.functional.relu(linear1)
        linear2 = self.linear2(relu1)
        relu2 = nn.functional.relu(linear2)
        linear3 = self.linear3(relu2)
        output = linear3
        # reshape the output to (batch_size, d_model, num_classes)
        output = output.reshape(output.shape[0], self.d_model, self.num_classes)
        return output
