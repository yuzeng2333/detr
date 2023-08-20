
import torch
from torch import nn

class DoubleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=6, num_classes=3):
        super(DoubleTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.transformer_horizontal_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            1
        )
        self.transformer_vertical_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            1
        )

        # TODO: fix this hard coding
        loop_iter = 512
        self.linear = nn.Linear(d_model, num_classes)


    def get_embedding(self, num):
        # num is a single number, map it to [sin(x), sin(2x), ..., sin(512x)]
        # the shape of output is (1, 512)
        output = torch.zeros(1, 512)
        for i in range(512):
            output[0][i] = torch.sin(num * (i + 1))
        return output

    def forward(self, src, masks):
        # the shape of src is (batch_size, variable_number, loop_iter) (200, 5, 512)
        # get the first two dimension size of src
        batch_size = src.shape[0]
        variable_number = src.shape[1]
        predict_tokens = torch.randn(batch_size, variable_number, 1)
        # concatenate the src and predict_tokens
        src = torch.cat((src, predict_tokens), dim=2)

        # map each number to its embedding
        # now the shape is (batch_size, variable_number, loop_iter, d_model) [200, 5, 512, 512]
        for i in range(batch_size):
            for j in range(variable_number):
                src[i][j] = self.get_embedding(src[i][j])

        # apply the transformer layer to the horizontal dimension (loop_iter)
        # the shape of output is (batch_size, variable_number, loop_iter, d_model)
        x_horizontal1 = self.transformer_horizontal_layer(src)
        x_horizontal2 = self.transformer_horizontal_layer(x_horizontal1)
        x_horizontal3 = self.transformer_horizontal_layer(x_horizontal2)

        # exchange the 2rd and 3rd dimension
        x_horizontal4 = x_horizontal3.transpose(1, 2)

        # apply the transformer layer to the vertical dimension (variable_number)
        # the shape of output is (batch_size, loop_iter, variable_number, d_model)
        x_vertical1 = self.transformer_vertical_layer(x_horizontal4)
        x_vertical2 = self.transformer_vertical_layer(x_vertical1)
        x_vertical3 = self.transformer_vertical_layer(x_vertical2)

        # do average to the dimension of loop_iter
        # the shape of output is (batch_size, variable_number, d_model)
        x_avg = torch.mean(x_vertical3, dim=1)
        output = self.linear(x_avg)
        return output
