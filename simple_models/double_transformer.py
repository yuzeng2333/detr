
import torch
from torch import nn

class DoubleTransformer(nn.Module):
    def __init__(self, args, nhead=1, num_layers=6, num_classes=3):
        super(DoubleTransformer, self).__init__()
        self.device = args.device
        # check if the device str contains 'cuda'
        if self.device[0:4] != 'cuda':
            print ("Warning: the device str does not contain 'cuda'")
        self.d_model = args.d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = args.batch_size
        self.max_var_num = args.max_var_num
        assert args.batch_size % args.gpu_num == 0
        self.pred_token_size = int(args.batch_size / args.gpu_num)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_token_size, args.max_var_num, 1, device=self.device))
        self.norm = nn.LayerNorm(self.d_model)
        self.transformer_horizontal_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, nhead)
            ,3
            #2
            #,norm = self.norm
        )
        self.transformer_vertical_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, nhead)
            ,3
            #2
            #,norm = self.norm
        )

        # TODO: fix this hard coding
        loop_iter = args.loop_iter
        self.linear = nn.Linear(self.d_model, num_classes)


    def get_embedding(self, num, d_model):
        # num is a single number, map it to [sin(x), sin(2x), ..., sin(512x)]
        # the shape of output is (1, d_model)
        output = torch.zeros(d_model, device=self.device) 
        for i in range(d_model):
            #output[i] = torch.sin(num * (i + 1))
            output[i] = num
        return output


    def forward(self, src, masks):
        use_pred_tokens = False
        # the shape of src is (batch_size, variable_number, loop_iter) (200, 5, 512)
        # get the first two dimension size of src
        batch_size = src.shape[0]
        variable_number = src.shape[1]
        loop_iter = src.shape[2]
        d_model = self.d_model
        if use_pred_tokens:
            assert batch_size == self.pred_token_size
        assert variable_number == self.max_var_num
        # concatenate the src and predict_tokens
        if use_pred_tokens:
            #print("src shape: ", src.shape)
            #print("pred_tokens shape: ", self.pred_tokens.shape)
            src = torch.cat((src, self.pred_tokens), dim=2)

        # map each number to its embedding
        # now the shape of src and src_extended is (batch_size, variable_number, extended_loop_iter, d_model) [200, 5, 513, 512]
        extended_loop_iter = loop_iter
        if use_pred_tokens:
            extended_loop_iter += 1
  
        if self.device[0:4] != 'cuda':
            print ("Warning: the device str does not contain 'cuda'")
        #multiplier = torch.arange(1, d_model+1, device=self.device)
        multiplier = torch.ones(d_model, device=self.device)
        src_extended = src.unsqueeze(3) * multiplier
        #src_extended = torch.sin(src_extended)
        # assert the last dimension size is d_model
        assert src_extended.shape[-1] == d_model

        # merge the first two dimensions
        # now the shape is (batch_size * variable_number, loop_iter, d_model)
        src_extended = src_extended.reshape(batch_size * variable_number, extended_loop_iter, d_model)
        # apply the transformer layer to the horizontal dimension (loop_iter)
        # the shape of output is (batch_size, variable_number, loop_iter, d_model)
        x_horizontal1 = self.transformer_horizontal_layer(src_extended)
        # reshape the output to (batch_size, variable_number, loop_iter, d_model)
        x_horizontal2 = x_horizontal1.reshape(batch_size, variable_number, extended_loop_iter, d_model)

        # exchange the 2rd and 3rd dimension
        x_horizontal3 = x_horizontal2.transpose(1, 2)

        # merge the first two dimensions
        # now the shape is (batch_size * loop_iter, variable_number, d_model)
        x_horizontal4 = x_horizontal3.reshape(batch_size * extended_loop_iter, variable_number, d_model)

        # apply the transformer layer to the vertical dimension (variable_number)
        # the shape of output is (batch_size, loop_iter, variable_number, d_model)
        #x_vertical1 = self.transformer_vertical_layer(x_horizontal4)
        x_vertical1 = x_horizontal4

        # reshape the output to (batch_size, loop_iter, variable_number, d_model)
        x_vertical1 = x_vertical1.reshape(batch_size, extended_loop_iter, variable_number, d_model)

        # do average to the dimension of loop_iter
        # the shape of output is (batch_size, variable_number, d_model)
        if use_pred_tokens == False:
            x_avg = torch.mean(x_vertical1, dim=1)
            output = self.linear(x_avg)
        else:
            pred_tokens = x_vertical1[:, -1, :, :]
            # the shape of pred_tokens is (batch_size, variable_number, d_model)
            output = self.linear(pred_tokens)
        return output