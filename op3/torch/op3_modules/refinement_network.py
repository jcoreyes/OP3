import torch
import torch.utils.data
from op3.torch.pytorch_util import from_numpy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from op3.pythonplusplus import identity
from op3.torch import pytorch_util as ptu
import numpy as np
import pdb


###Maps name to a tuple (class type, lambda function defining model architecture)
Refinement_Args = dict(
    large_reg = ("reg",
        lambda repsize: dict(
        input_width = 64,
        input_height = 64,
        input_channels=17,
        paddings=[0, 0, 0],
        kernel_sizes=[5, 5, 5],
        n_channels=[32, 32, 32],
        strides=[1, 1, 1],
        hidden_sizes=[128],
        output_size=repsize, #repsize
        lstm_size=128,
        lstm_input_size=repsize*5 + 128, #repsize*5 + hidden_sizes[-1]
        added_fc_input_size=0,
        hidden_activation = nn.ELU())
    ),
    size_dependent_conv = ("reg",
            lambda repsize: dict(
            input_width = 64,
            input_height = 64,
            input_channels=17,
            paddings=[2, 2, 2],
            kernel_sizes=[5, 5, 5],
            n_channels=[32, 32, repsize],
            strides=[1, 1, 1],
            hidden_sizes=[128],
            output_size=repsize, #repsize
            lstm_size=128,
            lstm_input_size=repsize*5 + 128, #repsize*5 + hidden_sizes[-1]
            added_fc_input_size=0,
            hidden_activation = nn.ELU())
        ),
    large_sequence = ("sequence_iodine",
        lambda repsize, action_size: dict(
            input_width=64,
            input_height = 64,
            input_channels=17,
            paddings=[0, 0, 0, 0],
            kernel_sizes=[5, 5, 5, 5],
            n_channels=[64, 64, 64, 64],
            strides=[2, 2, 2, 2],
            hidden_sizes=[128, 128],
            output_size=repsize, #repsize
            lstm_size=256,
            lstm_input_size=repsize*5 + 128, #repsize*5 + hidden_sizes[-1]
            added_fc_input_size=action_size,
            hidden_activation = nn.ELU())
        ),
    size_dependent_conv_no_share=("no_share",
         lambda repsize, k: dict(
             input_width=64,
             input_height=64,
             input_channels=17,
             paddings=[2, 2, 2],
             kernel_sizes=[5, 5, 5],
             n_channels=[32, 32, repsize],
             strides=[1, 1, 1],
             hidden_sizes=[128],
             output_size=repsize,  #repsize
             lstm_size=128,
             lstm_input_size=repsize * 5 + 128,  #repsize*5 + hidden_sizes[-1]
             added_fc_input_size=0,
             hidden_activation=nn.ELU(),
             k=k)
         ),
)

#######Default OP3 Refinement Network##########
class RefinementNetwork(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes,
            lstm_size,
            lstm_input_size,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            lambda_output_activation=identity,
            k=None,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lambda_output_activation = lambda_output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.K = k

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=input_width)

        self.lstm = nn.LSTM(lstm_input_size, lstm_size, num_layers=1, batch_first=True)

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            #self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        test_mat = self.avg_pooling(test_mat) #Avg pooling layer

        fc_input_size = int(np.prod(test_mat.shape))
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            #norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            #self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(lstm_size, output_size)
        #self.last_fc.weight.data.uniform_(-init_w, init_w)
        #self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.last_fc2 = nn.Linear(lstm_size, output_size)

        xcoords = np.expand_dims(np.linspace(-1, 1, self.input_width), 0).repeat(self.input_height, 0)
        ycoords = np.repeat(np.linspace(-1, 1, self.input_height), self.input_width).reshape((self.input_height, self.input_width))

        self.coords = from_numpy(np.expand_dims(np.stack([xcoords, ycoords], 0), 0)) #(1, 2, D, D)


    #add_fc_input is used for next step iodine where we want to add action information
    #Input: input (B*K,15,D,D),  hidden1 (B*K,R2),  hidden2 (B*K,R2),  extra_input (B*K,5*R),
    # add_fc_input is usually None except for next step refinement e.g. sequence iodine (B*K,A)
    #   See op3_model.refine(...) to see the exact inputs into the refinement network
    def forward(self, input, hidden1, hidden2, extra_input, add_fc_input=None):
        hi = input #(B*K,15,D,D)

        coords = self.coords.repeat(input.shape[0], 1, 1, 1) #(B*K,2,D,D)
        hi = torch.cat([hi, coords], 1) #(B*K,17,D,D)

        hi = self._apply_forward(hi, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv) #(B*K,64,1,1)
        hi = self.avg_pooling(hi) #Avg pooling layer
        # flatten channels for fc layers
        hi = hi.view(hi.size(0), -1) #(B*K, 64)

        if self.added_fc_input_size != 0:
            hi = torch.cat([hi, add_fc_input], dim=1) #(B*K, 64+A)
        output = self._apply_forward(hi, self.fc_layers, self.fc_norm_layers, use_batch_norm=self.batch_norm_fc) #(B*K, last_hidden_size)

        if extra_input is not None:
            output = torch.cat([output, extra_input], dim=1) #(B*K, last_hidden_size+R*5)

        if len(hidden1.shape) == 2: #(B*K,lstm_size)
            hidden1, hidden2 = hidden1.unsqueeze(0), hidden2.unsqueeze(0) #(1,B*K,lstm_size), (1,B*K,lstm_size)
            hidden1 = hidden1.contiguous()
            hidden2 = hidden2.contiguous()
        self.lstm.flatten_parameters() #For performance / multi-gpu reasons
        output, hidden = self.lstm(output.unsqueeze(1), (hidden1, hidden2)) #Note batch_first = True in lstm initialization
        #output: (B*K,1,R), hidden is tuple of size 2, each of size (B*K,1,lstm_size)


        output1 = self.lambda_output_activation(self.last_fc(output.squeeze(1))) #(B*K,R)
        output2 = self.lambda_output_activation(self.last_fc2(output.squeeze(1))) #(B*K,R)
        return output1, output2, hidden[0], hidden[1]

    def initialize_hidden(self, bs):
        return ptu.zeros((1, bs, self.lstm_size)), ptu.zeros((1, bs, self.lstm_size))

    def _apply_forward(self, input, hidden_layers, norm_layers, use_batch_norm=False):
        h = input
        for layer in hidden_layers:
            h = layer(h)
            # if use_batch_norm:
            #    h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


#######No-sharing OP3 Refinement Network##########
class RefinementNetwork_v2_No_Sharing(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes,
            lstm_size,
            lstm_input_size,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            lambda_output_activation=identity,
            k=None,
    ):
        super().__init__()
        if k is None:
            raise ValueError("A value of k is needed to initialize this model!")
        self.K = k
        self.models = nn.ModuleList()
        self.input_width = input_width
        self.lstm_size = lstm_size

        for i in range(self.K):
            self.models.append(RefinementNetwork(input_width,
                                                 input_height,
                                                 input_channels,
                                                 output_size,
                                                 kernel_sizes,
                                                 n_channels,
                                                 strides,
                                                 paddings,
                                                 hidden_sizes,
                                                 lstm_size,
                                                 lstm_input_size,
                                                 added_fc_input_size,
                                                 batch_norm_conv,
                                                 batch_norm_fc,
                                                 init_w,
                                                 hidden_init,
                                                 hidden_activation,
                                                 lambda_output_activation,
                                                 k=None))


    #add_fc_input is used for next step iodine where we want to add action information
    #Input: input (B*K,15,D,D),  hidden1 (B*K,R2),  hidden2 (B*K,R2),  extra_input (B*K,5*R),
    # add_fc_input is usually None except for next step refinement e.g. sequence iodine (B*K,A)
    def forward(self, input, hidden1, hidden2, extra_input=None, add_fc_input=None):
        vals_output1, vals_output2, vals_hidden1, vals_hidden2 = [], [], [], []
        for i in range(self.K):
            vals = self.models[i](self._get_ith_input(input, i), self._get_ith_input(hidden1, i),
                                          self._get_ith_input(hidden2, i), self._get_ith_input(extra_input, i),
                                          self._get_ith_input(add_fc_input, i))
            vals_output1.append(vals[0])  #(B,R)
            vals_output2.append(vals[1])  #(B,R)
            vals_hidden1.append(vals[2])  #(B,1,lstm_size)
            vals_hidden2.append(vals[3])  #(B,1,lstm_size)

        vals_output1 = torch.cat(vals_output1)
        vals_output2 = torch.cat(vals_output2)
        vals_hidden1 = torch.cat(vals_hidden1)
        vals_hidden2 = torch.cat(vals_hidden2)
        return vals_output1, vals_output2, vals_hidden1, vals_hidden2


    # Input: x (bs*k,*) or None, i representing which latent to pick (Sc)
    # Input: x (bs,*) or None
    def _get_ith_input(self, x, i):
        if x is None:
            return None
        x = x.view([-1, self.K] + list(x.shape[1:]))  #(bs,k,*)
        return x[:, i]

    def initialize_hidden(self, bs):
        return ptu.zeros((1, bs, self.lstm_size)), ptu.zeros((1, bs, self.lstm_size))




