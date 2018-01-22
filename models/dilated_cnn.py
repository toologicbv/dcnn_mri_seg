
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from building_blocks import Basic2DCNNBlock
from torch.autograd import Variable
import math

DEFAULT_DCNN_2D = {'num_of_layers': 10,
                   'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                   'channels': [32, 32, 32, 32, 32, 32, 32, 32, 192, 2],
                   'dilation': [(1, 1), (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (1, 1), (1, 1), (1, 1)],
                   'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   'batch_norm': [False, False, False, False, False, False, False, True, True, False],
                   'dropout': [0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.],
                   'loss_function': nn.Softmax,
                   }


class BaseDilated2DCNN(nn.Module):

    def __init__(self, architecture=DEFAULT_DCNN_2D, use_cuda=False):
        super(BaseDilated2DCNN, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.num_conv_layers = self.architecture['num_of_layers']
        self.model = self._build_dcnn()
        self.loss_func = self.architecture['loss_function']()
        if self.use_cuda:
            self.cuda()

    def _build_dcnn(self):

        layer_list = []
        num_conv_layers = self.architecture['num_of_layers']
        for l in np.arange(num_conv_layers):
            if l == 0:
                in_channels = 1
            else:
                # get previous output channel size
                in_channels = self.architecture['channels'][l - 1]
            print("Constructing layer {}".format(l+1))
            layer_list.append(Basic2DCNNBlock(in_channels, self.architecture['channels'][l],
                                              self.architecture['kernels'][l],
                                              stride=self.architecture['stride'][l],
                                              dilation=self.architecture['dilation'][l],
                                              batch_norm=self.architecture['batch_norm'][l],
                                              prob_dropout=self.architecture['dropout'][l]))

        return nn.Sequential(*layer_list)

    def forward(self, input):
        if not (isinstance(input, torch.autograd.variable.Variable) or isinstance(input, torch.autograd.variable.Variable)):
            raise ValueError("input is not of type torch.autograd.variable.Variable")

        out = self.model(input)
        print("Output shape {}".format(str(out.size())))
        out = self.loss_func(out)
        return out

    def cuda(self):
        super(BaseDilated2DCNN, self).cuda()

    def zero_grad(self):
        super(BaseDilated2DCNN, self).zero_grad()

    def save_params(self, absolute_path):
        torch.save(self.state_dict(), absolute_path)

    def sum_grads(self, verbose=False):
        sum_grads = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads

# dcnn_model = BaseDilated2DCNN(use_cuda=True)
