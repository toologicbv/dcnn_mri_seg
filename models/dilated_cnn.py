
import torch
import torch.nn as nn
from utils.config import DEFAULT_DCNN_2D
import numpy as np
from building_blocks import Basic2DCNNBlock


class BaseDilated2DCNN(nn.Module):

    def __init__(self, architecture=DEFAULT_DCNN_2D, use_cuda=False):
        super(BaseDilated2DCNN, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.num_conv_layers = self.architecture['num_of_layers']
        self.model = self._build_dcnn()
        # we're using CrossEntropyLoss. Implementation of PyTorch combines it with Softmax and hence
        # not need to incorporate Softmax layer in NN
        self.output = self.architecture['output']()
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
        """

        :param input:
        :return: (1) the raw output in order to compute loss with PyTorch cross-entropy (see comment below)
                 (2) the softmax output
        """
        if not isinstance(input, torch.autograd.variable.Variable):
            raise ValueError("input is not of type torch.autograd.variable.Variable")

        out = self.model(input)

        out_softmax = self.output(input.view(-1, input.size(1)))
        # dim parameter in softmax only works in new PyTorch version 0.4.0
        # out_softmax = self.output(input.view(-1, input.size(1), dim=input.size(1)))
        # we want to compute loss and analyse the segmentation predictions. PyTorch loss function CrossEntropy
        # combines softmax with log operation. Hence for loss calculation we need the raw output aka logits
        # without having them passed through the softmax non-linearity
        return out, out_softmax

    def get_loss(self, input, labels):
        # we need to reshape the tensors because CrossEntropy expects 2D tensor (N, C) where C is num of classes
        # the input tensor is in our case [batch_size, num_of_classes, height, width]
        # the labels are                  [batch_size, 1, height, width]
        input = input.view(-1, input.size(1))
        labels = labels.view(-1)

        return self.loss_func(input, labels)

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
