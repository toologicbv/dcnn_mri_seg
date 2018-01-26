import torch.nn as nn
from torch.nn import init


class Basic2DCNNBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1), batch_norm=False,
                 prob_dropout=0.):
        super(Basic2DCNNBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, groups=1, bias=True)
        self.reset_weights()
        self.non_linearity = nn.ELU(inplace=False)
        self.batch_norm = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            self.batch_norm = True
        self.dropout = False
        if prob_dropout > 0.:
            self.layer_drop = nn.Dropout2d(p=prob_dropout)
            self.dropout = True

    def reset_weights(self):
        init.xavier_normal(self.conv_layer.weight.data)
        if self.conv_layer.bias is not None:
            self.conv_layer.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.non_linearity(out)
        if self.batch_norm:
            out = self.bn(out)
        if self.dropout:
            out = self.layer_drop(out)

        return out
