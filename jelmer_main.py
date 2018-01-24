import SimpleITK as sitk
import numpy as np
import lasagne
import lasagne.layers.dnn as ladnn
import theano.tensor as T
import theano
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import sys
import ConvNetUtils as cnu
import Upscale3DLayer
import LasagneEnhance as lase
import scipy.ndimage.interpolation as scndy
import time
from sklearn.metrics import precision_recall_curve


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def softmax_test(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def fcsoftmax(x):
    x = x.flatten(ndim=2)
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)


def getNetwork3D(input_var=None, input_shape=(None, 1, 256, 256, 8), nclass=7):
    net256 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    nfilt = 32
    # Downsample

    net128 = ladnn.Pool3DDNNLayer(net256, pool_size=(2,2,2), stride=None, mode='average_inc_pad')
    net64 = ladnn.Pool3DDNNLayer(net256, pool_size=(4,4,4), stride=None, mode='average_inc_pad')
    net32 = ladnn.Pool3DDNNLayer(net256, pool_size=(8,8,8), stride=None, mode='average_inc_pad')
    net16 = ladnn.Pool3DDNNLayer(net256, pool_size=(16,16,8), stride=None, mode='average_inc_pad')
    net8 = ladnn.Pool3DDNNLayer(net256, pool_size=(32,32,8), stride=None, mode='average_inc_pad')

    # net128 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=2, stride=None, mode='average_inc_pad')
    # net64 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=4, stride=None, mode='average_inc_pad')
    # net32 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=8, stride=None, mode='average_inc_pad')
    # net16 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=16, stride=None, mode='average_inc_pad')
    # net8 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=32, stride=None, mode='average_inc_pad')

    for x in range(6):
        net256 = ladnn.Conv3DDNNLayer(net256, num_filters=nfilt, filter_size=(3, 3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net128 = ladnn.Conv3DDNNLayer(net128, num_filters=nfilt, filter_size=(3, 3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net64 = ladnn.Conv3DDNNLayer(net64, num_filters=nfilt, filter_size=(3, 3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net32 = ladnn.Conv3DDNNLayer(net32, num_filters=nfilt, filter_size=(3, 3, 1), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net16 = ladnn.Conv3DDNNLayer(net16, num_filters=nfilt, filter_size=(3, 3, 1), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net8 = ladnn.Conv3DDNNLayer(net8, num_filters=nfilt, filter_size=(3, 3, 1), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Upsample
    net128 = Upscale3DLayer(net128, (2, 2, 2))
    net64 = Upscale3DLayer(net64, (4, 4, 4))
    net32 = Upscale3DLayer(net32, (8, 8, 8))
    net16 = Upscale3DLayer(net16, (16, 16, 8))
    net8 = Upscale3DLayer(net8, (32, 32, 8))

    # Concatenate along feature dimension
    network = lasagne.layers.ConcatLayer((net256, net128, net64, net32, net16, net8), axis=1) # net8

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=192, filter_size=(1,1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=nclass, filter_size=(1,1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network

def getNetworkNonDilated(input_var=None, input_shape=(None, 1, 256, 256), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 32

    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify)
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(3,3), nonlinearity=lasagne.nonlinearities.rectify)
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(7,7), nonlinearity=lasagne.nonlinearities.rectify)
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(15,15), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network

def getNetworkConventional(input_var=None, input_shape=(None, 1, 256, 256), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    C = 16

    # Get C feature maps
    pw = 67
    while pw>1:
        network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        pw -= 2

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network


# Dilated convolutions. Make sure to pad input before use!
def getNetworkNonDilated3D(input_var=None, input_shape=(None, 1, 67, 67, 67), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 16

    # Get C feature maps
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 67

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=192, filter_size=(1, 1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)

    network = ladnn.Conv3DDNNLayer(network, num_filters=nclass, filter_size=(1,1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    return network



# Dilated convolutions. Make sure to pad input before use!
def getNetworkNonDilated3D(input_var=None, input_shape=(None, 1, 67, 67, 67), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    C = 16

    # Get C feature maps
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 3 # 65
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 5 # 59
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 7 # 59
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 9 # 59
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 11 # 59
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 13 # 59
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 15 # 59
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=192, filter_size=(1, 1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)

    network = ladnn.Conv3DDNNLayer(network, num_filters=nclass, filter_size=(1,1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    return network


# Dilated convolutions. Make sure to pad input before use!
def getNetworkDilated3D(input_var=None, input_shape=(None, 1, 67, 67, 67), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 16

    # Get C feature maps
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 3 # 65

    # Large module
    #  network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 5 # 61
    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 5 # 59
    # network = lase.DilatedConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(1,1,1), nonlinearity=lasagne.nonlinearities.rectify) # 9 #
    # network = lase.DilatedConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(3,3,3), nonlinearity=lasagne.nonlinearities.rectify) # 17
    # network = lase.DilatedConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(7,7,7), nonlinearity=lasagne.nonlinearities.rectify) # 33
    # network = lase.DilatedConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(15,15,15), nonlinearity=lasagne.nonlinearities.rectify) # 65

    network = lase.DilatedDNNConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(2,2,2), nonlinearity=lasagne.nonlinearities.rectify) # 9 #
    network = lase.DilatedDNNConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(4,4,4), nonlinearity=lasagne.nonlinearities.rectify) # 17
    network = lase.DilatedDNNConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(8,8,8), nonlinearity=lasagne.nonlinearities.rectify) # 33

    network = lase.DilatedDNNConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(8,8,8), nonlinearity=lasagne.nonlinearities.rectify) # 49


    # network = lase.DilatedConv3DLayer(network, num_filters=C, filter_size=(3,3,3), dilation=(16,16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65

    network = ladnn.Conv3DDNNLayer(network, num_filters=C, filter_size=(3, 3, 3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 67

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=32, filter_size=(1, 1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)

    network = ladnn.Conv3DDNNLayer(network, num_filters=nclass, filter_size=(1,1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    return network

def getSubNet(network, C=16):
    fixed = False

    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')
    #     # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257

    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 67, 131, 259
    if fixed:
        network.params[network.W].remove('trainable')
        network.params[network.b].remove('trainable')

    return network



def getNetworkDilated25DSlabs(input_axi=None, input_sag=None, input_cor=None, nclass=7, wx=21, wy=21, wz=21):
    ps = 131 # 67 # 131

    net_axi = lasagne.layers.InputLayer(shape=(wz, 1, wx+ps-1, wy+ps-1), input_var=input_axi)
    net_sag = lasagne.layers.InputLayer(shape=(wx, 1, wy+ps-1, wz+ps-1), input_var=input_sag)
    net_cor = lasagne.layers.InputLayer(shape=(wy, 1, wx+ps-1, wz+ps-1), input_var=input_cor)
    C = 32
    net_axi = getSubNet(net_axi, C=C)
    net_sag = getSubNet(net_sag, C=C)
    net_cor = getSubNet(net_cor, C=C)

    # net_axi now has shape wz, 32, wx, wy
    # net_sag now has shape wx, 32, wy, wz
    # net_cor now has shape wy, 32, wx, wz
    # Let's reshape and shuffle
    net_axi = lasagne.layers.ReshapeLayer(net_axi, (1, wz, 32, wx, wy))
    net_axi = lasagne.layers.DimshuffleLayer(net_axi, (0, 2, 3, 4, 1))

    net_sag = lasagne.layers.ReshapeLayer(net_sag, (1, wx, 32, wy, wz))
    net_sag = lasagne.layers.DimshuffleLayer(net_sag, (0, 2, 1, 3, 4 ))

    net_cor = lasagne.layers.ReshapeLayer(net_cor, (1, wy, 32, wx, wz))
    net_cor = lasagne.layers.DimshuffleLayer(net_cor, (0, 2, 3, 1, 4))

    network = lasagne.layers.ConcatLayer((net_axi, net_sag, net_cor), axis=1)


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = ladnn.Conv3DDNNLayer(network, num_filters=192, filter_size=(1, 1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)

    network = ladnn.Conv3DDNNLayer(network, num_filters=nclass, filter_size=(1,1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network


# Dilated convolutions. Make sure to pad input before use!
def getNetworkDilated25D(input_axi=None, input_sag=None, input_cor=None, input_axi_shape=(None, 1, 256, 256), input_sag_shape=(None, 1, 256, 256), input_cor_shape=(None, 1, 256, 256), nclass=7):
    net_axi = lasagne.layers.InputLayer(shape=input_axi_shape, input_var=input_axi)
    net_sag = lasagne.layers.InputLayer(shape=input_sag_shape, input_var=input_sag)
    net_cor = lasagne.layers.InputLayer(shape=input_cor_shape, input_var=input_cor)

    C = 32

    net_axi = getSubNet(net_axi, C=C)
    net_sag = getSubNet(net_sag, C=C)
    net_cor = getSubNet(net_cor, C=C)

    network = lasagne.layers.ConcatLayer((net_axi, net_sag, net_cor), axis = 1)

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=fcsoftmax, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    return network, net_axi, net_sag, net_cor


# Dilated convolutions. Make sure to pad input before use!
# Input 203x203
def getNetworkDilatedCascadeNew(input_var=None, input_shape=(None, 1, 256, 256), nclass=7, inputx=203, inputy=203):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 32

    # FIRST NETWORK
    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257


    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 259


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    networkfirst = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # PAD OUTPUT
    # No padding required. Image now 137 x 137
    # network = lasagne.layers.PadLayer(networkfirst, 65) #
    inputtwo = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)


    # inputtwo = lasagne.layers.SliceLayer(inputtwo, )
    inputtwo = lasagne.layers.ExpressionLayer(inputtwo, lambda X: X[:, :, 33:inputx-33, 33:inputy-33], output_shape=(32, 1, inputx-66, inputy-66))

    network = lasagne.layers.ConcatLayer((network, inputtwo), axis=1)

    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257


    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 259


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())


    # IMAGE NOW 70x70
    return networkfirst, network



# Dilated convolutions. Make sure to pad input before use!
def getNetworkDilatedCascade(input_var=None, input_shape=(None, 1, 256, 256), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 32

    # FIRST NETWORK
    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257


    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 259


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    networkfirst = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # PAD OUTPUT
    network = lasagne.layers.PadLayer(networkfirst, 65) #
    inputtwo = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    network = lasagne.layers.ConcatLayer((network, inputtwo), axis=1)

    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257


    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 259


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    return networkfirst, network


# Dilated convolutions. Make sure to pad input before use!
def getNetworkDilatedFirstLayer(input_var=None, input_shape=(None, 1, 256, 256), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 32

    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65

    return network


# Dilated convolutions. Make sure to pad input before use!
def getNetworkDilated(input_var=None, input_shape=(None, 1, 256, 256), nclass=7):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    C = 32

    # Get C feature maps
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    # Large module
    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify)   # 9
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify)   # 17
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify)   # 33
    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify) # 65

    network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(32,32), nonlinearity=lasagne.nonlinearities.rectify) # 129
    # network = lase.DilatedConv2DLayer(network, num_filters=C, filter_size=(3,3), dilation=(64,64), nonlinearity=lasagne.nonlinearities.rectify) # 257


    network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # 259


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    # network = lasagne.layers.Conv2DLayer(network, num_filters=C, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())


    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network


def getNetwork(input_var = None, input_shape = (None, 1, 256, 256), nclass=7):
    net256 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    nfilt = 32
    # Downsample
    net128 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=2, stride=None, mode='average_inc_pad')
    net64 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=4, stride=None, mode='average_inc_pad')
    net32 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=8, stride=None, mode='average_inc_pad')
    net16 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=16, stride=None, mode='average_inc_pad')
    net8 = lasagne.layers.pool.Pool2DLayer(net256, pool_size=32, stride=None, mode='average_inc_pad')

    for x in range(6):
        net256 = lasagne.layers.Conv2DLayer(net256, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        net128 = lasagne.layers.Conv2DLayer(net128, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) #W=net256.W, b=net256.b) #
        net64 = lasagne.layers.Conv2DLayer(net64, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) #W=net256.W, b=net256.b) #
        net32 = lasagne.layers.Conv2DLayer(net32, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu,  W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) #W=net256.W, b=net256.b) #
        net16 = lasagne.layers.Conv2DLayer(net16, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # W=net256.W, b=net256.b) #
        net8 = lasagne.layers.Conv2DLayer(net8, num_filters=nfilt, filter_size=(3, 3), pad='same', nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal()) # W=net256.W, b=net256.b)

    # Upsample
    net128 = lasagne.layers.pool.Upscale2DLayer(net128, 2)
    net64 = lasagne.layers.pool.Upscale2DLayer(net64, 4)
    net32 = lasagne.layers.pool.Upscale2DLayer(net32, 8)
    net16 = lasagne.layers.pool.Upscale2DLayer(net16, 16)
    net8 = lasagne.layers.pool.Upscale2DLayer(net8, 32)

    # Concatenate along feature dimension
    network = lasagne.layers.ConcatLayer((net256, net128, net64, net32, net16, net8), axis=1) # net8

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())

    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.dropout(network, p=0.5)
    # network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
    return network



def generateBatchSecond(nsamp = 20):
    batch_im = np.zeros((nsamp, 1, 101, 101))
    batch_la = np.zeros((nsamp, 1, 101, 101))

    for ns in range(nsamp):
        x1 = np.random.randint(low=10, high=90, size=1)
        x2 = np.random.randint(low=10, high=90, size=1)
        y1 = np.random.randint(low=10, high=50, size=1)
        y2 = np.random.randint(low=10, high=50, size=1)
        batch_im[ns, 0, x1-4:x1+4, y1-4:y1+4] = 1
        batch_im[ns, 0, x2-4:x2+4, y2-4:y2+4] = 1



        if np.random.rand(1) < 0.5:
            batch_im[ns, 0, x1-4:x1+4, y1+10-4:y1+10+4] = 2
            batch_im[ns, 0, x2, y2+10] = 2
            batch_la[ns, 0, x1, y1] = 1
        else:
            batch_im[ns, 0, x1, y1+10] = 2
            batch_im[ns, 0, x2-4:x2+4, y2+10-4:y2+10+4] = 2
            batch_la[ns, 0, x2, y2] = 1

    batch_im = batch_im # + np.random.randn(batch_im.shape[0], batch_im.shape[1], batch_im.shape[2], batch_im.shape[3])

    return batch_im, batch_la


def generateBatchThird(nsamp=20):
    batch_im = np.random.rand(nsamp, 1, 256, 256)
    batch_la = (np.random.rand(nsamp, 1, 256, 256)<0.1).astype('int16')
    return batch_im, batch_la


def generateBatch3D(images, labels, nsamp=10, nclass=7):
    pw = 67 # 51  # patch width
    fov = 51 # 35   # field-of-view / receptive field

    batch_im = np.zeros((nsamp, 1, pw, pw, pw)) # 256, 256))
    batch_la = np.zeros((nsamp, nclass, (pw-fov)+1, (pw-fov)+1, (pw-fov)+1)) # 256, 256))

    for ns in range(nsamp):
        ind = np.random.randint(0, len(images), 1)
        imageim = images[ind]
        labelim = labels[ind]

        # Get random cube of pw x pw x pw
        offx = np.random.randint(low=0, high=imageim.shape[0]-pw)
        offy = np.random.randint(low=0, high=imageim.shape[1]-pw)
        offz = np.random.randint(low=0, high=imageim.shape[2]-pw)
        labelcrop = labelim[offx:offx+(pw-fov)+1, offy:offy+(pw-fov)+1, offz:offz+(pw-fov)+1]

        while np.sum(labelcrop) == 0:
            # print('Redraw')
            # Get random cube of pw x pw x pw
            offx = np.random.randint(low=0, high=imageim.shape[0]-pw)
            offy = np.random.randint(low=0, high=imageim.shape[1]-pw)
            offz = np.random.randint(low=0, high=imageim.shape[2]-pw)
            labelcrop = labelim[offx:offx+(pw-fov)+1, offy:offy+(pw-fov)+1, offz:offz+(pw-fov)+1]



        batch_im[ns, 0, :, :,:] = imageim[offx:offx+pw, offy:offy+pw, offz:offz+pw]




        for cl in range(nclass):
            batch_la[ns, cl, :, :] = (labelcrop==cl).astype('int16')

    return batch_im, batch_la







def generateBatch25D(images, labels, indices, nsamp=10, nclass=7):
    psx = 131 # 67 # 131 # 51 #  131 #67 # 131 #67 # 67, 131, 259
    psy = 131 # 67 # 131 # 51 # 131 # 67 # 131 # 67 # 67, 131, 259
    psz = 131 # 67 # 131 # 51 # 131 # 67 # 131 # 67 # 67, 131, 259

    padw = 65 # 33 # 65 # 25 # 65 # 33 # 65 # 33

    # Output = 10x10x10
    batch_axi = np.zeros((nsamp, 1, psx, psy))
    batch_sag = np.zeros((nsamp, 1, psy, psz))
    batch_cor = np.zeros((nsamp, 1, psx, psz))


    batch_la = np.zeros((nsamp))
    rando = np.random.randint(low=0, high=len(images), size=1)
    # print('Shape indices ' + str(indices[rando][0].shape))
    # print('Shape indices ' + str(indices[rando][1].shape))
    # print('Shape indices ' + str(indices[rando][2].shape))

    for ns in range(nsamp):
        ind = np.random.randint(low=0, high=len(images), size=1)

        imageim = images[ind]
        labelim = labels[ind]
        indexim = indices[ind]

        label_cur = int(ns/(nsamp/nclass))
        index = indexim[label_cur]

        coordid = np.random.randint(0, index.shape[0])

        offx = index[coordid, 0] # np.random.randint(0, labelim.shape[0])
        offy = index[coordid, 1] # np.random.randint(0, labelim.shape[1])
        offz = index[coordid, 2] # np.random.randint(0, labelim.shape[2])


        # imageim = imageim[offx:offx+psx, offy:offy+psy, offz:offz+psz]
        # GET THREE SLICES CENTERED AT VOXEL
        batch_axi[ns, 0, :, :] = np.squeeze(imageim[offx:offx+psx, offy:offy+psy, offz+padw])
        batch_sag[ns, 0, :, :] = np.squeeze(imageim[offx+padw, offy:offy+psy, offz:offz+psz])
        batch_cor[ns, 0, :, :] = np.squeeze(imageim[offx:offx+psx, offy+padw, offz:offz+psz])

        label_vox = np.squeeze(labelim[offx, offy, offz])
        batch_la[ns] = label_vox

    # print(batch_la)

    return batch_axi, batch_sag, batch_cor, batch_la

def generateBatch25DSlabs(images, labels, indices, nsamp=10, nclass=7):
    ww = 51
    ps = 131 + ww - 1 # 131 + 20

    padw = 65 # 33 # 65 # 33

    # Output = 10x10x10
    batch_axi = np.zeros((ww, 1, ps, ps))
    batch_sag = np.zeros((ww, 1, ps, ps))
    batch_cor = np.zeros((ww, 1, ps, ps))

    batch_la = np.zeros((1, 3, ww, ww, ww))

    # Pick random image
    ind = np.random.randint(0, len(images), 1)
    imageim = images[ind]
    labelim = labels[ind]

    # Fill labels
    chunk_la = np.zeros((ww, ww, ww))

    class_count = np.zeros((nclass))
    for nc in range(nclass):
        class_count[nc] = np.sum(chunk_la==nc)

    while np.count_nonzero(class_count) is not nclass:
        print('Looking for chunk')
        # Get random cube of ps by ps by ps
        offx = np.random.randint(low=0, high=imageim.shape[0]-ps)
        offy = np.random.randint(low=0, high=imageim.shape[1]-ps)
        offz = np.random.randint(low=0, high=imageim.shape[2]-ps)
        chunk_la = labelim[offx:offx+ww, offy:offy+ww, offz:offz+ww] # Is now ww*ww*ww
        for nc in range(nclass):
            class_count[nc] = np.sum(chunk_la==nc)

    chunk = imageim[offx:offx+ps, offy:offy+ps, offz:offz+ps]

    # Fill batch_axi
    for wi in range(ww):
        batch_axi[wi, 0, :, :] = chunk[:,:,wi+padw]

    # Fill batch_sag
    for wi in range(ww):
        batch_sag[wi, 0, :, :] = chunk[wi+padw, :, :]

    # Fill batch_cor
    for wi in range(ww):
        batch_cor[wi, 0, :, :] = chunk[:, wi+padw, :]


    # Fill labels
    # chunk_la = labelim[offx:offx+ww, offy:offy+ww, offz:offz+ww] # Is now ww*ww*ww

    for nc in range(nclass):
        batch_la[0, nc, :, :, :]=(chunk_la == nc).astype('int16')

    return batch_axi, batch_sag, batch_cor, batch_la


def generateBatch2DNew(images, labels, nsamp=10, nclass=7, classcount=(1,1)):
    ps = 203

    batch_im = np.zeros((nsamp, 1, ps, ps))
    batch_la_one = np.zeros((nsamp, nclass, 137, 137))
    batch_la_two = np.zeros((nsamp, nclass, 71, 71))

    for ns in range(nsamp):
        ind = np.random.randint(0, len(images), 1)
        imageim = images[ind]
        labelim = labels[ind]
        offx = np.random.randint(0, imageim.shape[0]-ps)
        offy = np.random.randint(0, imageim.shape[1]-ps)
        imageim = imageim[offx:offx+ps, offy:offy+ps]
        batch_im[ns, 0, :, :] = imageim
        labelim = labelim[offx:offx+137, offy:offy+137]
        for cl in range(nclass):
            batch_la_one[ns, cl, :, :] = (labelim==cl).astype('int16')
        labelim = labels[ind]
        labelim = labelim[offx+33:offx+104, offy+33:offy+104]
        # print(labelim.shape)
        # rint(labelim)
        for cl in range(nclass):
            batch_la_two[ns, cl, :, :] = (labelim==cl).astype('int16')

    print(batch_la_one.shape)
    print(batch_la_two.shape)


    return batch_im, batch_la_one, batch_la_two



def generateBatch2D(images, labels, nsamp=10, nclass=7, classcount=(1,1)):
    ps = 181 # 201 # 137 # 201 # 329 # 89 # 105 # 137 # 201 # 329 # 259 + 70 = 329 # 17 + 70 = 87 # 131 + 70 = 201 # 329 # 201
    ss = 50
    # mw = (259-1)/2

    # batch image patch
    batch_im = np.zeros((nsamp, 1, ps, ps))
    # batch reference image for the different classes (so for each reference class 1 image)
    batch_la = np.zeros((nsamp, nclass, ss+1, ss+1))
    # the "complete" reference image - containing all class labels
    class_im = np.zeros((1, nsamp, ss+1, ss+1))

    for ns in range(nsamp):
        ind = np.random.randint(0, len(images), 1)
        imageim = images[ind]
        # imageim = np.pad(imageim, 129, 'constant', constant_values=(0,)).astype('float32') # 65

        labelim = labels[ind]

        offx = np.random.randint(0, imageim.shape[0]-ps)
        offy = np.random.randint(0, imageim.shape[1]-ps)

        imageim = imageim[offx:offx+ps, offy:offy+ps]

        # rotangle = np.random.randint(low=-45, high=45)

        # imageim = scndy.rotate(imageim, rotangle, reshape=False)

        #
        # print('Mean intensity ' + str(np.mean(imageim)))
        # imame = np.mean(imageim)
        # imast = np.std(imageim)
        # imageim = (imageim-imame)/imast
        # print('Mean intensity ' + str(np.mean(imageim)))


        # Add random shift to intensities in training image

        # mu = 0.0
        # sigma = 0.2
        # shiftval = np.random.normal(mu, sigma, 1)
        # imageim = imageim + shiftval
        # print(shiftval)

        batch_im[ns, 0, :, :] = imageim

        labelim = labelim[offx:offx+ss+1, offy:offy+ss+1]

        # labelim = scndy.rotate(labelim, rotangle, reshape=False, order=0)

        class_im[0, ns, :, :] = labelim

        for cl in range(nclass):
            batch_la[ns, cl, :, :] = (labelim==cl).astype('int16')


    priors = [0.6, 0.1, 0.3]
    appear = [float(np.sum(batch_la[:, 0, :, :]))/float(np.sum(batch_la)), float(np.sum(batch_la[:, 1, :, :]))/float(np.sum(batch_la)), float(np.sum(batch_la[:, 2, :, :]))/float(np.sum(batch_la))]

    batch_mask = np.zeros((1, nsamp, ss+1, ss+1), dtype='float32')
    # for cl in range(nclass):
    #     batch_mask[class_im==cl] = priors[cl]/float(appear[cl])
    #
    # for cl in range(nclass):
    #     print('Class ' + str(cl) + ' ' + str(float(np.sum(batch_la[:, cl, :, :]))/float(np.sum(batch_la))))
    #     # batch_mask[np.squeeze(batch_la[:, cl, :, :]).astype('int16')] = (priors[cl]/(float(np.sum(batch_la[:, cl, :, :]))/float(np.sum(batch_la))))
    #
    #     # :, cl, :, :] = batch_la[:, cl, :, :] * (priors[cl]/(float(np.sum(batch_la[:, cl, :, :]))/float(np.sum(batch_la))))


    print(batch_la.shape)
    print(batch_mask.shape)

    # Get weights for all classes
    # batch_mask = np.ones((nsamp, nclass, 71, 71), dtype='float32')
    # for cl in range(nclass):
    #     ninc = np.sum(batch_la[:, cl, :, :])
    #     nwei = 1.0 - float(ninc)/float(np.prod(batch_la.shape)/nclass)
    #     print('Class ' + str(cl) + ' weight ' + str(nwei))
    #     batch_mask[:, cl, :, :] *= nwei

    return batch_im, batch_la, batch_mask



def generateBatchSixth(images, labels, nsamp=10, undersample=False, shift=False, nclass=7, size=256, classcount=(1,1)):
    print('Image size ' + str(size))

    # 129 for 259
    ps = 129 # 7

    # pad width out
    pw = 9
    batch_im = np.zeros((nsamp, 1, size+2*ps, size+2*ps)) # 256, 256))
    batch_la = np.zeros((nsamp, nclass, size+2*pw, size+2*pw)) # 256, 256))

    npos = 0
    while npos == 0:
        for ns in range(nsamp):
            ind = np.random.randint(0, len(images), 1)
            imageim = images[ind]
            labelim = labels[ind]
            # Shift
            # if shift:
            # shiftx = np.random.randint(0, 2*16, 1)
            # shifty = np.random.randint(0, 2*16, 1)
            # imageim = np.pad(imageim, 16, 'constant', constant_values=(0,))
            # imageim = imageim[shiftx:shiftx+size, shifty:shifty+size]

            # PAD IMAGE WITH 33 VOXELS
            imageim = np.pad(imageim, ps, 'constant', constant_values=(0,))

            # labelim = np.pad(labelim, 16, 'constant', constant_values=(0,))
            # labelim = labelim[shiftx:shiftx+size, shifty:shifty+size]

            # PAD IMAGE WITH 33 VOXELS
            labelim = np.pad(labelim, pw, 'constant', constant_values=(0,))

            batch_im[ns, 0, :, :] = imageim
            for cl in range(nclass):
                batch_la[ns, cl, :, :] = (labelim==cl).astype('int16')
            #batch_la[ns, 0, :, :] = labels[ind]
        npos = np.sum(batch_la)

    # Get weights for all classes
    batch_mask = np.ones((nsamp, nclass, size+2*pw, size+2*pw), dtype='float32')
    for cl in range(nclass):
        ninc = np.sum(batch_la[:, cl, :, :])
        nwei = 1.0 - float(ninc)/float(np.prod(batch_la.shape)/nclass)
        # nwei = 1.0/float(classcount[cl]) # float(np.sum(classcount))/float(classcount[cl])
        # nwei
	    # nwei = float(nsamp*size*size)/float(ninc+1)
        print('Class ' + str(cl) + ' weight ' + str(nwei))
        batch_mask[:, cl, :, :] *= nwei



    return batch_im, batch_la, batch_mask


def generateBatchFourth(images, labels, nsamp=10, undersample=False, shift=False, nclass=7, size=256, classcount=(1,1)):
    print('Image size ' + str(size))
    batch_im = np.zeros((nsamp, 1, size, size)) # 256, 256))
    batch_la = np.zeros((nsamp, nclass, size, size)) # 256, 256))

    npos = 0
    while npos == 0:
        for ns in range(nsamp):
            ind = np.random.randint(0, len(images), 1)
            imageim = images[ind]
            labelim = labels[ind]
            # Shift
            if shift:
                shiftx = np.random.randint(0, 2*16, 1)
                shifty = np.random.randint(0, 2*16, 1)
                imageim = np.pad(imageim, 16, 'constant', constant_values=(0,))
                imageim = imageim[shiftx:shiftx+size, shifty:shifty+size]
                labelim = np.pad(labelim, 16, 'constant', constant_values=(0,))
                labelim = labelim[shiftx:shiftx+size, shifty:shifty+size]
            batch_im[ns, 0, :, :] = imageim
            for cl in range(nclass):
                batch_la[ns, cl, :, :] = (labelim==cl).astype('int16')
            #batch_la[ns, 0, :, :] = labels[ind]
        npos = np.sum(batch_la)

    # Get weights for all classes
    batch_mask = np.ones((nsamp, nclass, size, size), dtype='float32')
    for cl in range(nclass):
        ninc = np.sum(batch_la[:, cl, :, :])
        nwei = 1.0 - float(ninc)/float(nsamp*size*size)
        # nwei = 1.0/float(classcount[cl]) # float(np.sum(classcount))/float(classcount[cl])
        # nwei
	    # nwei = float(nsamp*size*size)/float(ninc+1)
        print('Class ' + str(cl) + ' weight ' + str(nwei))
        batch_mask[:, cl, :, :] *= nwei



    # print('Batch contains ' + str(np.sum(batch_la)) + ' positives.')
    # nneg = float(np.prod(batch_la.shape)) - npos
    # pneg = npos/nneg
    # print('Probability for negative ' + str(pneg))


    # if undersample:
    #     batch_mask = (np.random.rand(nsamp, 1, 320, 320) < pneg).astype('int16')
    #     batch_mask[batch_la==1] = 1
    # else:
    #     batch_mask = np.ones((nsamp, 1, 320, 320), dtype='float32')
    #     batch_mask[batch_la==1] *= (1/pneg)

    # Flatten labels and mask
    # batch_la = batch_la.flatten()
    # batch_mask = batch_mask.flatten()


    return batch_im, batch_la, batch_mask

def generateBatchFirst(nsamp = 20):
    batch_im = np.zeros((nsamp, 1, 101, 101))
    batch_la = np.zeros((nsamp, 1, 101, 101))

    for ns in range(nsamp):
        x1 = np.random.randint(low=10, high=90, size=1)
        x2 = np.random.randint(low=10, high=90, size=1)
        y1 = np.random.randint(low=10, high=50, size=1)
        y2 = np.random.randint(low=10, high=50, size=1)
        batch_im[ns, 0, x1-4:x1+4, y1-4:y1+4] = 1
        batch_im[ns, 0, x2-4:x2+4, y2-4:y2+4] = 1



        if np.random.rand(1) < 0.5:
            batch_im[ns, 0, x1-4:x1+4, y1+25-4:y1+25+4] = 2
            batch_la[ns, 0, x1-4:x1+4, y1-4:y1+4] = 1
        else:
            batch_im[ns, 0, x2-4:x2+4, y2+25-4:y2+25+4] = 2
            batch_la[ns, 0, x2-4:x2+4, y2-4:y2+4] = 1

    batch_im = batch_im + np.random.randn(batch_im.shape[0], batch_im.shape[1], batch_im.shape[2], batch_im.shape[3])

    return batch_im, batch_la


# Only for 2D
def fitshape(x, shape, fillval=0):
    out = np.ones(shape)*fillval

    # Crop
    if x.shape[0]>shape[0]:
        x = x[:shape[0],:]
    if x.shape[1]>shape[1]:
        x = x[:,:shape[1]]

    # Pad
    out[:x.shape[0], :x.shape[1]] = x

    return out

def loadImageDir(imagefiles, nclass=2):
    print(imagefiles)

    # Images is a list of 256x256 2D images
    images = []
    # Labels is a list of 256x256 2D images
    labels = []
    processed = 0

    #imagefiles = imagefiles[:1]


    classcount = np.zeros((nclass))

    # Iterate over training images
    for f in imagefiles:
        print('Loading ' + str(processed) + '/' + str(len(imagefiles)))
        print(f)
        processed = processed + 1
        reffile = f.replace('images', 'reference')
        if 'brain' in f or 'breast' in f:
            reffile = reffile.replace('nii', 'mhd')
        # If reference file exists
        if os.path.isfile(reffile):
            # Load image file
            image, spacing = cnu.load_mhd_to_npy(f)
            # Load reference file
            ref, spacing = cnu.load_mhd_to_npy(reffile)

            # image = scndy.zoom(image, (0.5, 0.5, 0.5))
            # ref = scndy.zoom(ref, (0.5, 0.5, 0.5), order=0)
            # im = (im>0).astype('int8')
            # im = im.astype('int16')

            # if self.nclass == 2:
            # ref[ref!=4] = 0 # >1
            # ref[ref==4] = 1


            # if 'brain' in f:
            #     image = image[:256, :256, :256]
            #     ref = ref[:256, :256, :256]

            print(image.shape)
            # NORMALIZE
            # if 'cardiacmr' in reffile:
            #     print('Mean intensity ' + str(np.mean(image)))
            #     imame = np.mean(image)
            #     imast = np.std(image)
            #     image = (image-imame)/imast
            #     print('Mean intensity ' + str(np.mean(image)))

            # if 'pancreas' in reffile:
            #     print('Mean intensity ' + str(np.mean(image)))
            #     image = image.astype('float32')/1024.0
            #     print('Mean intensity ' + str(np.mean(image)))

            # print('Maximum intensity ' + str(np.max(image)))
            # image = image.astype('float32')/float(np.max(image))
            # print('Maximum intensity ' + str(np.max(image)))


            print(ref.shape)

            if 'cor' in f:
                image += 1024

            padwidth = 65 # 66 # 33 # 65 #33 # 65 # 129 #65 # 129 # 17 # 33 # 65 # 129 # 8 # 65

            for cl in range(nclass):
                classcount[cl] += np.sum(ref==cl)

            intorfloat = 'float32'
            if 'pancreas' in reffile:
                intorfloat = 'int16'
            for z in range(image.shape[2]):
                # print('Add axial')
                laslice = np.squeeze(ref[:,:,z])
                imslice = np.squeeze(image[:,:,z])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)






            # if not 'pancreas' in reffile:
            for y in range(image.shape[1]):
                # print('Add axial')
                laslice = np.squeeze(ref[:,y,:])
                imslice = np.squeeze(image[:,y,:])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)

            for x in range(image.shape[0]):
                # print('Add axial')
                laslice = np.squeeze(ref[x,:,:])
                imslice = np.squeeze(image[x,:,:])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)

            # if 'brain' in f or 'cardiacmr' in f: #  or 'pancreas' in f:
            #     for y in range(image.shape[1]):
            #         # print('Add sagittal')
            #         section = np.squeeze(ref[:,y,:])
            #         section = fitshape(section, (size, size))
            #         labels.append(section)
            #         section = np.squeeze(image[:,y,:])
            #         section = fitshape(section, (size, size), fillval=np.min(image))
            #         images.append(section)
            #     for x in range(image.shape[0]):
            #         # print('Add coronal')
            #         section = np.squeeze(ref[x,:,:])
            #         section = fitshape(section, (size, size))
            #         labels.append(section)
            #         section = np.squeeze(image[x,:,:])
            #         section = fitshape(section, (size, size), fillval=np.min(image))
            #         images.append(section)


    return images, labels, classcount


def getClassIndices(label_image, class_label):
    return np.nonzero(label_image==class_label)


def loadImageDir25D(imagefiles, nclass=3):
    print(imagefiles)
    images = []
    labels = []
    classindices = []
    processed = 0

    padw = 65 # 33 # 65 # 33 # 65 # 33 # 33, 65, 129

    # Iterate over training images
    for f in imagefiles:
        print('Loading ' + str(processed) + '/' + str(len(imagefiles)))
        print(f)
        processed = processed + 1
        reffile = f.replace('images', 'reference')
        if 'brain' in f or 'breast' in f:
            reffile = reffile.replace('nii', 'mhd')
        # If reference file exists
        if os.path.isfile(reffile):
            # Load image file
            image, spacing = cnu.load_mhd_to_npy(f)
            print('Image shape ' + str(image.shape))
            print('Mean intensity ' + str(np.mean(image)))
            imame = np.mean(image)
            imast = np.std(image)
            image = (image-imame)/imast
            print('Mean intensity ' + str(np.mean(image)))

            # Load reference file
            ref, spacing = cnu.load_mhd_to_npy(reffile)
            print('Reference shape ' + str(ref.shape))
            # ref = np.pad(ref, padw, 'constant', constant_values=(0,))
            # print('Reference shape ' + str(ref.shape))

            emptyclass = False
            print(ref.shape)
            # ref[image<50] = -1
            imageclassindices = []
            # Add offset classes
            for c in range(nclass):
                imageclassindices.append((np.asarray(getClassIndices(ref, c)).T).astype('int16'))
                print('Contains ' + str(imageclassindices[-1].shape[0]) + ' candidates in class ' + str(c))
                if imageclassindices[-1].shape[0] == 0:
                    emptyclass = True
            if emptyclass:
                print('Empty class, not adding this scan!')
            else:
                print('Adding this scan')
                classindices.append(imageclassindices)

            # PAD 3D VOLUME
            image = np.pad(image, padw, 'constant', constant_values=(0,))
            print('Image shape after padding ' + str(image.shape))

            print(image.shape)
            print(ref.shape)
            images.append(image)
            labels.append(ref)

    return images, labels, classindices




def loadImageDir3D(imagefiles):
    print(imagefiles)
    images = []
    labels = []
    processed = 0

    padw = 25 # 17

    # Iterate over training images
    for f in imagefiles:
        print('Loading ' + str(processed) + '/' + str(len(imagefiles)))
        print(f)
        processed = processed + 1
        reffile = f.replace('images', 'reference')
        if 'brain' in f or 'breast' in f:
            reffile = reffile.replace('nii', 'mhd')
        # If reference file exists
        if os.path.isfile(reffile):
            # Load image file
            image, spacing = cnu.load_mhd_to_npy(f)
            print('Image shape ' + str(image.shape))
            print('Mean intensity ' + str(np.mean(image)))
            if 'cardiacmr' in f:
                imame = np.mean(image)
                imast = np.std(image)
                image = (image-imame)/imast
            print('Mean intensity ' + str(np.mean(image)))
            image = np.pad(image, padw, 'constant', constant_values=(0,))
            print('Image shape ' + str(image.shape))

            # Load reference file
            ref, spacing = cnu.load_mhd_to_npy(reffile)
            print('Reference shape ' + str(ref.shape))
            # ref = np.pad(ref, padw, 'constant', constant_values=(0,))
            # print('Reference shape ' + str(ref.shape))

            print(image.shape)
            print(ref.shape)
            if 'cor' in f:
                image += 1024
            images.append(image)
            labels.append(ref)

    return images, labels





def saveNetwork(network, filename):
    param_values = lasagne.layers.get_all_param_values(network)
    with open(filename, 'wb') as f:
        pickle.dump(param_values, f)

def evaluateSegmentation(automatic,reference,nroflabels):
#    print 'Computing Dice coefficient...'
    segdim = automatic.shape
    refdim = reference.shape

    if segdim!=refdim:
        print 'Dimensions not equal.'
        return

    TP = np.zeros(nroflabels+1)
    TN = np.zeros(nroflabels+1)
    FP = np.zeros(nroflabels+1)
    FN = np.zeros(nroflabels+1)

    for labeli in xrange(1,nroflabels+1): #skip background (0)
        TP[labeli] = np.sum(np.logical_and(automatic==labeli,reference==labeli))
        FP[labeli] = np.sum(np.logical_and(automatic==labeli,reference!=labeli))
        TN[labeli] = np.sum(np.logical_and(automatic!=labeli,reference!=labeli))
        FN[labeli] = np.sum(np.logical_and(automatic!=labeli,reference==labeli))

    return TP, TN, FP, FN

def test25DFull(netname, imname):

    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)

    pw = 65 # 33 # 65 # 33 # 65 # 33 # 65 # 33

    with open(netname) as f:
        print('Loading network')
        input_axi = T.tensor4('input_axi')
        input_sag = T.tensor4('input_sag')
        input_cor = T.tensor4('input_cor')
        input_cla = T.tensor4('input_cla')

        nclass = 3
        ## LOAD IMAGE
        image = sitk.ReadImage(imname)
        spacing = image.GetSpacing()
        image = sitk.GetArrayFromImage(image)
        image = np.swapaxes(image, 0, 2)

        orshape = image.shape

        ## STANDARD VOXEL SIZE
        voxsize = 0.6500020027160645


        ## RESHAPE IMAGE
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
        image[image<0] = 0


        # NORMALIZE IMAGE
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst


        ## INITIALIZE FEATURE TENSOR
        features = np.zeros((96, image.shape[0], image.shape[1], image.shape[2]))

        params_tmp = pickle.load(f)
        convi = params_tmp[:-11]
        convin = len(convi)/3
        final = params_tmp[-11:]

        net_axi = lasagne.layers.InputLayer(shape=(None, 1, image.shape[0]+2*pw, image.shape[1]+2*pw), input_var=input_axi)
        net_sag = lasagne.layers.InputLayer(shape=(None, 1, image.shape[1]+2*pw, image.shape[2]+2*pw), input_var=input_sag)
        net_cor = lasagne.layers.InputLayer(shape=(None, 1, image.shape[0]+2*pw, image.shape[2]+2*pw), input_var=input_cor)
        C = 32
        net_axi = getSubNet(net_axi, C=C)
        lasagne.layers.set_all_param_values(net_axi, convi[:convin])
        axi_prediction = lasagne.layers.get_output(net_axi, deterministic=True)
        axi_pred_fn = theano.function([input_axi], axi_prediction, allow_input_downcast=True)

        net_sag = getSubNet(net_sag, C=C)
        lasagne.layers.set_all_param_values(net_sag, convi[convin:2*convin])
        sag_prediction = lasagne.layers.get_output(net_sag, deterministic=True)
        sag_pred_fn = theano.function([input_sag], sag_prediction, allow_input_downcast=True)

        net_cor = getSubNet(net_cor, C=C)
        lasagne.layers.set_all_param_values(net_cor, convi[convin*2:])
        cor_prediction = lasagne.layers.get_output(net_cor, deterministic=True)
        cor_pred_fn = theano.function([input_cor], cor_prediction, allow_input_downcast=True)

        # network = lasagne.layers.ConcatLayer((net_axi, net_sag, net_cor), axis = 1)

        network = lasagne.layers.InputLayer(shape=(None, 96, image.shape[0], image.shape[1]), input_var=input_cla)
        network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.dropout(network, p=0.5)
        network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.dropout(network, p=0.5)
        network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=softmax_test, W=lasagne.init.GlorotNormal(), b=lasagne.init.Normal())
        lasagne.layers.set_all_param_values(network, final)
        cla_prediction = lasagne.layers.get_output(network, deterministic=True)
        cla_pred_fn = theano.function([input_cla], cla_prediction, allow_input_downcast=True)

        ## FILL FEATURE VECTOR
        print('Axial')
        for iz in range(image.shape[2]):
            print(iz)
            slice = np.squeeze(image[:, :, iz])
            # print(slice.shape)
            slice = np.pad(slice, pw, 'constant', constant_values=(0,)).astype('float32') # 65
            batch = np.zeros((1, 1, image.shape[0]+2*pw, image.shape[1]+2*pw))
            batch[0, 0, :, :] = slice
            out = axi_pred_fn(floatX(batch))
            # print(out.shape)
            # print(out[0].shape)
            features[0:32, :, :, iz] = out[0]
        print('Sagittal')
        for ix in range(image.shape[0]):
            print(ix)
            slice = np.squeeze(image[ix, :, :])
            # print(slice.shape)
            slice = np.pad(slice, pw, 'constant', constant_values=(0,)).astype('float32') # 65
            batch = np.zeros((1, 1, image.shape[1]+2*pw, image.shape[2]+2*pw))
            batch[0, 0, :, :] = slice
            out = sag_pred_fn(floatX(batch))
            # print(out.shape)
            # print(out[0].shape)
            features[32:64, ix, :, :] = out[0]
        print('Coronal')
        for iy in range(image.shape[1]):
            print(iy)
            slice = np.squeeze(image[:, iy, :])
            # print(slice.shape)
            slice = np.pad(slice, pw, 'constant', constant_values=(0,)).astype('float32') # 65
            batch = np.zeros((1, 1, image.shape[0]+2*pw, image.shape[2]+2*pw))
            batch[0, 0, :, :] = slice
            out = cor_pred_fn(floatX(batch))
            # print(out.shape)
            # print(out[0].shape)
            features[64:, :, iy, :] = out[0]


        ## INITIALIZE  OUTPUT
        outim_bg = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
        outim_myo = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
        outim_blood = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
        outim_class = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
        print('Classify')
        for iz in range(image.shape[2]):
            print(iz)
            slice = np.squeeze(features[:, :, :, iz])
            print(slice.shape)
            batch = np.zeros((1, 96, image.shape[0], image.shape[1]))
            batch[0, :, :, :] = slice
            out = cla_pred_fn(floatX(batch))
            print(out.shape)
            print(out[0].shape)
            out = out[0]
            outim_bg[:, :, iz] = np.squeeze(out[0, :, :])
            outim_myo[:, :, iz] = np.squeeze(out[1, :, :])
            outim_blood[:, :, iz] = np.squeeze(out[2, :, :])

        outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')
        outim_bg = scndy.zoom(outim_bg, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        outim = np.swapaxes(outim_bg, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        # outim = sitk.Cast(outim, sitk.sitkInt8)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_bg.mhd'), True)

        outim_myo = scndy.zoom(outim_myo, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        outim = np.swapaxes(outim_myo, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        # outim = sitk.Cast(outim, sitk.sitkInt8)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_myo.mhd'), True)

        outim_blood = scndy.zoom(outim_blood, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        outim = np.swapaxes(outim_blood, 0, 2)
        # outim = sitk.Cast(outim, sitk.sitkInt8)
        outim = sitk.GetImageFromArray(outim)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_blood.mhd'), True)


        # outim_bg = np.swapaxes(outim_bg, 0, 2)
        # outim_myo = np.swapaxes(outim_myo, 0, 2)
        # outim_blood = np.swapaxes(outim_blood, 0, 2)

        outim_class = np.zeros((orshape[0], orshape[1], orshape[2]), dtype='float32')

        for iz in range(orshape[2]):
            outtmp = np.zeros((3, orshape[0], orshape[1]))
            outtmp[0,:,:] = np.squeeze(outim_bg[:,:,iz])
            outtmp[1,:,:] = np.squeeze(outim_myo[:,:,iz])
            outtmp[2,:,:] = np.squeeze(outim_blood[:,:,iz])
            outim_class[:, :, iz] = np.squeeze(np.argmax(outtmp, axis=0))

        outim = np.swapaxes(outim_class, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        outim = sitk.Cast(outim, sitk.sitkInt8)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename, True)







def test25D(netname, imname):
    input_axi = T.tensor4('input_axi')
    input_sag = T.tensor4('input_sag')
    input_cor = T.tensor4('input_cor')

    nclass = 3

    image, spacing = cnu.load_mhd_to_npy(imname)
    outim = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')

    # 33, 65, 129
    image = np.pad(image, 33, 'constant', constant_values=(0,)).astype('float32') # 65

    network = getNetworkDilated25D(input_axi=input_axi, input_sag=input_sag, input_cor=input_cor, input_axi_shape=(None, 1, 67, 67), input_sag_shape=(None, 1, 67, 67), input_cor_shape=(None, 1, 67, 67), nclass=nclass)
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_axi, input_sag, input_cor], test_prediction, allow_input_downcast=True)

    for iz in range(50,51):
        for iy in range(0, outim.shape[1], 1): # 160, 260): # outim.shape[1]): # 180, 220):
            print(iy)
            for ix in range(0, outim.shape[0], 1): #160, 260):# outim.shape[0]): # 180, 260):
                batch_axi = np.zeros((1,1,67,67))
                batch_axi[0, 0, :, :] = np.squeeze(image[ix:ix+67,iy:iy+67,iz+33])
                # print(np.mean(batch_axi))
                batch_sag = np.zeros((1,1,67,67))
                batch_sag[0, 0, :, :] = np.squeeze(image[ix+33,iy:iy+67,iz:iz+67])
                # print(np.mean(batch_sag))
                batch_cor = np.zeros((1,1,67,67))
                batch_cor[0, 0, :, :] = np.squeeze(image[ix:ix+67,iy+33,iz:iz+67])
                # print(np.mean(batch_cor))

                out = pred_fn(floatX(batch_axi), floatX(batch_sag), floatX(batch_cor))
                # print(out)
                outim[ix, iy, iz] = out[0][2] # 0,2,0,0]

    outim = np.swapaxes(outim, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    # outim = sitk.Cast(outim, sitk.sitkInt8)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, netname.replace('pickle', 'mhd'), True)


def test2DRotate(netname, imname):
    start = time.time()

    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)


    input_var = T.tensor4('input_var')
    if 'cardiacmr' in imname:
        nclass = 3
    else:
        nclass = 2

    image = sitk.ReadImage(imname)
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)

    # image = image[:,:,::-1]

    orshape = image.shape


    if 'cardiacmr' in imname:
        ## STANDARD VOXEL SIZE
        voxsize = 0.6500020027160645


        ## RESHAPE IMAGE
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
        image[image<0] = 0


        # NORMALIZE IMAGE
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst

    # image = image.astype('float32')/float(np.max(image))


    outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')

    setuptime = time.time()-start

    print('Image has shape ' + str(image.shape))

    # RECEPTIVE FIELD
    rf = 131 # 259 # 131 # 259 # 19 # 35 # 67 # 131 # 17 # 259 # 17 # 131 # 259
    bs = 1


    # AXIAL
    axi_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130
    netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130

    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[1]+rf-1))

    aximage = np.pad(image, (((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2), (0, 0)), 'constant', constant_values=(0,)).astype('float32')

    for z in range(image.shape[2]):
        imslice = np.squeeze(aximage[:,:,z])
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, :, :, z] += outs
        axi_outim[:, :, :, z] += outs

        imslice = imslice[:,::-1]
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, :, :, z] += outs[:,:,::-1]
        axi_outim[:, :, :, z] += outs[:,:,::-1]



    print('Image has shape ' + str(image.shape))


    # SAGITTAL
    sag_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[1] + rf-1, image.shape[2] + rf-1)) # 130
    netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130

    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[1]+rf-1, image.shape[2]+rf-1))

    sagimage = np.pad(image, ((0, 0), ((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')


    for x in range(image.shape[0]):
        imslice = np.squeeze(sagimage[x,:,:])
        # imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32') # 65
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, x, :, :] += outs
        sag_outim[:, x, :, :] += outs

        imslice = imslice[:,::-1]
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, x, :, :] += outs[:,:,::-1]
        sag_outim[:, x, :, :] += outs[:,:,::-1]





    # CORONAL
    cor_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[2] + rf-1)) # 130
    netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130

    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[2]+rf-1))


    corimage = np.pad(image, (((rf-1)/2, (rf-1)/2), (0, 0), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')

    for y in range(image.shape[1]):
        imslice = np.squeeze(corimage[:,y,:])
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, :, y, :] += outs
        cor_outim[:, :, y, :] += outs

        imslice = imslice[:,::-1]
        batch[0, 0, :, :] = imslice
        out = pred_fn((floatX(batch)))
        outs = np.squeeze(out[0])
        outim[:, :, y, :] += outs[:,:,::-1]
        cor_outim[:, :, y, :] += outs[:,:,::-1]


    toutim = outim

    print('Squeeze take class 2')

    writestart = time.time()

    outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')

    outim_bg = np.squeeze(toutim[0,:,:,:])/6.0
    if 'cardiacmr' in imname:
        outim_bg = scndy.zoom(outim_bg, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
    outim = np.swapaxes(outim_bg, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_bg.mhd'), True)

    outim_myo = np.squeeze(toutim[1,:,:,:])/6.0
    if 'cardiacmr' in imname:
        outim_myo = scndy.zoom(outim_myo, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
    outim = np.swapaxes(outim_myo, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_myo.mhd'), True)

    if nclass>2:
        outim_blood = np.squeeze(toutim[2,:,:,:])/6.0
        if 'cardiacmr' in imname:
            outim_blood = scndy.zoom(outim_blood, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        outim = np.swapaxes(outim_blood, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_blood.mhd'), True)

    outim_class = np.zeros((orshape[0], orshape[1], orshape[2]), dtype='float32')
    for iz in range(orshape[2]):
        outtmp = np.zeros((nclass, orshape[0], orshape[1]))
        outtmp[0,:,:] = np.squeeze(outim_bg[:,:,iz])
        outtmp[1,:,:] = np.squeeze(outim_myo[:,:,iz])
        if nclass>2:
            outtmp[2,:,:] = np.squeeze(outim_blood[:,:,iz])
        outim_class[:, :, iz] = np.squeeze(np.argmax(outtmp, axis=0))

    outim = np.swapaxes(outim_class, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim = sitk.Cast(outim, sitk.sitkInt8)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename, True)
    print('Processing this ' + str(outim_class.shape) + ' scan took ' + str(time.time()-start) + 's')
    print('Setup took ' + str(setuptime) + 's')
    print('Writing took ' + str(time.time()-writestart) + 's')


def test2DNew(netname, imname):
    start = time.time()

    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)


    input_var = T.tensor4('input_var')
    if 'cardiacmr' in imname:
        nclass = 3
    else:
        nclass = 2

    image = sitk.ReadImage(imname)
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)

    # image = image[:,:,::-1]

    orshape = image.shape


    if 'cardiacmr' in imname:
        ## STANDARD VOXEL SIZE
        voxsize = 0.6500020027160645


        ## RESHAPE IMAGE
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
        image[image<0] = 0


        # NORMALIZE IMAGE
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst

    # image = image.astype('float32')/float(np.max(image))
    # Crop for now
    # image = image[:71, :71, : 71]

    outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')

    setuptime = time.time()-start

    print('Image has shape ' + str(image.shape))

    # RECEPTIVE FIELD
    rf = 133 # 131 # 67 # 131 # 259 # 131 # 259 # 19 # 35 # 67 # 131 # 17 # 259 # 17 # 131 # 259
    bs = 10


    # AXIAL
    axi_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    netfirst, network = getNetworkDilatedCascadeNew(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1), inputx=image.shape[0] + rf-1, inputy=image.shape[1] + rf-1) # 130
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)

    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[1]+rf-1))

    aximage = np.pad(image, (((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2), (0, 0)), 'constant', constant_values=(0,)).astype('float32')

    for z in range(image.shape[2]):
        imslice = np.squeeze(aximage[:,:,z])
        # imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32')
        batch[z%bs, 0, :, :] = imslice
        if (z+1)%bs==0 or z==image.shape[2]-1:
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (z+1)-bs+bi < image.shape[2]:
                    outs = np.squeeze(out[bi])
                    outim[:, :, :, (z+1)-bs+bi] += outs
                    axi_outim[:, :, :, (z+1)-bs+bi] += outs
    #
    # toutim = np.squeeze(axi_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_axi_myo.mhd'), True)
    #
    # toutim = np.squeeze(axi_outim[2,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_axi_blood.mhd'), True)

    print('Image has shape ' + str(image.shape))


    # SAGITTAL
    sag_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    netfirst, network = getNetworkDilatedCascadeNew(input_var, nclass=nclass, input_shape=(None, 1, image.shape[1] + rf-1, image.shape[2] + rf-1), inputx=image.shape[1] + rf-1, inputy=image.shape[2] + rf-1) # 130
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[1] + rf-1, image.shape[2] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[1]+rf-1, image.shape[2]+rf-1))

    sagimage = np.pad(image, ((0, 0), ((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')


    for x in range(image.shape[0]):
        imslice = np.squeeze(sagimage[x,:,:])
        # imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32') # 65
        batch[x%bs, 0, :, :] = imslice
        if (x+1)%bs==0 or x==image.shape[0]-1:
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (x+1)-bs+bi < image.shape[0]:
                    outs = np.squeeze(out[bi])
                    outim[:, (x+1)-bs+bi, :, :] += outs
                    sag_outim[:, (x+1)-bs+bi, :, :] += outs


        # # batch = np.zeros((1, 1, imslice.shape[0], imslice.shape[1]))
        # batch[0, 0, :, :] = imslice
        # print(np.mean(batch))
        # out = pred_fn((floatX(batch)))
        # print(out.shape)
        # print(out[0].shape)
        # out = np.squeeze(out[0])
        # # out = np.squeeze(out[2,:,:])
        # outim[:, x, :, :] += out
        # sag_outim[:, x, :, :] += out

    # toutim = np.squeeze(sag_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_sag_myo.mhd'), True)
    #
    # toutim = np.squeeze(sag_outim[2,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_sag_blood.mhd'), True)



    # CORONAL
    cor_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    netfirst, network = getNetworkDilatedCascadeNew(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[2] + rf-1), inputx=image.shape[0] + rf-1, inputy=image.shape[2] + rf-1) # 130
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[2] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[2]+rf-1))


    corimage = np.pad(image, (((rf-1)/2, (rf-1)/2), (0, 0), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')

    for y in range(image.shape[1]):
        imslice = np.squeeze(corimage[:,y,:])
         #imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32') # 65
        batch[y%bs, 0, :, :] = imslice
        if (y+1)%bs==0 or y==image.shape[1]-1:
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (y+1)-bs+bi < image.shape[1]:
                    outs = np.squeeze(out[bi])
                    outim[:, :, (y+1)-bs+bi, :] += outs
                    cor_outim[:, :, (y+1)-bs+bi, :] += outs


        # batch = np.zeros((1, 1, imslice.shape[0], imslice.shape[1]))
        # batch[0, 0, :, :] = imslice
        # print(np.mean(batch))
        # out = pred_fn((floatX(batch)))
        # print(out.shape)
        # print(out[0].shape)
        # out = np.squeeze(out[0])
        # # out = np.squeeze(out[2,:,:])
        # outim[:, :, y, :] += out
        # cor_outim[:, :, y, :] += out
    #
    # toutim = np.squeeze(cor_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_cor_myo.mhd'), True)
    #
    # toutim = np.squeeze(cor_outim[2,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_cor_blood.mhd'), True)

    toutim = outim

    print('Squeeze take class 2')

    writestart = time.time()

    outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')

    outim_bg = np.squeeze(toutim[0,:,:,:])/3.0
    if 'cardiacmr' in imname:
        outim_bg = scndy.zoom(outim_bg, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
    outim = np.swapaxes(outim_bg, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_bg.mhd'), True)

    outim_myo = np.squeeze(toutim[1,:,:,:])/3.0
    if 'cardiacmr' in imname:
        outim_myo = scndy.zoom(outim_myo, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)


    maxmyo = 0.0
    maxblo = 0.0

    # # PRECISION_RECALL
    # ref = sitk.ReadImage(imname.replace('images', 'reference'))
    # ref = sitk.GetArrayFromImage(ref)
    # ref = np.swapaxes(ref, 0, 2)
    # ref = (ref==1).astype('bool').flatten()
    # precision, recall, thresholds = precision_recall_curve(ref, outim_myo.flatten())
    # f1scores = 2*(precision*recall)/(precision+recall)
    # f1scores[np.isnan(f1scores)] = 0
    # print('Myocardium')
    # print('Maximum Dice ' + str(np.max(f1scores)))
    # print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
    # maxmyo = np.max(f1scores)
    # print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))

    outim = np.swapaxes(outim_myo, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_myo.mhd'), True)



    if nclass>2:
        outim_blood = np.squeeze(toutim[2,:,:,:])/3.0
        if 'cardiacmr' in imname:
            outim_blood = scndy.zoom(outim_blood, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)


        # # PRECISION_RECALL
        # ref = sitk.ReadImage(imname.replace('images', 'reference'))
        # ref = sitk.GetArrayFromImage(ref)
        # ref = np.swapaxes(ref, 0, 2)
        # ref = (ref==2).astype('bool').flatten()
        # precision, recall, thresholds = precision_recall_curve(ref, outim_blood.flatten())
        # f1scores = 2*(precision*recall)/(precision+recall)
        # f1scores[np.isnan(f1scores)] = 0
        # print('Blood pool')
        # print('Maximum Dice ' + str(np.max(f1scores)))
        # print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
        # maxblo = np.max(f1scores)
        #
        # print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))


        outim = np.swapaxes(outim_blood, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_blood.mhd'), True)

    outim_class = np.zeros((orshape[0], orshape[1], orshape[2]), dtype='float32')
    for iz in range(orshape[2]):
        outtmp = np.zeros((nclass, orshape[0], orshape[1]))
        outtmp[0,:,:] = np.squeeze(outim_bg[:,:,iz])
        outtmp[1,:,:] = np.squeeze(outim_myo[:,:,iz])
        if nclass>2:
            outtmp[2,:,:] = np.squeeze(outim_blood[:,:,iz])
        outim_class[:, :, iz] = np.squeeze(np.argmax(outtmp, axis=0))

    outim = np.swapaxes(outim_class, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim = sitk.Cast(outim, sitk.sitkInt8)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename, True)
    print('Processing this ' + str(outim_class.shape) + ' scan took ' + str(time.time()-start) + 's')
    print('Setup took ' + str(setuptime) + 's')
    print('Writing took ' + str(time.time()-writestart) + 's')

    return maxmyo, maxblo

def filt2D(netname, imname):
    print('Testing with ' + netname)
    start = time.time()

    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)


    input_var = T.tensor4('input_var')
    if 'cardiacmr' in imname:
        nclass = 3
    else:
        nclass = 2

    image = sitk.ReadImage(imname)
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)

    orshape = image.shape


    if 'cardiacmr' in imname:
        ## STANDARD VOXEL SIZE
        voxsize = 0.6500020027160645

        ## RESHAPE IMAGE
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
        image[image<0] = 0

        # NORMALIZE IMAGE
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst


    # RECEPTIVE FIELD
    rf = 131
    bs = 1

    network = getNetworkDilatedFirstLayer(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp[:8])

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[1]+rf-1))
    aximage = np.pad(image, (((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2), (0, 0)), 'constant', constant_values=(0,)).astype('float32')

    imslice = np.squeeze(aximage[:,:,169])
    batch[0, 0, :, :] = imslice
    out = pred_fn((floatX(batch)))
    print(out.shape)
    print(out[0].shape)

    np.save(netdir+os.path.sep+imbase.replace('.nii', '.npy'), out[0])


def test2D(netname, imname):
    print('Testing with ' + netname)
    start = time.time()

    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)


    input_var = T.tensor4('input_var')
    if 'cardiacmr' in imname:
        nclass = 3
    else:
        nclass = 2

    image = sitk.ReadImage(imname)
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)

    # image = image[:,:,::-1]

    orshape = image.shape


    if 'cardiacmr' in imname:
        ## STANDARD VOXEL SIZE
        voxsize = 0.6500020027160645



        ## RESHAPE IMAGE
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
        image[image<0] = 0

        ## SECOND
        # image = scndy.zoom(image, (0.5, 0.5, 0.5))
        # image[image<0] = 0

        # NORMALIZE IMAGE
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst

        # image = image + 0.4

    # image = image.astype('float32')/float(np.max(image))


    outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')

    setuptime = time.time()-start

    # print('Image has shape ' + str(image.shape))

    # RECEPTIVE FIELD
    rf = 131 # 67 # 131 # 259 # 131 # 259 # 19 # 35 # 67 # 131 # 17 # 259 # 17 # 131 # 259
    bs = 1


    # AXIAL
    axi_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130
    network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[1] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)

    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[1]+rf-1))
    # print(image.shape)
    aximage = np.pad(image, (((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2), (0, 0)), 'constant', constant_values=(0,)).astype('float32')

    for z in range(image.shape[2]):
        imslice = np.squeeze(aximage[:,:,z])
        # imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32')
        batch[z%bs, 0, :, :] = imslice
        if (z+1)%bs==0 or z==image.shape[2]-1:
            # print(batch.shape)
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (z+1)-bs+bi < image.shape[2]:
                    outs = np.squeeze(out[bi])
                    outim[:, :, :, (z+1)-bs+bi] += outs
                    axi_outim[:, :, :, (z+1)-bs+bi] += outs
    #
    # toutim = np.squeeze(axi_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_axi_myo.mhd'), True)
    #
    toutim = np.squeeze(axi_outim[2,:,:,:])
    toutim[toutim<0.01] = 0.0
    toutim = np.swapaxes(toutim, 0, 2)
    toutim = sitk.GetImageFromArray(toutim)
    toutim.SetSpacing(spacing)
    sitk.WriteImage(toutim, netname.replace('.pickle', '_axi_blood.mhd'), True)

    # print('Image has shape ' + str(image.shape))


    # SAGITTAL
    sag_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[1] + rf-1, image.shape[2] + rf-1)) # 130
    network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[1] + rf-1, image.shape[2] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[1]+rf-1, image.shape[2]+rf-1))

    sagimage = np.pad(image, ((0, 0), ((rf-1)/2, (rf-1)/2), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')


    for x in range(image.shape[0]):
        imslice = np.squeeze(sagimage[x,:,:])
        # imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32') # 65
        batch[x%bs, 0, :, :] = imslice
        if (x+1)%bs==0 or x==image.shape[0]-1:
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (x+1)-bs+bi < image.shape[0]:
                    outs = np.squeeze(out[bi])
                    outim[:, (x+1)-bs+bi, :, :] += outs
                    sag_outim[:, (x+1)-bs+bi, :, :] += outs


        # # batch = np.zeros((1, 1, imslice.shape[0], imslice.shape[1]))
        # batch[0, 0, :, :] = imslice
        # print(np.mean(batch))
        # out = pred_fn((floatX(batch)))
        # print(out.shape)
        # print(out[0].shape)
        # out = np.squeeze(out[0])
        # # out = np.squeeze(out[2,:,:])
        # outim[:, x, :, :] += out
        # sag_outim[:, x, :, :] += out

    # toutim = np.squeeze(sag_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_sag_myo.mhd'), True)
    #
    toutim = np.squeeze(sag_outim[2,:,:,:])
    toutim[toutim<0.01] = 0.0
    toutim = np.swapaxes(toutim, 0, 2)
    toutim = sitk.GetImageFromArray(toutim)
    toutim.SetSpacing(spacing)
    sitk.WriteImage(toutim, netname.replace('.pickle', '_sag_blood.mhd'), True)



    # CORONAL
    cor_outim = np.zeros((nclass, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    # netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[2] + rf-1)) # 130
    network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0] + rf-1, image.shape[2] + rf-1)) # 130
    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)
        # lasagne.layers.set_all_param_values(netfirst, params_tmp[:26])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_prediction = lasagne.layers.get_output(netfirst, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    batch = np.zeros((bs, 1, image.shape[0]+rf-1, image.shape[2]+rf-1))


    corimage = np.pad(image, (((rf-1)/2, (rf-1)/2), (0, 0), ((rf-1)/2, (rf-1)/2)), 'constant', constant_values=(0,)).astype('float32')

    for y in range(image.shape[1]):
        imslice = np.squeeze(corimage[:,y,:])
         #imslice = np.pad(imslice, (rf-1)/2, 'constant', constant_values=(0,)).astype('float32') # 65
        batch[y%bs, 0, :, :] = imslice
        if (y+1)%bs==0 or y==image.shape[1]-1:
            out = pred_fn((floatX(batch)))
            # print(out.shape)
            # print(out[0].shape)
            for bi in range(bs):
                if (y+1)-bs+bi < image.shape[1]:
                    outs = np.squeeze(out[bi])
                    outim[:, :, (y+1)-bs+bi, :] += outs
                    cor_outim[:, :, (y+1)-bs+bi, :] += outs


        # batch = np.zeros((1, 1, imslice.shape[0], imslice.shape[1]))
        # batch[0, 0, :, :] = imslice
        # print(np.mean(batch))
        # out = pred_fn((floatX(batch)))
        # print(out.shape)
        # print(out[0].shape)
        # out = np.squeeze(out[0])
        # # out = np.squeeze(out[2,:,:])
        # outim[:, :, y, :] += out
        # cor_outim[:, :, y, :] += out
    #
    # toutim = np.squeeze(cor_outim[1,:,:,:])
    # toutim[toutim<0.01] = 0.0
    # toutim = np.swapaxes(toutim, 0, 2)
    # toutim = sitk.GetImageFromArray(toutim)
    # toutim.SetSpacing(spacing)
    # sitk.WriteImage(toutim, netname.replace('.pickle', '_cor_myo.mhd'), True)
    #
    toutim = np.squeeze(cor_outim[2,:,:,:])
    toutim[toutim<0.01] = 0.0
    toutim = np.swapaxes(toutim, 0, 2)
    toutim = sitk.GetImageFromArray(toutim)
    toutim.SetSpacing(spacing)
    sitk.WriteImage(toutim, netname.replace('.pickle', '_cor_blood.mhd'), True)

    toutim = outim


    outim_myo = 0.0
    outim_blood = 0.0
    print('Squeeze take class 2')

    writestart = time.time()

    outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')

    outim_bg = np.squeeze(toutim[0,:,:,:])/3.0
    if 'cardiacmr' in imname:
        outim_bg = scndy.zoom(outim_bg, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        # outim_bg = scndy.zoom(outim_bg, (2.0, 2.0, 2.0), order=1)


    outim = np.swapaxes(outim_bg, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_bg.mhd'), True)

    outim_myo = np.squeeze(toutim[1,:,:,:])/3.0
    if 'cardiacmr' in imname:
        outim_myo = scndy.zoom(outim_myo, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        # outim_myo = scndy.zoom(outim_myo, (2.0, 2.0, 2.0), order=1)



    # PRECISION_RECALL
    # ref = sitk.ReadImage(imname.replace('images', 'reference'))
    # ref = sitk.GetArrayFromImage(ref)
    # ref = np.swapaxes(ref, 0, 2)
    # ref = (ref==1).astype('bool').flatten()
    # precision, recall, thresholds = precision_recall_curve(ref, outim_myo.flatten())
    # f1scores = 2*(precision*recall)/(precision+recall)
    # f1scores[np.isnan(f1scores)] = 0
    # print('Myocardium')
    # maxmyo = np.max(f1scores)
    # print('Maximum Dice ' + str(np.max(f1scores)))
    # print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
    # print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))

    outim = np.swapaxes(outim_myo, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd', '_myo.mhd'), True)



    if nclass>2:
        outim_blood = np.squeeze(toutim[2,:,:,:])/3.0
        if 'cardiacmr' in imname:
            outim_blood = scndy.zoom(outim_blood, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
            # outim_blood = scndy.zoom(outim_blood, (2.0, 2.0, 2.0), order=1)


        # PRECISION_RECALL
        # ref = sitk.ReadImage(imname.replace('images', 'reference'))
        # ref = sitk.GetArrayFromImage(ref)
        # ref = np.swapaxes(ref, 0, 2)
        # ref = (ref==2).astype('bool').flatten()
        # precision, recall, thresholds = precision_recall_curve(ref, outim_blood.flatten())
        # f1scores = 2*(precision*recall)/(precision+recall)
        # f1scores[np.isnan(f1scores)] = 0
        # print('Blood pool')
        # maxblo = np.max(f1scores)
        # print('Maximum Dice ' + str(np.max(f1scores)))
        # print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
        # print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))


        outim = np.swapaxes(outim_blood, 0, 2)
        outim = sitk.GetImageFromArray(outim)
        outim.SetSpacing(spacing)
        sitk.WriteImage(outim, outbasename.replace('.mhd', '_blood.mhd'), True)

    outim_class = np.zeros((orshape[0], orshape[1], orshape[2]), dtype='float32')
    for iz in range(orshape[2]):
        outtmp = np.zeros((nclass, orshape[0], orshape[1]))
        outtmp[0,:,:] = np.squeeze(outim_bg[:,:,iz])
        outtmp[1,:,:] = np.squeeze(outim_myo[:,:,iz])
        if nclass>2:
            outtmp[2,:,:] = np.squeeze(outim_blood[:,:,iz])
        outim_class[:, :, iz] = np.squeeze(np.argmax(outtmp, axis=0))

    outim = np.swapaxes(outim_class, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim = sitk.Cast(outim, sitk.sitkInt8)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename, True)
    print('Processing this ' + str(outim_class.shape) + ' scan took ' + str(time.time()-start) + 's')
    # print('Setup took ' + str(setuptime) + 's')
    # print('Writing took ' + str(time.time()-writestart) + 's')
    return outim_myo, outim_blood

def batch3D(netname, dirname):
    filenames = glob.glob(dirname + os.path.sep + '*.nii')
    for filename in filenames:
        print(filename)
        test3D(netname, filename)


def test3D(netname, imname):
    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)
    outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')

    ftensor5 = T.TensorType('float32', (False,)*5)
    input_var = ftensor5('input_var')
    nclass = 3
    if 'brainmr' in imname:
        nclass = 7
    ps = 51 # 35


    image, spacing = cnu.load_mhd_to_npy(imname)

    ## STANDARD VOXEL SIZE
    voxsize = 0.6500020027160645


    ## RESHAPE IMAGE
    print(image.shape)
    if 'cardiacmr' in imname:
        image = scndy.zoom(image, (spacing[0]/voxsize, spacing[1]/voxsize, spacing[2]/voxsize))
    print(image.shape)
    image[image<0] = 0

    # Normalize image
    if 'cardiacmr' in imname:
        imme = np.mean(image)
        imst = np.std(image)
        image = (image-imme)/imst

    outim_bg = np.zeros(image.shape, dtype='float32')
    outim_myo = np.zeros(image.shape, dtype='float32')
    outim_blood = np.zeros(image.shape, dtype='float32')

    # Pad image
    image = np.pad(image, (ps-1)/2, 'constant', constant_values=(0,)).astype('float32')

    # CROP
    # image = image[0:-1, 0:-1, 0:-1]

    network = getNetworkDilated3D(input_var, nclass=nclass, input_shape=(None, 1, image.shape[0], image.shape[1], image.shape[2]))

    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)

    # image = np.pad(image, (ps-1)/2, 'constant', constant_values=(0,)).astype('float32')

    batch = np.zeros((1, 1, image.shape[0], image.shape[1], image.shape[2]))
    batch[0, 0, :, :, :] = image

    out = pred_fn(floatX(batch))

    print(out.shape)
    print(out[0].shape)

    out = np.squeeze(out[0])
    hs = (ps-1)/2
    # outim_bg[0+hs:-hs-1, 0+hs:-hs-1, 0+hs:-hs-1] = np.squeeze(out[0, :, :, :])
    # outim_myo[0+hs:-hs-1, 0+hs:-hs-1, 0+hs:-hs-1] = np.squeeze(out[1, :, :, :])
    # outim_blood[0+hs:-hs-1, 0+hs:-hs-1, 0+hs:-hs-1] = np.squeeze(out[2, :, :, :])


    maxout = np.zeros((out.shape[1], out.shape[2], out.shape[3]))
    for z in range(out.shape[3]):
        zslab = np.squeeze(out[:, :, :, z])
        maxout[:, :, z] = np.squeeze(np.argmax(zslab, axis=0))



    # outim_bg = np.squeeze(out[0, :, :, :])
    # outim_myo = np.squeeze(out[1, :, :, :])
    # outim_blood = np.squeeze(out[2, :, :, :])

    # print(outim_bg.shape)

    if 'cardiacmr' in imname:
        maxout = scndy.zoom(maxout, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=0)
        # outim_bg = scndy.zoom(outim_bg, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        # outim_myo = scndy.zoom(outim_myo, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)
        # outim_blood = scndy.zoom(outim_blood, (voxsize/spacing[0], voxsize/spacing[1], voxsize/spacing[2]), order=1)

    # print(outim_bg.shape)
    #
    # outim = np.zeros(outim_bg.shape, dtype='int16')
    #
    # for z in range(outim_blood.shape[2]):
    #     thrclass = np.zeros((3, outim_blood.shape[0], outim_blood.shape[1]))
    #     thrclass[0, :, :] = outim_bg[:, :, z]
    #     thrclass[1, :, :] = outim_myo[:, :, z]
    #     thrclass[2, :, :] = outim_blood[:, :, z]
    #
    #     outim[:, :, z] = np.squeeze(np.argmax(thrclass, axis=0))

    outim = np.swapaxes(maxout, 0, 2) # outim, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, outbasename.replace('.mhd','.nii'), True)



def main3DDummy(tag, task='brainmr'):
    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    ftensor5 = T.TensorType('float32', (False,)*5)
    input_var = ftensor5('input_var')
    target_values = ftensor5('labelmap')

    nclass = 3

    network = getNetworkDilated3D(input_var, nclass=nclass, input_shape=(None, 1, 101, 101, 101))
    print('Network contains ' + str(lasagne.layers.count_params(network)) + ' parameters')



    with open(r'/home/jelmer/MICCAI2016PIM/experiments/DILATE3D/6200.pickle') as f: #/home/jelmer/MICCAI2016PIM/experiments/MULTI135/12400.pickle') as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)


    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    loss = lasagne.objectives.squared_error(prediction, target_values).mean()
    updates = lasagne.updates.adam(loss, all_params)
    train_fn = theano.function([input_var, target_values], loss, updates=updates, allow_input_downcast=True) # , mask_var

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_values).mean()
    eval_fn = theano.function([input_var, target_values], test_loss, allow_input_downcast=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)

    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    if task != r'brainmr' and task != r'cardiacmr':
        filetype = r'*.mhd'

    trainimages, trainlabels = loadImageDir3D(glob.glob(traindir + os.path.sep + filetype))
    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels = loadImageDir3D(glob.glob(valdir + os.path.sep + filetype))

    startIt = -1
    num_epochs = 100000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0


    for it in range(num_epochs):
        print(it)
        images, labels = generateBatch3D(trainimages, trainlabels, nclass=nclass, nsamp=10)
        errors[it] = train_fn(floatX(images), floatX(labels))

        partbatch = np.zeros((1, 1, 101, 101, 101))
        partbatch[0, :, :, :, :] = images[0, :, :, :, :]

        out = pred_fn(floatX(partbatch))
        out = out[0]
        print(str(out.shape) + ' is out shape.')

        tmpim = sitk.GetImageFromArray(np.squeeze(out[0,:,:,:]))
        sitk.WriteImage(tmpim, expdir + os.path.sep + str(it) + '_out.mhd', True)
        tmpim = sitk.GetImageFromArray(np.squeeze(partbatch))
        sitk.WriteImage(tmpim, expdir + os.path.sep + str(it) + '_in.mhd', True)
        tmpim = sitk.GetImageFromArray(np.squeeze(labels[0, 2,:,:,:]))
        sitk.WriteImage(tmpim, expdir + os.path.sep + str(it) + '_label.mhd', True)

        print('Train ' + str(errors[it]))
        images, labels = generateBatch3D(valimages, vallabels, nclass=nclass, nsamp=10)
        testes[it] = eval_fn(floatX(images), floatX(labels))
        print('Test ' + str(testes[it]))



        t = range(it)
        plt.clf()
        plt.plot(t, np.log10(errors[:it]), label='Train loss')
        plt.plot(t, np.log10(testes[:it]), label='Validation loss')
        plt.legend(loc=3)
        figname = expdir + os.path.sep + 'error' + str(it) + '.png'
        plt.savefig(figname)
        if it % 10 == 0:
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)



def main3D(tag, task='brainmr', fold='1'):
    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    ftensor5 = T.TensorType('float32', (False,)*5)
    input_var = ftensor5('input_var')
    target_values = ftensor5('labelmap')

    nclass = 7
    if 'cta' in task or 'breastmr' in task or 'cor' in task or 'pancreas' in task:
        nclass = 2
    if 'cardiacmr' in task:
        nclass = 3

    # pw = 51 # 35 + 16
    pw = 67 # 51 + 16
    bs = 48 # 64

    network = getNetworkDilated3D(input_var, nclass=nclass, input_shape=(bs, 1, pw, pw, pw))
    print('Network contains ' + str(lasagne.layers.count_params(network)) + ' parameters')

    with open(r'/home/jelmer/MICCAI2016PIM/experiments/DILATE3DBRAIN51/8440.pickle') as f: #/home/jelmer/MICCAI2016PIM/experiments/MULTI135/12400.pickle') as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)

    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    # loss = lasagne.objectives.squared_error(prediction, target_values).mean()
    loss = categorical_crossentropy_3D(prediction, target_values).mean()
    updates = lasagne.updates.adam(loss, all_params)
    train_fn = theano.function([input_var, target_values], loss, updates=updates, allow_input_downcast=True) # , mask_var

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_loss = lasagne.objectives.squared_error(test_prediction, target_values).mean()
    test_loss = categorical_crossentropy_3D(test_prediction, target_values).mean()
    eval_fn = theano.function([input_var, target_values], test_loss, allow_input_downcast=True)
    pred_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)

    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    # if task != r'brainmr' and task != r'cardiacmr':
    #     filetype = r'*.mhd'
    print(traindir)
    trainimages, trainlabels = loadImageDir3D(glob.glob(traindir + os.path.sep + filetype))
    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels = loadImageDir3D(glob.glob(valdir + os.path.sep + filetype))

    startIt = -1
    num_epochs = 100000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0


    for it in range(num_epochs):
        print(it)
        images, labels = generateBatch3D(trainimages, trainlabels, nclass=nclass, nsamp=bs)
        if it == 0:
            np.save(expdir + os.path.sep + 'samples.npy', images)
            np.save(expdir + os.path.sep + 'labels.npy', labels)

        errors[it] = train_fn(floatX(images), floatX(labels))
        print('Train ' + str(errors[it]))
        images, labels = generateBatch3D(valimages, vallabels, nclass=nclass, nsamp=bs)
        testes[it] = eval_fn(floatX(images), floatX(labels))
        print('Test ' + str(testes[it]))
        t = range(it)
        plt.clf()
        plt.plot(t, np.log10(errors[:it]), label='Train loss')
        plt.plot(t, np.log10(testes[:it]), label='Validation loss')
        ax = plt.gca()
        # ax.set_ylim([0.0, 0.01])
        plt.legend(loc=3)
        figname = expdir + os.path.sep + 'error' + str(it) + '.png'
        plt.savefig(figname)
        if it % 10 == 0:
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)


def test25DSlabs(netname, imname):
    input_axi = T.tensor4('input_axi')
    input_sag = T.tensor4('input_sag')
    input_cor = T.tensor4('input_cor')

    nclass = 3

    image, spacing = cnu.load_mhd_to_npy(imname)

    image = image[:, :, :100]

    outim = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='float32')

    # 33, 65, 129
    padw = 65
    # image = np.pad(image, 33, 'constant', constant_values=(0,)).astype('float32') # 65

    network = getNetworkDilated25DSlabs(input_axi, input_sag, input_cor, nclass=nclass, wx=image.shape[0], wy=image.shape[1], wz=image.shape[2]) # Is 70 in 2D

    with open(netname) as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_axi, input_sag, input_cor], test_prediction, allow_input_downcast=True)

    batch_axi = np.zeros((image.shape[2], 1, image.shape[0] + 2*padw, image.shape[1] + 2*padw))
    batch_sag = np.zeros((image.shape[0], 1, image.shape[1] + 2*padw, image.shape[2] + 2*padw))
    batch_cor = np.zeros((image.shape[1], 1, image.shape[0] + 2*padw, image.shape[2] + 2*padw))

    # Fill axial batch
    for iz in range(image.shape[2]):
        batch_axi[iz, 0, :, :] = np.pad(np.squeeze(image[:, :, iz]), 65, 'constant', constant_values=(0,)).astype('float32')

    # Fill sagittal batch
    for ix in range(image.shape[0]):
        batch_sag[ix, 0, :, :] = np.pad(np.squeeze(image[ix, :, :]), 65, 'constant', constant_values=(0,)).astype('float32')

    # Fill coronal batch
    for iy in range(image.shape[1]):
        batch_cor[iy, 0, :, :] = np.pad(np.squeeze(image[:, iy, :]), 65, 'constant', constant_values=(0,)).astype('float32')

    out = pred_fn(floatX(batch_axi), floatX(batch_sag), floatX(batch_cor))
    print(out.shape)

    outim = np.squeeze(out[0, 2, :, :, :])

    outim = np.swapaxes(outim, 0, 2)
    outim = sitk.GetImageFromArray(outim)
    # outim = sitk.Cast(outim, sitk.sitkInt8)
    outim.SetSpacing(spacing)
    sitk.WriteImage(outim, netname.replace('pickle', 'mhd'), True)


def main25DSlabs(tag, task='brainmr', fold='1'):

    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    input_axi = T.tensor4('input_axi')
    input_sag = T.tensor4('input_sag')
    input_cor = T.tensor4('input_cor')

    ftensor5 = T.TensorType('float32', (False,)*5)
    target_values = ftensor5('target_values')

    nclass = 3

    # network = getNetwork(input_var, nclass=nclass, input_shape=(None, 1, imsize, imsize))
    ps = 131 # 67 # 131 # 67 # 67, 131, 259 # --> Extend a bit to 201 x 201 so we classify 70 x 70 patches
    network = getNetworkDilated25DSlabs(input_axi, input_sag, input_cor, nclass=nclass, wx=51, wy=51, wz=51) # Is 70 in 2D

    print('Network generated, I will now set the parameters.')
    # othernetwork = getNetworkConventional(input_var, nclass=nclass, input_shape=(None, 1, imsize+2*ps, imsize+2*ps)) # 66 because 67x67 receptive field, pad both sided with 33
    # print('Conventional network contains ' + str(lasagne.layers.count_params(othernetwork)) + ' parameters.')


    with open(r'/home/jelmer/MICCAI2016PIM/experiments/DILATE25DCARDIACFOLD1_131/1600.pickle') as f: #/home/jelmer/MICCAI2016PIM/experiments/MULTI135/12400.pickle') as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)

    print('Parameters set, I will now build the functions.')
    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    # loss = lasagne.objectives.binary_crossentropy(prediction, target_values)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_values).mean()
    loss = categorical_crossentropy_4D(prediction, target_values)
    # loss = lasagne.objectives.squared_error(prediction, target_values).mean()

    # My own version of categorical crossentropy
    #loss = -T.sum(target_values * T.log(prediction), axis=1)

    # loss = lasagne.objectives.aggregate(loss, mask_var, mode='normalized_sum')
    # loss = lasagne.objectives.aggregate(loss, mode='mean')

    updates = lasagne.updates.adam(loss, all_params)
    train_fn = theano.function([input_axi, input_sag, input_cor, target_values], loss, updates=updates, allow_input_downcast=True) # , mask_var

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.binary_crossentropy(prediction, target_values)
    #test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values).mean()
    test_loss = categorical_crossentropy_4D(test_prediction, target_values)
    # test_loss = lasagne.objectives.squared_error(test_prediction, target_values).mean() # categorical_crossentropy(prediction, target_values)
    #test_loss = lasagne.objectives.aggregate(loss, mask_var, mode='normalized_sum')
    #test_loss = -T.sum(target_values * T.log(test_prediction))
    # test_loss = lasagne.objectives.aggregate(test_loss, mask_var, mode='normalized_sum') #mean')

    eval_fn = theano.function([input_axi, input_sag, input_cor, target_values], test_loss, allow_input_downcast=True)
    # , mask_var
    pred_fn = theano.function([input_axi, input_sag, input_cor], test_prediction, allow_input_downcast=True)


    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    if task != r'brainmr' and task != r'cardiacmr':
        filetype = r'*.mhd'


    trainimages, trainlabels, trainindices = loadImageDir25D(glob.glob(traindir + os.path.sep + filetype))
    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels, valindices = loadImageDir25D(glob.glob(valdir + os.path.sep + filetype))

    startIt = -1
    num_epochs = 100000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0


    for it in range(num_epochs):
        print(it)
        im_axi, im_sag, im_cor, labels = generateBatch25DSlabs(trainimages, trainlabels, trainindices, nclass=nclass)
        print(labels.shape)
        for nc in range(nclass):
            print(str(np.sum(labels[0, nc, :, :, :])) + ' samples in class ' + str(nc))
        # print('Label distribution ' + str(np.squeeze(np.sum(np.squeeze(labels), axis=1))))
        preerror = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        print('Pre-update error ' + str(preerror))
        error = train_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        if it % 2 == 0:
            errors[it:it+10] = error
            print('Train ' + str(errors[it]))
            im_axi, im_sag, im_cor, labels = generateBatch25DSlabs(trainimages, trainlabels, trainindices, nclass=nclass)# valimages, vallabels, valindices, nclass=nclass, nsamp=90)
            print(labels.shape)
            for nc in range(nclass):
                print(str(np.sum(labels[0, nc, :, :, :])) + ' samples in class ' + str(nc))
            testes[it:it+2] = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
            print('Test ' + str(testes[it]))
            t = range(it)
            plt.clf()
            plt.plot(t, np.log10(errors[:it]), label='Train loss')
            plt.plot(t, np.log10(testes[:it]), label='Validation loss')
            ax = plt.gca()
            plt.legend(loc=3)
            figname = expdir + os.path.sep + 'error' + str(it) + '.png'
            plt.savefig(figname)
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)

def initializeNetworkTriplanarM(input_var_axi = None, input_var_sag = None, input_var_cor = None, pw = 23, ks = 3, input_shape=(None, 1, 1, 1), nclass = 2):
    bn = True
    print('Initializing network with ' + str(nclass) + ' classes.')
    print('Kernel size ' + str(ks))
    if input_shape == (None, 1, 1, 1):
        input_shape = (None, 1, pw, pw)

    nfilt = 32

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    net_axi = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var_axi)
    net_sag = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var_sag)
    net_cor = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var_cor)

    print(pw)
    while pw > 1:
        net_axi = lasagne.layers.Conv2DLayer(net_axi, num_filters=nfilt, filter_size=(ks, ks), nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.GlorotNormal())
        net_sag = lasagne.layers.Conv2DLayer(net_sag, num_filters=nfilt, filter_size=(ks, ks), nonlinearity=lasagne.nonlinearities.elu,W=net_axi.W, b=net_axi.b)
        net_cor = lasagne.layers.Conv2DLayer(net_cor, num_filters=nfilt, filter_size=(ks, ks), nonlinearity=lasagne.nonlinearities.elu,W=net_axi.W, b=net_axi.b)
        if bn:
            net_axi = lasagne.layers.BatchNormLayer(net_axi)
            net_sag = lasagne.layers.BatchNormLayer(net_sag, beta=net_axi.beta, gamma=net_axi.gamma, mean=net_axi.mean, inv_std=net_axi.inv_std)
            net_cor = lasagne.layers.BatchNormLayer(net_cor, beta=net_axi.beta, gamma=net_axi.gamma, mean=net_axi.mean, inv_std=net_axi.inv_std)

        pw = pw - (ks-1)
        print(pw)
        print(nfilt)

    # Concatenate axial, sagittal and coronal layers to 96 features
    network = lasagne.layers.ConcatLayer((net_axi, net_sag, net_cor), axis = 1)
    if bn:
        network = lasagne.layers.batch_norm(network)

    network = lasagne.layers.dropout(network, p = 0.5)

    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal())

    if bn:
        network = lasagne.layers.batch_norm(network)

    network = lasagne.layers.dropout(network, p = 0.5)

    network = lasagne.layers.Conv2DLayer(network, num_filters=max((192, 5 * nclass)), filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.GlorotNormal())
    if bn:
        network = lasagne.layers.batch_norm(network)

    network = lasagne.layers.dropout(network, p = 0.5)

    network = lasagne.layers.Conv2DLayer(network, num_filters=nclass, filter_size=(1,1), nonlinearity=fcsoftmax, W=lasagne.init.GlorotNormal())

    return network, net_axi, net_sag, net_cor



def main25DBaseline(tag, task='brainmr', fold='1'):

    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    input_axi = T.tensor4('input_axi')
    input_sag = T.tensor4('input_sag')
    input_cor = T.tensor4('input_cor')

    target_values = T.ivector('targets')

    nclass = 3

    network, net_axi, net_sag, net_cor = initializeNetworkTriplanarM(input_var_axi = input_axi, input_var_sag = input_sag, input_var_cor = input_cor, pw = 51, ks = 3, input_shape=(None, 1, 1, 1), nclass = nclass)


    print('Network generated, I will now set the parameters.')


    print('Parameters set, I will now build the functions.')
    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_values).mean()


    updates = lasagne.updates.adam(loss, all_params)
    # updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=0.001)
    train_fn = theano.function([input_axi, input_sag, input_cor, target_values], loss, updates=updates, allow_input_downcast=True) # , mask_var

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values).mean()


    eval_fn = theano.function([input_axi, input_sag, input_cor, target_values], test_loss, allow_input_downcast=True)
    # , mask_var
    pred_fn = theano.function([input_axi, input_sag, input_cor], test_prediction, allow_input_downcast=True)


    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    if task != r'brainmr' and task != r'cardiacmr':
        filetype = r'*.mhd'


    trainimages, trainlabels, trainindices = loadImageDir25D(glob.glob(traindir + os.path.sep + filetype))
    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels, valindices = loadImageDir25D(glob.glob(valdir + os.path.sep + filetype))

    startIt = -1
    num_epochs = 100000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0


    for it in range(num_epochs):
        print(it)
        im_axi, im_sag, im_cor, labels = generateBatch25D(trainimages, trainlabels, trainindices, nclass=nclass, nsamp=180)
        print('Label distribution ' + str(np.sum(np.squeeze(labels), axis=0)))
        preerror = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        print('Pre-update error ' + str(preerror))
        error = train_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        if it % 10 == 0:
            errors[it:it+10] = error
            print('Train ' + str(errors[it]))
            im_axi, im_sag, im_cor, labels = generateBatch25D(valimages, vallabels, valindices, nclass=nclass, nsamp=180)# valimages, vallabels, valindices, nclass=nclass, nsamp=90)
            print('Label distribution ' + str(np.sum(np.squeeze(labels), axis=0)))
            testes[it:it+10] = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
            print('Test ' + str(testes[it]))
            t = range(it)
            plt.clf()
            plt.plot(t, np.log10(errors[:it]), label='Train loss')
            plt.plot(t, np.log10(testes[:it]), label='Validation loss')
            ax = plt.gca()
            plt.legend(loc=3)
            figname = expdir + os.path.sep + 'error' + str(it) + '.png'
            plt.savefig(figname)
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)







def main25D(tag, task='brainmr', fold='1'):

    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    input_axi = T.tensor4('input_axi')
    input_sag = T.tensor4('input_sag')
    input_cor = T.tensor4('input_cor')

    target_values = T.ivector('targets')

    nclass = 3

    # network = getNetwork(input_var, nclass=nclass, input_shape=(None, 1, imsize, imsize))
    ps = 131 # 67 # 131 # 67 # 131 # 67 # 67, 131, 259 # --> Extend a bit to 201 x 201 so we classify 70 x 70 patches
    network, net_axi, net_sag, net_cor = getNetworkDilated25D(input_axi, input_sag, input_cor, input_axi_shape=(None, 1, ps, ps), input_sag_shape=(None, 1, ps, ps), input_cor_shape=(None, 1, ps, ps), nclass=nclass)



    print('Network generated, I will now set the parameters.')


    # othernetwork = getNetworkConventional(input_var, nclass=nclass, input_shape=(None, 1, imsize+2*ps, imsize+2*ps)) # 66 because 67x67 receptive field, pad both sided with 33
    # print('Conventional network contains ' + str(lasagne.layers.count_params(othernetwork)) + ' parameters.')


    # with open(r'/home/jelmer/MICCAI2016PIM/experiments/DILATE2DCARDIACFOLD2_131_NORMALIZED/5000.pickle') as f: #/home/jelmer/MICCAI2016PIM/experiments/MULTI135/12400.pickle') as f:
    #     print('Loading network')
    #     params_tmp = pickle.load(f)
    #     print(len(params_tmp))
    #     for p in params_tmp:
    #         print(p.shape)
    #     # lasagne.layers.set_all_param_values(network, params_tmp)
    #     lasagne.layers.set_all_param_values(net_axi, params_tmp[:16])
    #     lasagne.layers.set_all_param_values(net_sag, params_tmp[:16])
    #     lasagne.layers.set_all_param_values(net_cor, params_tmp[:16])

    with open(r'/home/jelmer/MICCAI2016PIM/experiments/DILATE25DCARDIACFOLD1_131_NORMALIZED/220.pickle') as f: #/home/jelmer/MICCAI2016PIM/experiments/MULTI135/12400.pickle') as f:
        print('Loading network')
        params_tmp = pickle.load(f)
        lasagne.layers.set_all_param_values(network, params_tmp)


    print('Parameters set, I will now build the functions.')
    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    # loss = lasagne.objectives.binary_crossentropy(prediction, target_values)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_values).mean()
    # loss = lasagne.objectives.squared_error(prediction, target_values).mean()

    # My own version of categorical crossentropy
    #loss = -T.sum(target_values * T.log(prediction), axis=1)

    # loss = lasagne.objectives.aggregate(loss, mask_var, mode='normalized_sum')
    # loss = lasagne.objectives.aggregate(loss, mode='mean')

    updates = lasagne.updates.adam(loss, all_params)
    # updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=0.001)
    train_fn = theano.function([input_axi, input_sag, input_cor, target_values], loss, updates=updates, allow_input_downcast=True) # , mask_var

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.binary_crossentropy(prediction, target_values)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values).mean()
    # test_loss = lasagne.objectives.squared_error(test_prediction, target_values).mean() # categorical_crossentropy(prediction, target_values)
    #test_loss = lasagne.objectives.aggregate(loss, mask_var, mode='normalized_sum')
    #test_loss = -T.sum(target_values * T.log(test_prediction))
    # test_loss = lasagne.objectives.aggregate(test_loss, mask_var, mode='normalized_sum') #mean')

    eval_fn = theano.function([input_axi, input_sag, input_cor, target_values], test_loss, allow_input_downcast=True)
    # , mask_var
    pred_fn = theano.function([input_axi, input_sag, input_cor], test_prediction, allow_input_downcast=True)


    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    if task != r'brainmr' and task != r'cardiacmr':
        filetype = r'*.mhd'


    trainimages, trainlabels, trainindices = loadImageDir25D(glob.glob(traindir + os.path.sep + filetype))
    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels, valindices = loadImageDir25D(glob.glob(valdir + os.path.sep + filetype))

    startIt = -1
    num_epochs = 100000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0


    for it in range(num_epochs):
        print(it)
        im_axi, im_sag, im_cor, labels = generateBatch25D(trainimages, trainlabels, trainindices, nclass=nclass, nsamp=180)
        print('Label distribution ' + str(np.sum(np.squeeze(labels), axis=0)))
        preerror = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        print('Pre-update error ' + str(preerror))
        error = train_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
        if it % 10 == 0:
            errors[it:it+10] = error
            print('Train ' + str(errors[it]))
            im_axi, im_sag, im_cor, labels = generateBatch25D(valimages, vallabels, valindices, nclass=nclass, nsamp=180)# valimages, vallabels, valindices, nclass=nclass, nsamp=90)
            print('Label distribution ' + str(np.sum(np.squeeze(labels), axis=0)))
            testes[it:it+10] = eval_fn(floatX(im_axi), floatX(im_sag),floatX(im_cor), floatX(labels))
            print('Test ' + str(testes[it]))
            t = range(it)
            plt.clf()
            plt.plot(t, np.log10(errors[:it]), label='Train loss')
            plt.plot(t, np.log10(testes[:it]), label='Validation loss')
            ax = plt.gca()
            plt.legend(loc=3)
            figname = expdir + os.path.sep + 'error' + str(it) + '.png'
            plt.savefig(figname)
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)

# Computes the categorical crossentropy for two 3D tensors
# Dimensions are x, y, classes
def categorical_crossentropy_4D(coding_dist, true_dist):
    return (-T.sum(true_dist * T.log(coding_dist), axis=2)).mean()


def categorical_crossentropy_3D(coding_dist, true_dist):
    # return (-T.sum(true_dist * T.log(coding_dist), axis=2)) # .mean()
    return (-T.sum(true_dist * T.log(coding_dist), axis=1)).mean()


def main2D(tag, task='brainmr', fold='1'):

    print('Experiment tag ' + tag)
    expdir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep + 'experiments' + os.path.sep + tag
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    input_var = T.tensor4('input')
    #target_one = T.tensor4('label_one')
    #target_two = T.tensor4('label_two')
    target = T.tensor4('labels')
    mask_var = T.tensor4('mask')

    nclass = 7
    if 'cta' in task or 'breastmr' in task or 'cor' in task or 'pancreas' in task:
        nclass = 2

    if 'cardiacmr' in task:
        nclass = 3



    # network = getNetwork(input_var, nclass=nclass, input_shape=(None, 1, imsize, imsize))
    # ps = 131 # 259 # 131 # --> Extend a bit to 201 x 201 so we classify 70 x 70 patches

    ps = 131 # 131 # 131 # 259 # 19 # 35 # 67 # 131 # 17 # Non-dilated

    network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, ps + 50, ps + 50)) # 66 because 67x67 receptive field, pad both sided with 33
    # netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, ps + 70, ps + 70)) # 66 because 67x67 receptive field, pad both sided with 33
    # netfirst, network = getNetworkDilatedCascade(input_var, nclass=nclass, input_shape=(None, 1, 203, 203)) # 66 because 67x67 receptive field, pad both sided with 33

    # netfirst, network = getNetworkDilatedCascadeNew(input_var, nclass=nclass, input_shape=(None, 1, 203, 203), inputx=203, inputy=203) # 66 because 67x67 receptive field, pad both sided with 33


    # network = getNetworkNonDilated(input_var, nclass=nclass, input_shape=(None, 1, ps + 70, ps + 70)) # 66 because 67x67 receptive field, pad both sided with 33


    # othernetwork = getNetworkConventional(input_var, nclass=nclass, input_shape=(None, 1, imsize+2*ps, imsize+2*ps)) # 66 because 67x67 receptive field, pad both sided with 33
    # print('Conventional network contains ' + str(lasagne.layers.count_params(othernetwork)) + ' parameters.')

    # print('Load network')
    # with open(r'/home/jelmer/MICCAI2016PIM/experiments/ALLTRAININGSCANSLONG/19990.pickle') as f:
    #     print('Loading network')
    #     params_tmp = pickle.load(f)
    #     lasagne.layers.set_all_param_values(network, params_tmp)



    # prediction_first = lasagne.layers.get_output(netfirst)
    prediction_second = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    # loss = categorical_crossentropy_3D(prediction_first, target_one)
    loss = categorical_crossentropy_3D(prediction_second, target)
    # loss = loss.mean()
    # loss = lasagne.objectives.aggregate(loss, mask_var, mode='normalized_sum')
    # loss += categorical_crossentropy_3D(prediction_first, target_values)



    updates = lasagne.updates.adam(loss, all_params)
    # updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=0.001)


    train_fn = theano.function([input_var, target], loss, updates=updates, allow_input_downcast=True, on_unused_input='warn') # , mask_var

    # test_prediction_first = lasagne.layers.get_output(netfirst, deterministic=True)
    test_prediction_second = lasagne.layers.get_output(network, deterministic=True)
    # test_loss = categorical_crossentropy_3D(test_prediction_first, target_one)
    test_loss = categorical_crossentropy_3D(test_prediction_second, target)
    # test_loss_mean = lasagne.objectives.aggregate(test_loss, mask_var, mode='normalized_sum')
    # test_loss += categorical_crossentropy_3D(test_prediction_first, target_values)


    eval_fn = theano.function([input_var, target], test_loss, allow_input_downcast=True, on_unused_input='warn') # , mask_var
    # pred_fn = theano.function([input_var, target_one, target_two], test_loss, allow_input_downcast=True, on_unused_input='warn') # , mask_var


    # Load train data
    traindir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'train' + os.path.sep + 'images'
    filetype = r'*.nii'
    if task != r'brainmr' and task != r'cardiacmr':
        filetype = r'*.mhd'




    trainimages, trainlabels, traincount = loadImageDir(glob.glob(traindir + os.path.sep + filetype), nclass=nclass)
    # Jorg: traincount contains numpy array with class counts from reference images

    # Load validation data
    valdir = os.path.dirname(os.path.realpath(__file__)) +  os.path.sep + '..' + os.path.sep + task + os.path.sep + 'fold' + str(fold) + os.path.sep + 'validate' + os.path.sep + 'images'
    valimages, vallabels, valcount = loadImageDir(glob.glob(valdir + os.path.sep + filetype), nclass=nclass)

    startIt = -1
    num_epochs = 50000
    errors = np.empty(startIt + int(num_epochs))
    errors[:startIt] = 0
    testes = np.empty(startIt + int(num_epochs))
    testes[:startIt] = 0

    itInt = 10

    for it in range(num_epochs):
        print(it)
        images, labels, masks = generateBatch2D(trainimages, trainlabels, nclass=nclass, nsamp=128, classcount=traincount)
        # Jórg: masks is not used...currently...contains empty numpy array
        
        # images, labels_one, labels_two = generateBatch2DNew(trainimages, trainlabels, nclass=nclass, nsamp=32, classcount=traincount)
        print(images.shape)

        # error = train_fn(floatX(images), floatX(labels), floatX(masks))
        error = train_fn(floatX(images), floatX(labels)) # _one), floatX(labels_two))

        print('Train ' + str(error))
        if it % itInt == 0:
            errors[it:it+itInt] = error
            images, labels, masks = generateBatch2D(valimages, vallabels, nclass=nclass, nsamp=128, classcount=valcount)
            # images, labels_one, labels_two = generateBatch2DNew(valimages, vallabels, nclass=nclass, nsamp=32, classcount=valcount)
            # testes[it:it+itInt] = eval_fn(floatX(images), floatX(labels), floatX(masks))
            testes[it:it+itInt] = eval_fn(floatX(images), floatX(labels)) # _one), floatX(labels_two))


            # loss_map = pred_fn(floatX(images), floatX(labels), floatX(masks))
            # print('Loss map has shape ' + str(loss_map.shape))

            print('Test ' + str(testes[it]))
            t = range(it)
            plt.clf()
            plt.plot(t, np.log10(errors[:it]), label='Train loss')
            plt.plot(t, np.log10(testes[:it]), label='Validation loss')
            ax = plt.gca()
            # ax.set_ylim([0.0, 0.01])
            plt.legend(loc=3)
            figname = expdir + os.path.sep + 'error' + str(it) + '.png'
            plt.savefig(figname)
            netname = expdir + os.path.sep + str(it) + '.pickle'
            saveNetwork(network, netname)
        # if it % 100 == 0:
        #     # Compute dice
        #     image, spacing = cnu.load_mhd_to_npy(r'/home/jelmer/MICCAI2016PIM/pancreas/validate/images/0004.mhd')
        #     label, spacing = cnu.load_mhd_to_npy(r'/home/jelmer/MICCAI2016PIM/pancreas/validate/reference/0004.mhd')
        #     # image = valimages[0]
        #     outim = np.zeros(image.shape, dtype='int16')
        #     batch = np.zeros((1, 1, imsize, imsize))
        #     for z in range(image.shape[2]):
        #         batch[0,0,:,:] = np.squeeze(image[:,:,z])
        #         predslice = pred_fn(floatX(batch))
        #         predslice = np.squeeze(np.argmax(np.squeeze(predslice), axis=0))
        #         outim[:,:,z] = predslice
        #     TP, TN, FP, FN = evaluateSegmentation(outim, label, nclass)
        #     for labeli in range(1, nclass):
        #         print('Class ' + str(labeli) + ' DSC ' + str(2 * TP[labeli] / (TP[labeli] + FN[labeli] + TP[labeli] + FP[labeli])))

def search2DNew(netname, imname):
    maxmyo = 0.0
    maxblo = 0.0
    maxit = 0.0
    for it in range(8000, 10000, 50):
        myo, blo = test2DNew(netname.replace('8000', str(it)), imname)
        if (myo+blo)/2.0 > (maxmyo+maxblo)/2.0:
            maxmyo = myo
            maxblo = blo
            maxit = it
        print('For now best iteration for scan ' + os.path.split(imname)[-1] + ' ' + str(maxit) + ' myocardium ' + str(maxmyo) + ' blood pool ' + str(maxblo))


def search2D(netname, imname):
    maxmyo = 0.0
    maxblo = 0.0
    maxit = 0.0
    for it in range(4500, 5200, 20):
        myo, blo = test2D(netname.replace('5000', str(it)), imname)
        if (myo+blo)/2.0 > (maxmyo+maxblo)/2.0:
            maxmyo = myo
            maxblo = blo
            maxit = it
        print('For now best iteration for scan ' + os.path.split(imname)[-1] + ' ' + str(maxit) + ' myocardium ' + str(maxmyo) + ' blood pool ' + str(maxblo))



def ensemble2D(netname, imname):
    netdir, netbase = os.path.split(netname)
    imdir, imbase = os.path.split(imname)
    outbasename = (netdir + os.path.sep + imbase).replace('.nii','.mhd')
    image = sitk.ReadImage(imname)
    spacing = image.GetSpacing()

    it = int((os.path.split(netname)[-1]).split('.')[0])
    outim_myo, outim_blo = test2D(netname, imname)
    included = 1
    ref = sitk.ReadImage(imname.replace('images', 'reference'))
    ref = sitk.GetArrayFromImage(ref)
    ref = np.swapaxes(ref, 0, 2)
    ref_myo = (ref==1).astype('bool').flatten()
    ref_blo = (ref==2).astype('bool').flatten()

    precision, recall, thresholds = precision_recall_curve(ref_myo, outim_myo.flatten())
    f1scores = 2*(precision*recall)/(precision+recall)
    f1scores[np.isnan(f1scores)] = 0
    print('ENSEMBLE SCORE INCLUDED ' + str(included))
    print('Myocardium')
    print('Maximum Dice ' + str(np.max(f1scores)))
    print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
    print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))
    # print('Iteration ' + str(it))

    # for include in range(1,50):
    #     out_myo, out_blo = test2D(netname.replace(str(it), str(it+include*50)), imname)
    #     outim_myo += out_myo
    #     outim_blo += out_blo
    #     included += 1
    #     print('ENSEMBLE SCORE INCLUDED ' + str(included))
    #     precision, recall, thresholds = precision_recall_curve(ref_myo, outim_myo.flatten()/float(included))
    #     f1scores = 2*(precision*recall)/(precision+recall)
    #     f1scores[np.isnan(f1scores)] = 0
    #     print('Myocardium')
    #     print('Maximum Dice ' + str(np.max(f1scores)))
    #     print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
    #     print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))
    #     precision, recall, thresholds = precision_recall_curve(ref_blo, outim_blo.flatten()/float(included))
    #     f1scores = 2*(precision*recall)/(precision+recall)
    #     f1scores[np.isnan(f1scores)] = 0
    #     print('Blood pool')
    #     print('Maximum Dice ' + str(np.max(f1scores)))
    #     print('Threshold ' + str(thresholds[np.argmax(f1scores)]))
    #     print('Dice at threshold 0.5 ' + str(f1scores[np.argmin(abs(thresholds-0.5))]))
    #
    #
    #     outim = np.swapaxes(outim_blo, 0, 2)
    #     outim = sitk.GetImageFromArray(outim/float(included))
    #     outim.SetSpacing(spacing)
    #     sitk.WriteImage(outim, outbasename.replace('.mhd', '_ens_blood.mhd'), True)
    #     outim = np.swapaxes(outim_myo, 0, 2)
    #     outim = sitk.GetImageFromArray(outim/float(included))
    #     outim.SetSpacing(spacing)
    #     sitk.WriteImage(outim, outbasename.replace('.mhd', '_ens_myo.mhd'), True)



    #
    #
    # maxmyo = 0.0
    # maxblo = 0.0
    # maxit = 0.0
    # for it in range(4500, 5200, 20):
    #     outim_myo, outim_blo = test2D(netname, imname)
    #     # .replace('5000', str(it))
    #     if (myo+blo)/2.0 > (maxmyo+maxblo)/2.0:
    #         maxmyo = myo
    #         maxblo = blo
    #         maxit = it
    #     print('For now best iteration for scan ' + os.path.split(imname)[-1] + ' ' + str(maxit) + ' myocardium ' + str(maxmyo) + ' blood pool ' + str(maxblo))


def batch2D(netname, indirname):
    filenames = glob.glob(indirname + os.path.sep + '*.nii')
    for filename in filenames:
        test2D(netname, filename)


if __name__ == "__main__":
    # # main_regress()
    # # main(sys.argv[1], sys.argv[2])
    #
    # input_var = T.tensor4('input')
    # target_values = T.tensor4('labelmap')
    # mask_var = T.tensor4('mask')
    # nclass = 3
    # ps = 17 # Non-dilated
    # network = getNetworkNonDilated(input_var, nclass=nclass, input_shape=(None, 1, ps + 70, ps + 70)) # 66 because 67x67 receptive field, pad both sided with 33
    # print('Contains ' + str(lasagne.layers.count_params(network)) + ' parameters')
    #
    # countable = 0
    # all_params = lasagne.layers.get_all_param_values(network, trainable=True)
    # for p in all_params:
    #     print(p.shape)
    #     print(np.prod(p.shape))
    #     countable += np.prod(p.shape)
    # print('Trainable ' + str(countable))
    #
    #
    # ps = 259
    # network = getNetworkDilated(input_var, nclass=nclass, input_shape=(None, 1, ps + 70, ps + 70)) # 66 because 67x67 receptive field, pad both sided with 33
    # print('Contains ' + str(lasagne.layers.count_params(network)) + ' parameters')


    if sys.argv[1] == 'train':
        if '3D' in sys.argv[2]:
            main3D(sys.argv[2], sys.argv[3], sys.argv[4])
        elif '25D' in sys.argv[2]:
            main25D(sys.argv[2], sys.argv[3], sys.argv[4]) # Slabs
        else:
            main2D(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1] == 'test':
        if '3D' in sys.argv[2]:
            batch3D(sys.argv[2], sys.argv[3])
        if '25D' in sys.argv[2]:
            test25DFull(sys.argv[2], sys.argv[3]) # Full
        else:
            if 'NEW' in sys.argv[2]:
                # search2DNew(sys.argv[2], sys.argv[3])
                test2DNew(sys.argv[2], sys.argv[3])
            else:
                # search2D(sys.argv[2], sys.argv[3])
                # test2D(sys.argv[2], sys.argv[3])
                batch2D(sys.argv[2], sys.argv[3])
                # filt2D(sys.argv[2], sys.argv[3])
                # ensemble2D(sys.argv[2], sys.argv[3])


