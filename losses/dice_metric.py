import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from sklearn.metrics import precision_recall_curve

# https://github.com/EKami/carvana-challenge/blob/master/src/nn/losses.py
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


# taken from torchbiomed package
def dice_coeff(prediction, label):
    """
    Compute the Dice aka F1-score

    We assume prediction and label are PyTorch Variables:

    :param prediction: tensor with shape [samples]
    :param label: tensor with shape [samples]
    :return: Dice/F1 score, real valued
    """
    np_preds = None
    np_labels = None
    if not (isinstance(prediction, Variable) or isinstance(prediction, torch.FloatTensor)):
        raise TypeError('expected torch.autograd.Variable or torch.FloatTensor, but got: {}'
                        .format(torch.typename(prediction)))
    if isinstance(prediction, Variable):
        np_preds = prediction.data.cpu().squeeze().numpy()
        np_labels = label.data.cpu().squeeze().numpy()

    eps = 0.000001
    precision, recall, thresholds = precision_recall_curve(np_labels, np_preds)
    union = precision + recall + 2*eps
    intersect = precision * recall
    # make sure
    intersect[intersect <= eps] = eps
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    IoU[np.isnan(IoU)] = 0
    dice_scores = 2 * IoU
    # print('Maximum Dice ' + str(np.max(dice_scores)))
    # print('Threshold ' + str(thresholds[np.argmax(dice_scores)]))
    # print('Dice at threshold 0.5 ' + str(dice_scores[np.argmin(abs(thresholds-0.5))]))
    dice_at_threshold = dice_scores[np.argmin(abs(thresholds-0.5))]

    return dice_at_threshold
