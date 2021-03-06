import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics.classification import UndefinedMetricWarning

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
def dice_coeff(pred_scores, pred_labels, true_labels):
    """
    Compute the Dice aka F1-score

    We assume prediction and label are PyTorch Variables:

    :param prediction: tensor with shape [samples]
    :param label: tensor with shape [samples]
    :return: Dice/F1 score, real valued
    """
    np_preds = None
    np_labels = None
    if not (isinstance(pred_scores, Variable) or isinstance(pred_scores, torch.FloatTensor)):
        raise TypeError('expected torch.autograd.Variable or torch.FloatTensor, but got: {}'
                        .format(torch.typename(pred_scores)))
    if isinstance(pred_scores, Variable):
        np_preds = pred_scores.data.cpu().squeeze().numpy()
        np_true_labels = true_labels.data.cpu().squeeze().numpy()
        np_pred_labels = pred_labels.data.cpu().squeeze().numpy()

    eps = 0.000001
    dice = 0.
    # precision, recall, thresholds = precision_recall_curve(np_true_labels, np_preds)
    if np.all(np_true_labels == 0):
        # print(np.sum(np_pred_labels == 1))
        if np.all(np_pred_labels == 0):
            dice = 1.
    if dice != 1.:
        try:
            dice = f1_score(np_true_labels, np_pred_labels)
        except UndefinedMetricWarning:
            print("WARNING - No true positives (TP) in image slice")
        except:
            print("Some unknown error occurred")

    # union = precision + recall + 2*eps
    # intersect = precision * recall
    # make sure
    # intersect[intersect <= eps] = eps
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    # IoU = intersect / union
    # IoU[np.isnan(IoU)] = 0
    # dice_scores = 2 * IoU
    # print('Maximum Dice ' + str(np.max(dice_scores)))
    # print('Threshold ' + str(thresholds[np.argmax(dice_scores)]))
    # print('Dice at threshold 0.5 ' + str(dice_scores[np.argmin(abs(thresholds-0.5))]))
    # dice_at_threshold = dice_scores[np.argmin(abs(thresholds-0.5))]
    dice_at_threshold = []
    return dice_at_threshold, dice
