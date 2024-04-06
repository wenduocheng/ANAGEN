import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import operator
from itertools import product
from functools import reduce, partial


"""Customized Task Metrics"""

def auroc(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()
    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc


def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    score = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, score


def calculate_stats(output, target, class_indices=None):
    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    for k in class_indices:
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        dict = {'AP': avg_precision}
        stats.append(dict)

    return stats


def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr


"""Customized Task Losses"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean' if self.size_average else 'sum')
        if torch.cuda.is_available():
             self.criterion =  self.criterion.cuda()

    def forward(self, output, target):
        model_out = F.softmax(output, dim = 1) + 1e-9

        ce = torch.multiply(target, -torch.log(model_out))
        weight = torch.multiply(target, (1 - model_out) ** self.gamma)
        fl = self.alpha * torch.multiply(weight, ce)
        reduced_fl = torch.sum(fl, axis=1)
        return reduced_fl.mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.cuda = cuda
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')


    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            
            if self.cuda:
                self.criterion = self.criterion.cuda()
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        
        loss = self.criterion(logit, target.long())
        
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n = logit.size()[0]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def logCoshLoss(y_t, y_prime_t, reduction='mean', eps=1e-12):
    if reduction == 'mean':
        reduce_fn = torch.mean
    elif reduction == 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = lambda x: x
    x = y_prime_t - y_t
    return reduce_fn(torch.log((torch.exp(x) + torch.exp(-x)) / 2))


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


"""Hepler Funcs"""

def count_params(model):
    c = 0
    for p in model.parameters():
        try:
            c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def print_grad(model, kernel_choices, dilation_choices):
    param_values, ks, ds = [], [], []
    for name, param in model.named_arch_params():
        param_values.append(param.data.argmax(0))
        print(name, param.data)
        ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

    print("Kernel pattern (name, k, d):", ks, np.sum(ks), ds, np.sum(ds))

def mask(img, ignore):
    return img * (1 - ignore)

