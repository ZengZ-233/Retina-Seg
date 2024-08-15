from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Directable = {'upper_left': [-1, -1], 'up': [0, -1], 'upper_right': [1, -1], 'left': [-1, 0], 'right': [1, 0],
              'lower_left': [-1, 1], 'down': [0, 1], 'lower_right': [1, 1]}
TL_table = ['lower_right', 'down', 'lower_left', 'right', 'left', 'upper_right', 'up', 'upper_left']


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may change
        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))
        score = (2. * intersection + smooth) / (i + j + smooth)
        return 1 - score

    def soft_dice_loss(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0], 1, y_true.shape[1], y_true.shape[2])
        for i in range(1):
            y_true = torch.concat((y_true, y_true), dim=1)
        loss = self.soft_dice_coeff(y_pred, y_true)
        return loss.mean()

    def __call__(self, y_pred, y_true):
        b = self.soft_dice_loss(y_pred, y_true)
        return b


def edge_loss(vote_out, con_target):
    vote_out=F.softmax(vote_out,dim=1)
    # print(vote_out.shape,con_target.shape)
    # vote_out=vote_out[:,0,:,:,:]
    con_target = con_target.view(con_target.shape[0], 1, con_target.shape[1], con_target.shape[2],
                                 con_target.shape[3])
    for i in range(1):
        con_target = torch.concat((con_target, con_target), dim=1)
    sum_conn = torch.sum(con_target.clone(), dim=1)
    edge = torch.where((sum_conn < 8) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

    pred_mask_min, _ = torch.min(vote_out.cpu(), dim=1)

    pred_mask_min = pred_mask_min * edge

    minloss = F.binary_cross_entropy(pred_mask_min, torch.full_like(pred_mask_min, 0))
    # print(minloss)
    return minloss  # +maxloss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)


class bicon_loss(nn.Module):
    def __init__(self, size):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()
        self.dice_loss = dice_loss()
        self.ce = nn.CrossEntropyLoss()
        self.num_class = 1
        hori_translation = torch.zeros([1, self.num_class, size[1], size[1]])
        for i in range(size[1] - 1):
            hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
        verti_translation = torch.zeros([1, self.num_class, size[0], size[0]])
        for j in range(size[0] - 1):
            verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
        self.hori_trans = hori_translation.float().cpu()
        self.verti_trans = verti_translation.float().cpu()

    def forward(self, c_map, target, con_target):

        batch_num = c_map.shape[0]
        num_class = c_map.shape[1]
        # con_target = con_target.type(torch.FloatTensor).cpu()

        # find edge ground truth
        # sum_conn = torch.sum(con_target, dim=1)
        # edge = torch.where(sum_conn < 8, torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))
        # edge0 = torch.where(sum_conn > 0, torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))
        # edge = edge * edge0

        # c_map = F.sigmoid(c_map)#原代码这里真的逆天

        # construct the translation matrix
        self.hori_translation = self.hori_trans.repeat(batch_num, num_class, 1, 1).cpu()
        self.verti_translation = self.verti_trans.repeat(batch_num, num_class, 1, 1).cpu()

        final_pred, bimap = self.Bilater_voting(c_map)
        ce_l = self.ce(c_map, target)

        # target = target.type(torch.FloatTensor).cpu()
        # target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        # for i in range(1):
        #     target = torch.concat((target, target), dim=1)
        # target=torch.tensor(target).long()
        # target = target.to(torch.float)
        # ce = self.cross_entropy_loss(final_pred, target)
        # bimap_l = self.cross_entropy_loss(bimap.squeeze(1), con_target_new)#这一个注释了之后效果反而变好了从72-91
        # apply any loss you want below using final_pred, e.g., dice loss.
        # loss_dice = self.dice_loss(final_pred, target)#这个改成CE损失函数效果又增长了
        # vote_out = vote_out.squeeze(1)
        # decouple loss
        # print(c_map.max(),bimap.max(),final_pred.max())
        decouple_loss = edge_loss(bimap.squeeze(1), con_target)

        # print("final_pred",final_pred.size())
        # print("target",target.size())
        # target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
        # for i in range(1):
        #     target = torch.concat((target, target), dim=1)

        # bce_loss = self.cross_entropy_loss(final_pred, target)
        loss = 0.2 * decouple_loss + 0.8 * ce_l  ## add dice loss if needed for biomedical data
#计算原图像和mask损失以及计算边缘损失,比例2:8可以修改比例
        # print(loss.item(), decouple_loss.item()*0.15, ce_l.item()*0.85)
        # 总，边缘和边缘，预测和目标
        return loss

    def shift_diag(self, img, shift):
        ## shift = [1,1] moving right and down
        # print(img.shape,self.hori_translation.shape)
        # print("img",img.size())
        batch, class_num, row, column = img.size()

        if shift[0]:  ###horizontal
            img = torch.bmm(img.view(-1, row, column), self.hori_translation.view(-1, column, column)) if shift[
                                                                                                              0] == 1 else torch.bmm(
                img.view(-1, row, column), self.hori_translation.transpose(3, 2).view(-1, column, column))
        if shift[1]:  ###vertical
            img = torch.bmm(self.verti_translation.transpose(3, 2).view(-1, row, row), img.view(-1, row, column)) if \
                shift[1] == 1 else torch.bmm(self.verti_translation.view(-1, row, row), img.view(-1, row, column))
        return img.view(batch, class_num, row, column)

    def Bilater_voting(self, c_map):
        c_map = c_map.view(c_map.shape[0], -1, 1, c_map.shape[2], c_map.shape[3])
        for i in range(3):
            c_map = torch.concat((c_map, c_map), dim=2)
        # print(c_map.shape)
        # batch, class_num, channel, row, column = c_map.size()

        shifted_c_map = torch.zeros(c_map.size()).cpu()
        for i in range(8):
            shifted_c_map[:, :, i] = self.shift_diag(c_map[:, :, 7 - i].clone(), Directable[TL_table[i]])
        vote_out = c_map * shifted_c_map
        pred_mask, _ = torch.max(vote_out, dim=2)
        return pred_mask, vote_out  # , bimap



###------------------------------------Lovasz_Softmax_loss------------------------------------------------

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""



import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n