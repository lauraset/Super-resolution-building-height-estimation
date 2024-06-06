import torch
import torch.nn as nn
import torch.nn.functional as F
from losses_pytorch.iou_loss import IOU
from losses_pytorch.ssim_loss import SSIM


class CE_MSE(nn.Module):
    def __init__(self, weight=None, beta=0.7):
        super().__init__()
        self.weight = weight
        self.beta = beta # the balance term
    def forward(self, pmask, rmask, pbd,  rbd):
        ce = F.cross_entropy(pmask, rmask, weight=self.weight)
        mse = F.mse_loss(pbd, rbd.float()/255.) # normed to 0-1
        loss = ce + self.beta*mse
        return loss


class BCE_SSIM_IOU(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim + loss_iou
        return loss


# 2021.11.26:
class BCE_IOU(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        # loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce  + loss_iou
        return loss


class BCE_SSIM(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        # self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        # loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim # + loss_iou
        return loss


class BCE_SSIM_IOU_BD(nn.Module):
    def __init__(self,issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.bd = torch.nn.MSELoss()
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask, pbd, rbd):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss_bd = self.bd(pbd, rbd.float()/255.)
        loss = loss_ce + loss_ssim + loss_iou + loss_bd
        return loss

# 2022.1.8
class Dice(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # pred: sigmoid
        # targer: 0, 1
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
        return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# 2022.1.8
# pmask: sigmoid
# rmask: 0,1
class BCE_DICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.dice = Dice()
    def forward(self, pmask, rmask):
        loss_ce = self.ce(pmask, rmask)
        loss_dice = self.dice(pmask, rmask)
        loss = loss_ce  + loss_dice
        return loss

class BCE_DICE_logits(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.dice = Dice()
    def forward(self, pmask, rmask):
        loss_ce = self.ce(pmask, rmask)
        loss_dice = self.dice(torch.sigmoid(pmask), rmask)
        loss = loss_ce  + loss_dice
        return loss

# 2022.8.24: define building loss with 2 classes
# 2022.8.31: add ignore_label=-1
class CE_DICE(nn.Module):
    def __init__(self):
        super(CE_DICE, self).__init__()
        # self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.dice = Dice()
    def forward(self, pmask, rmask):
        '''
        :param pmask: the prediction: N C H W , raw and unnormalized values
        :param rmask: the reference: N H W, with discrete values of [0, C-1]
        :return: loss
        '''
        loss_ce = self.ce(pmask, rmask)
        pmask = pmask.softmax(dim=1)
        pmask = pmask[:, 1]
        loss_dice = self.dice(pmask, rmask) # for the minor class, e.g.,buildings
        loss = loss_ce + loss_dice
        return loss


class CE_DICE_IOU(nn.Module):
    def __init__(self, ignore_label=-1):
        super(CE_DICE_IOU, self).__init__()
        self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_label)
        self.dice = Dice()
        self.iou = IOU(size_average=True)

    def forward(self, pmask, rmask):
        '''
        :param pmask: the prediction: N C H W , raw and unnormalized values
        :param rmask: the reference: N H W, with discrete values of [0, C-1]
        :return: loss
        '''
        loss_ce = self.ce(pmask, rmask)
        pmask = pmask.softmax(dim=1) # normalized, NCHW
        pmask = pmask[:, 1:].sum(dim=1) # , N H W
        # rmask[(rmask>0) & (rmask!=self.ignore_label)] = 1 # positive>0, and change to 0,1, ignore
        valid = (rmask!=self.ignore_label) # N H W
        loss_dice = self.dice(pmask[valid], rmask[valid]) # for the minor class, e.g.,buildings
        loss_iou = self.iou(pmask[valid], rmask[valid])
        loss = loss_ce + loss_dice + loss_iou
        return loss