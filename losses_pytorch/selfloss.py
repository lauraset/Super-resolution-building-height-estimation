import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

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


class SmoothL1(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        loss = F.smooth_l1_loss(inputs, targets, reduction=self.reduction)
        return loss

class WeightMSE(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, weights):
        loss = F.mse_loss(inputs, targets, reduction=self.reduction)
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss


class MSE(_Loss):
    def __init__(self, reduction='mean'):
        super().__init__()
    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction='reduction')
        return loss


### desin adaptive params
class MSE_adapt(_Loss):
    def __init__(self, log_var=0.0):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction='mean')
        precision = torch.exp(-self.log_var)
        loss = loss*precision + self.log_var
        return loss

class MSE_adapt_weight(_Loss):
    def __init__(self, log_var=0.0):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
    def forward(self, inputs, targets, weight):
        loss = F.mse_loss(inputs, targets, reduction='none')
        loss = (loss*weight).mean()
        precision = torch.exp(-self.log_var)
        loss = loss*precision + self.log_var
        return loss

# add hir for height hierarchy
class MSE_adapt_weight_hir(_Loss):
    def __init__(self, buildhir, heightweight, log_var=0.0, device='cuda'):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
        self.buildhir = torch.from_numpy(buildhir).long().to(device)
        self.heightweight = torch.from_numpy(heightweight).float().to(device)

    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction='none')
        # weight
        build = self.buildhir[targets.long()] #
        weight = self.heightweight[build]
        loss = (loss*weight).mean()
        precision = torch.exp(-self.log_var)
        loss = loss*precision + self.log_var
        return loss

# add ssim loss
# class MSE_SSIM_adapt_weight(_Loss):
#     def __init__(self, log_var=0.0):
#         super().__init__()
#         self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
#     def forward(self, inputs, targets, weight):
#         loss = F.mse_loss(inputs, targets, reduction='none')
#         loss = (loss*weight).mean()
#         precision = torch.exp(-self.log_var)
#         loss = loss*precision + self.log_var
#         return loss

class CE_DICE_adapt(nn.Module):
    def __init__(self, log_var=0.0):
        super().__init__()
        # self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.dice = Dice()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
    def forward(self, pmask, rmask):
        '''
        :param pmask: the prediction: N C H W , raw and unnormalized values
        :param rmask: the reference: N H W, with discrete values of [0, C-1]
        :return: loss
        '''
        loss_ce = self.ce(pmask, rmask)
        pmask = pmask.softmax(dim=1)
        pmask = pmask[:, 1:].sum(dim=1)
        loss_dice = self.dice(pmask, (rmask>0)) # for the minor class, e.g.,buildings
        loss = loss_ce + loss_dice
        precision = torch.exp(-self.log_var)
        loss = loss*precision + self.log_var
        return loss


class CE_DICE_adapt_weight(nn.Module):
    def __init__(self, log_var=0.0):
        super().__init__()
        # self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.dice = Dice()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))
    def forward(self, pmask, rmask, weight):
        '''
        :param pmask: the prediction: N C H W , raw and unnormalized values
        :param rmask: the reference: N H W, with discrete values of [0, C-1]
        :return: loss
        '''
        loss_ce = self.ce(pmask, rmask)
        loss_ce = (loss_ce*weight).mean()

        pmask = pmask.softmax(dim=1)
        pmask = pmask[:, 1:].sum(dim=1)
        loss_dice = self.dice(pmask, (rmask>0)) # for the minor class, e.g.,buildings

        loss = loss_ce + loss_dice
        precision = torch.exp(-self.log_var)
        loss = loss*precision + self.log_var
        return loss


if __name__=="__main__":
    # import numpy as np
    # from BH_loader import hierweight
    city = 'globe'
    preweight = f'../datastatsglobe/bh_stats_{city}.txt'
    # stats = np.loadtxt(preweight)
    # hir = (0, 3, 12, 21, 30, 60, 90, 256)
    # num_hier = len(hir)-1
    # heightweight = hierweight(stats, hir)
    # buildhir = np.zeros((256,), dtype='uint8')
    # for i in range(num_hier):
    #     buildhir[hir[i]:hir[i + 1]] = i
    #
    # loss = MSE_adapt_weight_hir(buildhir, heightweight)
    # device = 'cuda'
    # inputs = torch.zeros((1,1,64,64)).to(device)
    # targets = torch.randint_like(inputs, high=255).to(device)
    # value = loss(inputs, targets)