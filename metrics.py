import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class SegmentationMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset()
        self.count = 0
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # F1-score
    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)
    # MIOU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        IoU = self.IntersectionOverUnion()
        mIoU = torch.mean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return IoU
    # FWIOU
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = torch.sum(self.confusionMatrix, dim=1) / (torch.sum(self.confusionMatrix) + 1e-8)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix) + 1e-8)
        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = freq*iu
        return FWIoU

    def mFWIoU(self):
        return self.Frequency_Weighted_Intersection_over_Union().sum()

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass), dtype=torch.float64).to(self.device) # float, dtype=torch.int64) # int 64 is important


class ClassificationMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset(device)
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self, device):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        if device=='cuda':
            self.confusionMatrix = self.confusionMatrix.cuda()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HeightMetric(nn.Module):
    def __init__(self,  numClass=7, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device=device
        self.reset()

    def calc_rmse(self, pred, ref):
        rmse = torch.sqrt(((pred-ref)**2).mean())
        return rmse
    #
    # def calc_mse(self, pred, ref):
    #     mse = ((pred-ref)**2).mean()
    #     return mse

    def calc_mae(self, pred, ref):
        mae = (torch.abs(pred-ref)).mean()
        return mae

    def calc_me(self, pred, ref):
        me = (pred-ref).mean()
        return me

    def addBatch(self, pred, ref, buildhir):

        for i in range(self.numClass):
            mask = (buildhir==i)
            count = mask.sum().float()
            if int(count.item())==0:
                continue
            rmse = self.calc_rmse(pred[mask], ref[mask])
            mae = self.calc_mae(pred[mask], ref[mask])
            me = self.calc_me(pred[mask], ref[mask])

            self.stats[i, 0] += rmse * count
            self.stats[i, 1] += mae * count
            self.stats[i, 2] += me * count
            self.count[i] += count

    def getAvgEach(self):
        res = self.stats/(self.count+1e-10)
        return res

    def getAvgBalance(self):
        res = self.getAvgEach().mean(dim=0)
        return res

    def getAvgAll(self):
        res = self.stats.sum(dim=0)/(self.count.sum())
        return res

    def getCount(self):
        return self.count
    # def getAvgEach(self):
    #     res = self.__AvgEach_()
    #     # res[:, 0] = torch.sqrt(res[:, 0])
    #     return res
    #
    # def getAvgBalance(self):
    #     res = self.__AvgBalance_()
    #     # res[0] = torch.sqrt(res[0])
    #     return res
    #
    # def getAvgAll(self):
    #     res = self.__AvgAll_()
    #     # res[0] = torch.sqrt(res[0])
    #     return res

    def reset(self):
        self.count = torch.zeros((self.numClass, 1), dtype=torch.float64).to(self.device)
        self.stats = torch.zeros((self.numClass, 3), dtype=torch.float64).to(self.device) # rmse, mae, me
        self.balance_stats = torch.zeros((self.numClass, 3), dtype=torch.float64).to(self.device)  # balanced version


# multi-label classification metric
class MultilabelMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset(device)
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self, device):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        if device=='cuda':
            self.confusionMatrix = self.confusionMatrix.cuda()


def accprint(acc_total):
    oa = acc_total.OverallAccuracy().cpu().numpy()
    # miou = acc_total.meanIntersectionOverUnion().cpu().numpy()
    # iou = acc_total.IntersectionOverUnion().cpu().numpy()
    f1 = acc_total.F1score().cpu().numpy()
    ua = acc_total.Precision().cpu().numpy()
    pa = acc_total.Recall().cpu().numpy()
    cm = acc_total.getConfusionMatrix().cpu().numpy().T  # row-predict, col-ref
    print('oa, miou, iou， f1, ua, pa, confusion_matrix')
    # print('%.3f' % oa)
    # print('%.3f' % miou)
    # for i in iou:
    #     print('%.3f ' % (i), end='')
    print('\n')
    plot_confusionmatrix(np.vstack([f1, ua, pa]))
    plot_confusionmatrix(cm)
    print('numtotal: %d'%cm.sum())


def accprint_seg(acc_total):
    oa = acc_total.OverallAccuracy().cpu().numpy()
    miou = acc_total.meanIntersectionOverUnion().cpu().numpy()
    iou = acc_total.IntersectionOverUnion().cpu().numpy()
    f1 = acc_total.F1score().cpu().numpy()
    ua = acc_total.Precision().cpu().numpy()
    pa = acc_total.Recall().cpu().numpy()
    cm = acc_total.getConfusionMatrix().cpu().numpy().T  # row-predict, col-ref
    print('oa, miou, iou， f1, ua, pa, confusion_matrix')
    print('%.3f' % oa)
    print('%.3f' % miou)
    for i in iou:
        print('%.3f ' % (i), end='')
    print('\n')
    plot_confusionmatrix(np.vstack([f1, ua, pa]))
    plot_confusionmatrix(cm)
    print('numtotal: %d'%cm.sum())
    # ADD OA, IOU, F1, UA, PA
    print('%.3f'%oa)
    print('%.3f'%iou[1])
    print('%.3f'%f1[1])
    print('%.3f'%ua[1])
    print('%.3f'%pa[1])

def plot_confusionmatrix(cm):
    r = cm.shape[0]
    c = cm.shape[1]
    for i in range(r):
        for j in range(c):
            print('%.3f'%cm[i,j], end=' ')
        print('\n', end='')


def acc2file_cls(acc_total, txtpath):
    oa = acc_total.OverallAccuracy().cpu().numpy()
    # miou = acc_total.meanIntersectionOverUnion().cpu().numpy()
    # iou = acc_total.IntersectionOverUnion().cpu().numpy()
    f1 = acc_total.F1score().cpu().numpy()
    ua = acc_total.Precision().cpu().numpy()
    pa = acc_total.Recall().cpu().numpy()
    cm = acc_total.getConfusionMatrix().cpu().numpy().T  # row-predict, col-ref
    # write
    with open(txtpath, "w") as f:
        f.write('oa, miou, iou, f1, ua, pa, confusion_matrix\n')
        f.write(str(oa)+'\n')
        # f.write(str(miou) + '\n')
        # for i in iou:
        #     f.write(str(i)+' ')
        f.write('\n')
        for i in f1:
            f.write(str(i)+' ')
        f.write('\n')
        for i in ua:
            f.write(str(i)+' ')
        f.write('\n')
        for i in pa:
            f.write(str(i)+' ')
        f.write('\n')

        r = cm.shape[0]
        for i in range(r):
            for j in range(r):
                f.write(str(cm[i,j])+' ')
            f.write('\n')
        # ADD OA, IOU, F1, UA, PA
        f.write(str(oa)+'\n')
        # f.write(str(iou[1]) + '\n')
        f.write(str(f1[1]) + '\n')
        f.write(str(ua[1]) + '\n')
        f.write(str(pa[1]) + '\n')


def acc2file(acc_total, txtpath):
    oa = acc_total.OverallAccuracy().cpu().numpy()
    miou = acc_total.meanIntersectionOverUnion().cpu().numpy()
    iou = acc_total.IntersectionOverUnion().cpu().numpy()
    f1 = acc_total.F1score().cpu().numpy()
    ua = acc_total.Precision().cpu().numpy()
    pa = acc_total.Recall().cpu().numpy()
    cm = acc_total.getConfusionMatrix().cpu().numpy().T  # row-predict, col-ref

    fiou = acc_total.Frequency_Weighted_Intersection_over_Union().cpu().numpy()
    mfiou = acc_total.mFWIoU().cpu().numpy()
    # write
    r,c = cm.shape
    data = np.zeros((6+r, c), 'float64')
    data[0,:3] = [oa, miou, mfiou]
    data[1, :] = fiou
    data[2, :] = iou
    data[3, :] = f1
    data[4,:] = ua
    data[5, :] = pa
    data[6:6+r, :] = cm
    df = pd.DataFrame(data=data)
    namelist = ['oa_miou_mfiou', 'fiou', 'iou', 'f1', 'ua', 'pa'] + r*['cm']
    df['name'] = namelist
    df.to_csv(txtpath[:-4]+'.csv', index=False, header=False)

    # with open(txtpath, "w") as f:
    #     f.write('oa, miou, iou, f1, ua, pa, confusion_matrix\n')
    #     f.write(str(oa)+'\n')
    #     f.write(str(miou) + '\n')
    #     for i in iou:
    #         f.write(str(i)+' ')
    #     f.write('\n')
    #     for i in f1:
    #         f.write(str(i)+' ')
    #     f.write('\n')
    #     for i in ua:
    #         f.write(str(i)+' ')
    #     f.write('\n')
    #     for i in pa:
    #         f.write(str(i)+' ')
    #     f.write('\n')
    #
    #     r = cm.shape[0]
    #     for i in range(r):
    #         for j in range(r):
    #             f.write(str(cm[i,j])+' ')
    #         f.write('\n')
    #
    #     # ADD OA, IOU, F1, UA, PA
    #     f.write(str(oa)+'\n')
    #     f.write(str(iou[1]) + '\n')
    #     f.write(str(f1[1]) + '\n')
    #     f.write(str(ua[1]) + '\n')
    #     f.write(str(pa[1]) + '\n')


def acc2fileRMSE(acc_total, txtpath):
    rmse = acc_total.avg
    with open(txtpath, "w") as f:
        f.write(str(rmse))


def acc2fileHeight(acc, txtpath):
    acceach = acc.getAvgEach().cpu().numpy()
    accbalance = acc.getAvgBalance().cpu().numpy()
    accall = acc.getAvgAll().cpu().numpy()
    counts = acc.getCount().cpu().numpy()

    acceach = np.concatenate([acceach, counts], axis=1)
    accbalance0 = np.zeros((1, 4))
    accbalance0[0, :3] = accbalance

    accall0 = np.zeros((1, 4))
    accall0[0, :3] = accall

    data = np.concatenate([acceach, accbalance0, accall0], axis=0)

    df = pd.DataFrame(data=data,
                      columns=['rmse', 'mae', 'me', 'count'])
    df.to_csv(txtpath[:-4]+'.csv', index=False)

if __name__=="__main__":
    m = SegmentationMetric(3,device='cpu')
    ref = torch.tensor([0,0,1,1,2,2,2,2,2])
    pred = torch.tensor([0,1,0,1,0,2, 0, 0,0 ])
    m.addBatch(pred, ref)
    # print(m.Precision())
    # print(m.Recall())
    # print(m.F1score())
    # print(m.OverallAccuracy())
    fiou = m.Frequency_Weighted_Intersection_over_Union()
    mfiou = m.mFWIoU()
    print(fiou)
    print(mfiou)
    acc2file(m, 'tmpacc_seg.csv')

    # acc = HeightMetric(device='cpu', numClass=7)
    # ref = torch.tensor([0,0,3,6,5,1]).float()
    # pred = torch.tensor([0,1,0,1,0,2]).float()
    # acc.addBatch(pred, ref, ref)
    #
    # a1 = acc.getAvgEach()
    # a2 = acc.getAvgAll()
    # a3 = acc.getAvgBalance()
    #
    # print(a1)
    # print(a2)
    # print(a3)
    #
    # acc2fileHeight(acc, 'tmpacc.csv')
