'''
extract hrfeature from realesanet, and use it to train height estimation model
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from tensorboardX import SummaryWriter
from BH_loader import myImageFloder_S12_globe
from metrics import AverageMeter, acc2fileRMSE, SegmentationMetric, acc2file, HeightMetric, acc2fileHeight
from SR.rrdbnet_arch import RealESRGAN
from utils.preprocess import array2raster_rio, array2raster
from mymodels import SRRegress_Cls_feature
import shutil
from osgeo import gdal
import argparse
from losses_pytorch.selfloss import CE_DICE_adapt, MSE_adapt, MSE_adapt_weight, CE_DICE_adapt_weight
from BH_loader import wholeimgLoader


def get_args(city='globe'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=r'.\data')
    parser.add_argument('--trainlist', default=f'datalist_{city}_train_0.7.csv')
    parser.add_argument('--vallist', default=f'datalist_{city}_test_0.7_val_0.3.csv')
    parser.add_argument('--testlist', default=f'datalist_{city}_test_0.7_test_0.3.csv')
    parser.add_argument('--logdir', default=fr'.\weights\realesrgan_feature_aggre_weight_{city}')
    parser.add_argument('--logdirhr', default=fr'.\weights\realesrgan\checkpoint2.tar') # add
    parser.add_argument('--rgbseq', default=[0, 1, 2], help='the location of RGB bands')

    parser.add_argument('--checkpoint',default='checkpoint.tar')
    parser.add_argument('--nchans', default=8)
    parser.add_argument('--nchanss2', default=6)
    # model train parameter
    parser.add_argument('--maxepoch', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    # BMSE: balance MSE
    # parser.add_argument('--bmse', type=bool, default=False)
    # parser.add_argument('--init_noise_sigma', type=float, default=1.0)
    # parser.add_argument('--sigma_lr', type=float, default=0.01)
    # parser.add_argument('--fix_noise_sigma', type=bool, default=False)
    parser.add_argument('--wmse', type=bool, default=False)
    parser.add_argument('--datastats', type=str, default='datastatsglobe')
    parser.add_argument('--preweight', type=str, default=f'datastatsglobe/bh_stats_{city}.txt', help='None') # weight
    parser.add_argument('--s1dir', type=str, default=f's1{city}_check')
    parser.add_argument('--s2dir', type=str, default=f's2{city}_check')
    parser.add_argument('--bhdir', type=str, default=f'bh{city}')
    parser.add_argument('--normheight', type=float, default=1.0)
    parser.add_argument('--smoothl1', type=bool, default=False)
    parser.add_argument('--isaggre', type=bool, default=True)
    parser.add_argument('--ishir', type=bool, default=True)
    parser.add_argument('--hir', type=tuple, default=(0, 3, 12, 21, 30, 60, 90, 256))
    parser.add_argument('--chans_build', type=int, default=7) # the channels of building hierarchical classification

    # parser.add_argument('--ismodelhir', type=bool, default=False)
    # save predicted images
    parser.add_argument('--wholeimgpath', type=str, default=r'D:\data\Landcover\s12range')
    parser.add_argument('--cityname', type=str, default='lanzhou') # ningbo, tianjin, lanzhou
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--stride', type=int, default=60) # overlap 4 pixels =40m = 2.5m*16
    args = parser.parse_args()
    return args


def adjust_learning_rate(init_lr, epoch, optimizer):
    if epoch<=10:
        lr = init_lr
    elif epoch<=20:
        lr = 0.1*init_lr
    else:
        lr = 0.01*init_lr
    # lr = init_lr * (0.1 ** (epoch // 10))

    for param_group in optimizer.param_groups:
        if ('lossweight' in param_group) and (param_group['name'] == 'lossweight'):
            continue
        param_group['lr'] = lr
    return lr


def main(args):
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    # Setup dataloader
    data_path = args.datapath
    trainlist = args.trainlist
    vallist = args.vallist
    batch_size = 16
    num_workers = 8
    trainlist = os.path.join(data_path, trainlist) # training
    vallist = os.path.join(data_path, vallist) # validation
    epochs = args.maxepoch
    epoch_eval = 1
    nchannels = args.nchans # 6+2
    classes = 1
    logdir = args.logdir
    writer = SummaryWriter(log_dir=logdir)
    logdir_hr = args.logdirhr
    best_acc = 0
    init_lr = args.lr

    # train & val dataloader
    num_sample =0
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder_S12_globe(trainlist, data_path, datastats=args.datastats,
                                    normmethod='minmax', datarange=(0, 1),
                                    aug=True, num_sample=num_sample,
                                    preweight=args.preweight,
                                    s1dir=args.s1dir, s2dir=args.s2dir, heightdir=args.bhdir,
                                    isaggre=args.isaggre, ishir=args.ishir,
                                    hir=args.hir, nchans=args.nchanss2),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    valdataloader = torch.utils.data.DataLoader(
        myImageFloder_S12_globe(vallist, data_path, datastats=args.datastats,
                                   normmethod='minmax', datarange=(0, 1),
                                   aug=False, num_sample=num_sample//2,
                                   s1dir=args.s1dir, s2dir=args.s2dir, heightdir=args.bhdir,
                                    isaggre=False, ishir=False,
                                    nchans=args.nchanss2),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Super-resolution image reconstruction
    net_hr = RealESRGAN(pretrain_g_path=None,
                     pretrain_d_path=None,
                     device=device, scale=4,
                     num_block=23)
    net_hr.net_g.load_state_dict(torch.load(logdir_hr)['net_g_ema'])
    net_hr.net_g.eval()
    for p in net_hr.net_g.parameters():
        p.requires_grad = False

    # super-resolution semantic segmentation
    net = SRRegress_Cls_feature(encoder_name="efficientnet-b4",
                                   in_channels=nchannels, super_in=64,
                                   super_mid=16, upscale=4,
                                isaggre=args.isaggre,
                                chans_build=args.chans_build,
                                ).to(device)

    # resume the model
    start_epoch = 0
    lossweight = [0.0, 0.0, 0.0]
    resume = os.path.join(logdir, 'checkpoint.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        lossweight = checkpoint['log_vars']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer.global_step = start_epoch*iter_per_epoch
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")
        # return

    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, weight_decay=1e-4)
    # for height
    criterion = [MSE_adapt_weight(lossweight[0])]
    if args.isaggre:
        criterion.append(MSE_adapt_weight(lossweight[1]))
    # for building
    criterion.append(CE_DICE_adapt_weight(lossweight[2]))

    optimizer.add_param_group({'params':[i.log_var for i in criterion],
                               'lr': 0.001, 'name': 'lossweight'})

    for epoch in range(epochs):
        if epoch<start_epoch:
            continue
        epoch = epoch + 1 # current epochs
        lr = adjust_learning_rate(init_lr=init_lr, epoch=epoch, optimizer=optimizer)
        print('epoch %d, lr: %.6f'%(epoch, lr))
        if args.isaggre:
            train_loss, train_rmse, lossweight = train_epoch_aggre_weight(net, net_hr.net_g, criterion, traindataloader,
                                                      optimizer, device, epoch, classes, args.rgbseq)
        else:
            train_loss, train_rmse = train_epoch(net, net_hr.net_g, criterion, traindataloader,
                                                     optimizer, device, epoch, classes, args.rgbseq)
        # eval
        val_rmse = 0
        if (epoch % epoch_eval==0) or (epoch==1):
            val_loss, val_rmse = vtest_epoch(net, net_hr.net_g, criterion[0], valdataloader,
                                             device, epoch, classes, args.rgbseq)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint.tar')
        is_best = val_rmse < best_acc # the lower of rmse, the higher accuracy
        best_acc = min(val_rmse, best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            'log_vars': lossweight if args.isaggre else 1.0,
            'best_acc': best_acc,
            #'optimizer': optimizer.state_dict(),
        }, savefilename)
        if is_best:
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        if epoch%5==0:
            shutil.copy(savefilename, os.path.join(logdir, f'checkpoint{epoch}.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/rmse', train_rmse, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/rmse', val_rmse, epoch)
        if args.isaggre:
            writer.add_scalar('lossweight/w1', lossweight[0], epoch)
            writer.add_scalar('lossweight/w2', lossweight[1], epoch)
            writer.add_scalar('lossweight/w3', lossweight[2], epoch)
    writer.close()

def train_epoch_aggre_weight(net, net_hr, criterion, dataloader, optimizer, device, epoch, classes, rgbseq):
    net.train()
    losses = AverageMeter()
    acc = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    lossweight = 1.0

    for idx, (lr, heightall, build, weightall) in enumerate(dataloader):
        # combine pos and neg
        lr = lr.to(device, non_blocking=True) # N C H W
        # hr = hr.to(device, non_blocking=True)
        height = heightall[0].to(device, non_blocking=True)
        height_aggre = heightall[1].to(device, non_blocking=True)
        build = build.to(device, non_blocking=True)
        weight = weightall[0].to(device, non_blocking=True)
        weight_aggre = weightall[1].to(device, non_blocking=True)

        with torch.no_grad():
            hr_fea = net_hr.forward_feature(lr[:, rgbseq]) # RGB of sentinel-2 images

        height_pred, build_pred, height_pred_aggre = net.forward(lr, hr_fea)

        height_pred = height_pred.squeeze(1) # N H W
        height_pred_aggre = height_pred_aggre.squeeze(1)

        loss = criterion[0](height_pred, height, weight) + \
               criterion[1](height_pred_aggre, height_aggre, weight_aggre) + \
               criterion[2](build_pred, build, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsize = lr.size(0)
        losses.update(loss.item(), bsize)
        with torch.no_grad():
            rmse = torch.sqrt(((height_pred - height) ** 2).mean())

        acc.update(rmse.item(), bsize)
        lossweight = [i.log_var.item() for i in criterion]
        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. RMSE {rmse:.3f}. w1 {w1:.3f}. w2 {w2:.3f}. w3 {w3:.3f}.'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg, rmse=acc.avg,  w1=lossweight[0], w2=lossweight[1], w3=lossweight[2]))
        pbar.update()
    pbar.close()

    return losses.avg, acc.avg, lossweight


def train_epoch(net, net_hr, criterion, dataloader, optimizer, device, epoch, classes, rgbseq):
    net.train()
    losses = AverageMeter()
    acc = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (lr, height, build) in enumerate(dataloader):
        # combine pos and neg
        lr = lr.to(device, non_blocking=True) # N C H W
        # hr = hr.to(device, non_blocking=True)
        height = height.to(device, non_blocking=True)
        build = build.to(device, non_blocking=True)

        with torch.no_grad():
            hr_fea = net_hr.forward_feature(lr[:, rgbseq]) # RGB of sentinel-2 images

        height_pred, build_pred = net.forward(lr, hr_fea)
        height_pred = height_pred.squeeze(1)

        loss = criterion[0](height_pred, height) +\
                criterion[1](build_pred, build)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsize = lr.size(0)
        losses.update(loss.item(), bsize)
        with torch.no_grad():
            rmse = torch.sqrt(((height_pred - height) ** 2).mean())

        acc.update(rmse.item(), bsize)
        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. RMSE {rmse:.3f}'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg, rmse=acc.avg))
        pbar.update()
    pbar.close()

    return losses.avg, acc.avg


def vtest_epoch(model, net_hr,  criterion, dataloader, device, epoch, classes, rgbseq):
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true, _, _) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)

            hr_fea = net_hr.forward_feature(x[:, rgbseq])  # RGB of sentinel-2 images

            ypred = model.forward(x, hr_fea)
            ypred = ypred[0].squeeze(1)
            # ypred = ypred.reshape(-1, 1)
            # y_true = y_true.reshape(-1, 1)

            loss = torch.mean((ypred-y_true)**2) # rmse
            losses.update(loss.item(), x.size(0))
            rmse = torch.sqrt(((ypred - y_true) ** 2).mean())
            acc.update(rmse, x.size(0))

            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. RMSE {rmse:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, rmse=acc.avg))
            pbar.update()
        pbar.close()

    return losses.avg, acc.avg


def main_test(args, num_sample=100, suffix='',
              iswhole=False, istest=True, is1km=False, ispred=False,
              batch_size=1, num_workers=6, issave=False,
              ):
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    data_path = args.datapath
    testlist_path = os.path.join(data_path, args.testlist)
    logdir_hr = args.logdirhr

    # Setup parameters
    classes = args.chans_build
    nchannels = args.nchans
    device = 'cuda'
    logdir = args.logdir

    # num_sample = 100
    tstdataloader = torch.utils.data.DataLoader(
        myImageFloder_S12_globe(testlist_path, data_path, datastats=args.datastats,
                                   normmethod='minmax', datarange=(0, 1),
                                   aug=False, num_sample=num_sample,
                                preweight=args.preweight,
                                s1dir=args.s1dir, s2dir=args.s2dir, heightdir=args.bhdir,
                                isaggre=False, ishir=True,
                                hir=args.hir,
                                nchans=args.nchanss2),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # super-resolution semantic segmentation
    net = SRRegress_Cls_feature(encoder_name="efficientnet-b4",
                                   in_channels=nchannels, super_in=64,
                                   super_mid=16, upscale=4,
                                chans_build=args.chans_build).to(device)
    # Super-resolution image reconstruction
    net_hr = RealESRGAN(pretrain_g_path=None,
                     pretrain_d_path=None,
                     device=device, scale=4,
                     num_block=23)
    net_hr.net_g.load_state_dict(torch.load(logdir_hr)['net_g_ema'])
    net_hr.net_g.eval()
    for p in net_hr.net_g.parameters():
        p.requires_grad = False

    # print the model
    resume = os.path.join(logdir, args.checkpoint)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        if 'iter' in checkpoint.keys():
            start_epoch = checkpoint['iter']
        else:
            start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")
        return

    # scale = 1.0
    id = str(start_epoch) # + str(scale)
    txtpath = os.path.join(logdir, 'acc' + id + suffix
                           +'_'+ str(num_sample)+'.txt')  # save acc

    respath = os.path.join(logdir, 'pred_' + id+'_'+suffix)
    if issave:
        os.makedirs(respath, exist_ok=True)

    if istest:
        vtest_epoch2(net, net_hr.net_g, tstdataloader, device,
                     classes, start_epoch, txtpath, issave, respath, args.rgbseq)


def vtest_epoch2(model, net_hr, dataloader, device, classes, epoch, txtpath, issave=False, respath=None, rgbseq=[0,1,2]):
    model.eval()
    acc_total = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    acc_seg = SegmentationMetric(classes, device)
    acc_he = HeightMetric(classes, device)

    with torch.no_grad():
        for idx, (x, y_true, build, testlist) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            build = build.to(device, non_blocking=True)

            hr_fea = net_hr.forward_feature(x[:, rgbseq])  # RGB of sentinel-2 images
            ypred, build_pred = model.forward(x, hr_fea)  # N C H W
            ypred = ypred.squeeze(1)
            rmse = torch.sqrt(((ypred - y_true) ** 2).mean())
            acc_total.update(rmse.item(), x.size(0))
            build_pred = build_pred.argmax(1)

            acc_seg.addBatch(build_pred, build)
            acc_he.addBatch(ypred, y_true, build)

            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. RMSE:{rmse:.3f} OA: {oa:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, rmse=acc_total.avg, oa=acc_seg.OverallAccuracy()))
            pbar.update()
            # save to dir
            if issave:
                for k, imgpath in enumerate(testlist):
                    ibase = os.path.basename(imgpath)[:-4]  #
                    # pred_name = os.path.join(respath, ibase + "_pred.png")
                    predprob_name = os.path.join(respath, ibase + "_predprob.tif")
                    predhr_name = os.path.join(respath, ibase+"_rgb.tif")
                    predbuild_name = os.path.join(respath, ibase+"_build.tif")
                    # building height
                    tmp = ypred[k].squeeze().cpu().numpy()
                    tmp[tmp<0] =0 # positive
                    tmp = (np.round(tmp*10)).astype('uint16')
                    array2raster_rio(predprob_name, tmp, imgpath, iscmap=False, compress='PACKBITS')
                    # high-resolution image
                    # tmp = hr_pred[k].permute((1,2,0)).cpu().numpy()
                    # tmp = tmp*norm[1]+norm[0]
                    # tmp = tmp.astype('uint8')
                    # array2raster(predhr_name, tmp, imgpath,
                    #              nresolution=2.5,
                    #              datatype=gdal.GDT_Byte)
                    # building footprint
                    tmp = build_pred[k].cpu().numpy().astype('uint8') # N 2 H W
                    # tmp = tmp*255
                    array2raster_rio(predbuild_name, tmp, imgpath, iscmap=True, compress='PACKBITS')

        pbar.close()

    # accprint_seg(acc_total)
    acc2fileRMSE(acc_total, txtpath)
    acc2file(acc_seg, txtpath[:-4]+'_seg.txt')
    acc2fileHeight(acc_he, txtpath[:-4]+'_he.txt')
    return True


if __name__=="__main__":
    args = get_args(city='globe')
    args.maxepoch = 20 #30, 20 is enough and time-efficient than 30 through preliminary experiments
    main(args)
    args.checkpoint = 'checkpoint20.tar'
    main_test(args, num_sample=100,
              iswhole=False, istest=True, is1km=False,
              batch_size=16, num_workers=8)
    # accuracy assessment
    citylist = ['china', 'eu', 'usa']
    for city in citylist:
        args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
        main_test(args, num_sample=0, suffix=city,
                  iswhole=False, istest=True, is1km=False,
                  batch_size=16, num_workers=8,
                  issave=False)

    # predict for separate model
    # citylist = ['china', 'eu', 'usa']
    # for city in citylist:
    #     args = get_args(city=city)
    #     args.checkpoint = 'checkpoint20.tar'
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     # main_test(args, num_sample=200, suffix=city,
    #     #           iswhole=False, istest=True, is1km=False,
    #     #           batch_size=16, num_workers=8,
    #     #           issave=True)
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=False)

    # predict for combined model
    # args = get_args(city='globe')
    # citylist = ['china', 'eu', 'usa']
    # args.checkpoint = 'checkpoint20.tar'
    # for city in citylist:
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=False)

    # train on 4 bands: china
    # city = 'china'
    # city  = 'eu'
    # city = 'usa'
    # city = 'globe'
    # args = get_args(city=city)
    # args.maxepoch = 20
    #args.logdir = args.logdir+'_rgbn'
    # args.nchans = 8 #  6
    # args.nchanss2 = 4
    # main(args)
    # main_test(args, num_sample=0, suffix=city,
    #           iswhole=False, istest=True, is1km=False,
    #           batch_size=16, num_workers=8)
    # args = get_args(city='globe')
    # citylist = ['china', 'eu', 'usa']
    # # args.checkpoint = 'checkpoint20.tar'
    # args.checkpoint = 'checkpoint30.tar'
    # for city in citylist:
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=False)

    # train with new super-resolution weight files with the inverse sequence of RGB bands
    # results show there is little difference.
    # 2023.12.7
    # city = 'globe'
    # args = get_args(city=city)
    # args.maxepoch = 5
    # args.logdir = fr'D:\code\BHNetdata\realesrgan_feature_aggre_weight_{city}_hr'
    # args.logdirhr = fr'D:\code\BHNetdata\realesrgan_{city}\checkpoint.tar'
    # # args.rgbseq = [2, 1, 0]
    # # args.logdir = fr'D:\code\BHNetdata\realesrgan_feature_aggre_weight_{city}'
    # # args.logdirhr = fr'D:\code\BHNetdata\realesrgan\checkpoint.tar'
    # # main(args)
    # # args.checkpoint='checkpoint.tar'
    # # city='china'
    # city = 'usa'
    # args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    # main_test(args, num_sample=0, suffix=city,
    #           iswhole=False, istest=True, is1km=False,
    #           batch_size=16, num_workers=8,
    #           issave=False)

    # save test images: predict all test images
    # args = get_args(city='globe')
    # citylist = ['china', 'eu', 'usa']
    # args.checkpoint = 'checkpoint20.tar'
    # for city in citylist:
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=True)

    # predict for separate model
    # citylist = ['china', 'eu', 'usa']
    # for city in citylist:
    #     args = get_args(city=city)
    #     args.checkpoint = 'checkpoint20.tar'
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=True)

    # 2024.5.16: using pan and s1 bands
    # train on 4 bands: china
    # city = 'globe'
    # args = get_args(city=city)
    # args.maxepoch = 20
    # args.logdir = args.logdir+'_rgbn'
    # args.nchans = 8 #  6
    # args.nchanss2 = 4
    # main(args)
    # main_test(args, num_sample=0, suffix=city,
    #           iswhole=False, istest=True, is1km=False,
    #           batch_size=16, num_workers=8)
    # citylist = ['china', 'eu', 'usa']
    # args.checkpoint = 'checkpoint20.tar'
    # for city in citylist:
    #     args.testlist = f"datalist_{city}_test_0.7_test_0.3.csv"
    #     main_test(args, num_sample=0, suffix=city,
    #               iswhole=False, istest=True, is1km=False,
    #               batch_size=16, num_workers=8,
    #               issave=False)