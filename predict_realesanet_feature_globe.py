'''
predit the whole images
'''
import os
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from BH_loader import myImageFloder_S12_globe
from metrics import AverageMeter, acc2fileRMSE, SegmentationMetric, acc2file, HeightMetric, acc2fileHeight
from SR.rrdbnet_arch import RealESRGAN
from utils.preprocess import array2raster_rio, array2raster
from mymodels import SRRegress_Cls_feature
import shutil
from osgeo import gdal
import argparse
from losses_pytorch.selfloss import CE_DICE_adapt, MSE_adapt, MSE_adapt_weight, CE_DICE_adapt_weight
from BH_loader import gridimgLoader


def get_args(city='globe'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=r'.\data')
    parser.add_argument('--trainlist', default=f'datalist_{city}_train_0.7.csv')
    parser.add_argument('--vallist', default=f'datalist_{city}_test_0.7_val_0.3.csv')
    parser.add_argument('--testlist', default=f'datalist_{city}_test_0.7_test_0.3.csv')
    parser.add_argument('--logdir', default=fr'.\weights\realesrgan_feature_aggre_weight_{city}')
    parser.add_argument('--logdirhr', default=r'.\weights\realesrgan\checkpoint.tar')
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
    parser.add_argument('--cityname', type=list, default=['lanzhou', ]) # ningbo, tianjin, lanzhou
    # parser.add_argument('--grid', type=int, default=64)
    # parser.add_argument('--stride', type=int, default=60) # overlap 4 pixels =40m = 2.5m*16
    args = parser.parse_args()
    return args


def main_test(args, num_sample=100, suffix='',
              iswhole=False, istest=True, is1km=False, ispred=False,
              batch_size=1, num_workers=6, issave=False,
              respath=None, gridvalid=None):
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

    if iswhole:
        if respath is None:
            respath = os.path.join(logdir, 'pred_' + id+'_'+suffix)
            os.makedirs(respath, exist_ok=True)
        for cityname in args.cityname:
            if os.path.exists(os.path.join(respath, cityname+'_build.tif')):
                continue
            print('process: %s'%cityname)
            predict_whole_image_grid(args, cityname, net, net_hr.net_g, device,
                                start_epoch, respath=respath, gridvalid=gridvalid)


# 2023.12.1: only predict values on grids
def predict_whole_image_grid(args, cityname, model, net_hr, device,
                        epoch, respath=None,gridvalid='isv'):
    # load img
    dataset = gridimgLoader(rootname=args.wholeimgpath,
                 cityname=cityname, datastats=args.datastats,
                 normmethod='minmax', datarange=(0, 1),
                 s1dir=args.s1dir, s2dir=args.s2dir,
                 gridvalid=gridvalid, nchans=args.nchanss2)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    width = dataset.width * 4
    height = dataset.height * 4
    src_tif = dataset.s2path
    srcgeotrans = dataset.geotrans
    nres = srcgeotrans[1]/4.0 #

    res_height = np.zeros((height, width), dtype=np.uint16)
    res_build = np.zeros((args.chans_build, height, width), dtype=np.uint16)
    res_weight = np.zeros((height, width), dtype=np.uint8) # the initial weight is set to 0
    # predict
    model.eval()
    net_hr.eval()
    # acc_total = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    with torch.no_grad():
        for idx, (x, posall) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)

            hr_fea = net_hr.forward_feature(x[:, :3])  # RGB of sentinel-2 images
            ypred, build_pred = model.forward(x, hr_fea)  # N C H W
            ypred = ypred.cpu().numpy() # N 1 H W
            ypred[ypred<0] = 0 # set to postive
            ypred = np.round(ypred*10).astype(np.uint16)

            build_pred = torch.softmax(build_pred, dim=1).cpu().numpy() # N C H W -> N H W, [0,1]
            build_pred = np.round(build_pred*255).astype(np.uint16)

            # save
            n = x.shape[0]
            for i in range(n):
                [xoff, yoff, xcount, ycount] = posall[i]*4
                res_height[yoff:yoff+ycount, xoff:xoff+xcount] += ypred[i, 0, :ycount, :xcount]
                res_build[:, yoff:yoff + ycount, xoff:xoff + xcount] += build_pred[i, :, :ycount, :xcount]
                res_weight[yoff:yoff+ycount, xoff:xoff+xcount] += 1

            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}'.format(
                    epoch=epoch, batch=idx, iter=num))
            pbar.update()
        pbar.close()

        # res_build = res_build/res_weight
        # building prediction
        res_build = np.argmax(res_build, axis=0).astype(np.uint8) # C H W -> H W
        res_tif = os.path.join(respath, cityname + '_build.tif')
        array2raster_rio(res_tif, res_build, src_tif, bands=1, nresolution=nres)
        res_build = None

        # normalized by weight, only on valid region
        mask = (res_weight > 0)
        # res_weight = 1  # in case of zeros
        res_height[mask] = np.round(res_height[mask] / res_weight[mask]).astype(np.uint16)
        res_weight = None

        # save
        res_tif = os.path.join(respath, cityname+'_height.tif')
        array2raster(res_tif, res_height, src_tif, datatype=gdal.GDT_UInt16, nresolution=nres,
                     compressoption=['COMPRESS=DEFLATE', 'TILED=YES'])
        res_height = None


def getcitynamelist(args):
    suffix = '_s2.tif'
    flist = pathlib.Path(args.wholeimgpath).glob('*'+suffix)
    flist = [i.stem[:-3] for i in flist]
    print(len(flist))
    return flist


if __name__=="__main__":
    # predict on the whole images
    city = 'globe'
    args = get_args(city=city)
    args.checkpoint = 'checkpoint20.tar'
 
    isonamelist = ['chn_large', 'usa_large', 'europe_large',  'chn_metro', 'usa_metro', 'europe_metro']

    for isoname in isonamelist:
        args.wholeimgpath = r'.\data\urban\input_data\s2'+isoname
        # flist = getcitynamelist(args)
        main_test(args, num_sample=0, suffix='city'+isoname,
                iswhole=True, batch_size=16, num_workers=8, gridvalid='isv')

    # predict Jilin-1
    # args.wholeimgpath = r'.\data\jilin\dsm'
    # args.cityname = ['Changchun', 'Urumqi', 'mei']
    # main_test(args, num_sample=0, suffix='city',
    #           iswhole=True, batch_size=16, num_workers=8,
    #           respath = args.wholeimgpath, gridvalid=None)