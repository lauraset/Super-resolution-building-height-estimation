import math
import multiprocessing
import shutil

import cv2
import rasterio
import tifffile as tif
import os
from os.path import join
import numpy as np
import time
from osgeo import gdal, ogr
from copy import deepcopy
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import rasterio as rio
from functools import partial

def cal_mean_std(stats_s1):
    mean_all = []
    std_all = []
    for stats_b in stats_s1:
        imean = stats_b[:, 2]
        istd = stats_b[:, 3]

        ix2 = istd * istd + imean * imean
        ix2_all = ix2.mean()
        imean_all = imean.mean()
        istd_all = ix2_all - imean_all * imean_all
        istd_all = math.sqrt(istd_all)

        mean_all.append(imean_all)
        std_all.append(istd_all)

    print(mean_all)
    print(std_all)
    return mean_all, std_all

# 2% linear stretching
def cal_min_max(stats_s1, tmin=2, tmax=98):
    max_all = []
    min_all = []
    for stats_b in stats_s1:
        imin = stats_b[:, 0]
        imax = stats_b[:, 1]
        imin_all = np.percentile(imin, tmin) # the minimum
        imax_all = np.percentile(imax, tmax) # the maximum

        min_all.append(imin_all)
        max_all.append(imax_all)

    print(min_all)
    print(max_all)
    return min_all, max_all

def main_stats(ipath=r'D:\data\Landcover\samples',
               subdir='s1_vvvhratio', nband=3, resroot='datastats',
               imglistpath=None):

    s1path = join(ipath, subdir)

    # stats the mean and the std of the dataset
    if imglistpath is None:
        imglist = glob(join(s1path, '*.tif'))
    else:
        df = pd.read_csv(imglistpath,header=None)
        namelist = df[0].values
        imglist = [os.path.join(s1path, i) for i in namelist]

    num = len(imglist)
    print(num)

    # loop over each bands
    # nband = 3
    stats_s1 = [np.zeros((num, 4)) for _ in range(nband)]
    t0 = time.time()
    for i, rasterpath in enumerate(imglist[:num]):
        raster = gdal.Open(rasterpath, 0)
        for b in range(nband):
            band1 = raster.GetRasterBand(b+1)
            tmp = band1.ComputeStatistics(False)
            stats_s1[b][i, :] = deepcopy(tmp) # imin, imax, imean, istd
            # print(tmp)
        raster = None
    print('time: %.2f'%(time.time()-t0))
    np.save(join(resroot, subdir+'.npy'), stats_s1)

    # mean & std
    mean_all, std_all = cal_mean_std(stats_s1)
    min_all, max_all = cal_min_max(stats_s1, tmin=2, tmax=98)

    res = np.array([mean_all, std_all])
    np.savetxt(join(resroot, subdir + '_meanstd.txt'), res)

    res = np.array([min_all, max_all])
    np.savetxt(join(resroot, subdir + '_minmax.txt'), res)


# merge several results
def main_stats_merge(s1list = ('s1china_check', 's1usa_check', 's1eu_check'),
                     subdir='s1_vvvhratio',nband=2, resroot = 'datastatsglobe'):

    # load npy and merge all stats
    stats_s1 = [[] for _ in range(nband)]
    for i in s1list:
        tmp = os.path.join(resroot, i + '.npy')
        data = np.load(tmp)
        # print(data.shape)  # (2, 15000, 4) # band, numsample, image (imin, imax, imean, istd)
        for b, data_b in enumerate(data):
            stats_s1[b].append(data_b)

    # merge each part
    for b, data_b in enumerate(stats_s1):
        stats_s1[b] = np.concatenate(data_b, axis=0) # cat all dirs

    # mean & std, min & max

    mean_all, std_all = cal_mean_std(stats_s1)
    min_all, max_all = cal_min_max(stats_s1, tmin=2, tmax=98)

    res = np.array([mean_all, std_all])
    np.savetxt(join(resroot, subdir + '_meanstd.txt'), res)

    res = np.array([min_all, max_all])
    np.savetxt(join(resroot, subdir + '_minmax.txt'), res)


def main_stats_buildingheight(height_path, savepath, savename,
                              filelist=None):
    ################ stats for building height ##########
    if filelist is None:
        filelist = [str(i) for i in Path(height_path).rglob('*.tif')]
    else:
        df = pd.read_csv(filelist, header=None) # read from list
        namelist = df[0].values
        filelist = [os.path.join(height_path, i) for i in namelist]

    num = len(filelist)
    print(num)
    stats_path = os.path.join(savepath, savename+'.csv')
    if not os.path.exists(stats_path):
        # Loop over all images
        stats = np.zeros((256,))
        for file in tqdm(filelist):
            data = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if (data.shape[0]!=256) or (data.shape[1]!=256):
                print(file)
            unique, counts = np.unique(data, return_counts=True)
            stats[unique] += counts

        np.savetxt(stats_path[:-4]+'.txt', stats)
        df = pd.DataFrame(data={
            'height':np.arange(256), 'number': stats,'rate': stats/stats.sum()})
        df.to_csv(stats_path)

    # stats = np.loadtxt(stats_path)
    stats = pd.read_csv(stats_path, header=0, index_col=0)
    stats = stats.loc[:,'number'].values
    diff = num*(256*256)-stats.sum()
    print(diff)
    stats = stats/stats.sum()
    # plot
    x = np.arange(256)
    fig, ax = plt.subplots()
    ax.bar(x[:50], stats[:50])
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Number of floors')
    figpath = os.path.join(savepath, savename + '_plot.jpg')
    plt.savefig(figpath, dpi=300)
    plt.show()


# merge all heights from us, china, & eu
def main_stats_buildheight_merge(bhlist=('bh_stats_china','bh_stats_bhusa','bh_stats_bheu'),
                                 savepath='datastatsglobe',
                                 savename='bh_stats_globe'):
    stats = np.zeros((256,))
    for tmp in bhlist:
        tmppath = os.path.join(savepath, tmp+'.csv')
        data = pd.read_csv(tmppath, header=0, index_col=0)
        data = data['number'].values
        stats = stats + np.array(data)
    # save
    stats_path = os.path.join(savepath, savename)
    np.savetxt(stats_path + '.txt', stats)
    df = pd.DataFrame(data={
        'height': np.arange(256), 'number': stats, 'rate': stats / stats.sum()})
    df.to_csv(stats_path+'.csv')

    # show
    diff = 45000*(256*256)-stats.sum()
    print(diff)
    stats = stats/stats.sum()
    # plot
    x = np.arange(256)
    fig, ax = plt.subplots()
    ax.bar(x[:50], stats[:50])
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Number of floors')
    figpath = os.path.join(savepath, savename + '_plot.jpg')
    plt.savefig(figpath, dpi=300)
    plt.show()


# convert china building height
def floor2height(index, tifile, resdir='bhchina_height'):
    # tifile = Path(tifile)
    fbase = tifile.name
    # froot = tifile.parents[1]
    resfile = os.path.join(resdir, fbase)
    if os.path.exists(resfile):
        return
    else:
        tifile = str(tifile)
        with rasterio.open(tifile, mode='r') as src:
            profile = src.profile
            data = src.read(1)
            data[(data>0) & (data<3)] = 2 # set the minimum floor is 2
            data = data*3
        with rasterio.open(resfile, mode='w', **profile) as dst:
            dst.write(data, 1)


if __name__=="__main__":
    ipath = r'.\data'

    resroot = 'datastatsglobe'
    os.makedirs(resroot, exist_ok=True)
    
    # sentinel-1
    subdirlist = ['s1usa_check', 's1eu_check']
    nband = 2
    for subdir in subdirlist:
        main_stats(ipath, subdir, nband, resroot)

    # sentinel-2
    subdirlist = ['s2usa_check', 's2eu_check']
    nband = 6
    for subdir in subdirlist:
        main_stats(ipath, subdir, nband, resroot)

    # China, only use 15000 samples
    imglistpath = os.path.join(ipath, 'datalist_china.csv')
    main_stats(ipath=ipath,
               subdir='s1china_check', nband=2, resroot=resroot,
               imglistpath=imglistpath)
    main_stats(ipath=ipath,
               subdir='s2china_check', nband=6, resroot=resroot,
               imglistpath=imglistpath)

    ## BH: USA
    height_path = os.path.join(ipath, 'bhusa')
    savepath = 'datastatsglobe'
    savename = 'bh_stats_usa'
    main_stats_buildingheight(height_path, savepath, savename)

    ## BH: EU
    height_path = os.path.join(ipath, 'bheu')
    savepath = 'datastatsglobe'
    savename = 'bh_stats_eu'
    main_stats_buildingheight(height_path, savepath, savename)

    ## BH: China
    height_path = os.path.join(ipath, 'bhchina')
    savepath = 'datastatsglobe'
    savename = 'bh_stats_china'
    filelist = os.path.join(ipath, 'datalist_china.csv')
    main_stats_buildingheight(height_path, savepath, savename,
                              filelist=filelist)

    ################### calculte eu, china, usa all samples: Sentinel-1 & 2
    resroot = 'datastatsglobe'
    os.makedirs(resroot, exist_ok=True)
    subdirlist = ['s1china_check', 's1usa_check', 's1eu_check']
    main_stats_merge(s1list=subdirlist,
                     subdir='s1globe', nband=2, resroot='datastatsglobe')
    # [-7.571452465569495, -15.03005464321122]
    # [4.196955755865296, 3.875731021725474]
    # [-22.079342880249023, -28.524991188049317]
    # [23.191689529418944, 12.922104854583715]

    resroot = 'datastatsglobe'
    os.makedirs(resroot, exist_ok=True)
    subdirlist = ['s2china_check', 's2usa_check', 's2eu_check']
    main_stats_merge(s1list=subdirlist,
                     subdir='s2globe', nband=6, resroot='datastatsglobe')
    # [823.8878316840278, 1016.7921477213541, 1069.0579309624566, 2140.977429432509, 1910.2865483235678,
    #  1499.8546897894964]
    # [550.8267728801269, 577.0084996002943, 671.5732155180854, 794.0811566001486, 675.5460524441831, 635.715121669847]
    # [66.0, 113.0, 77.0, 80.0, 84.0, 69.0]
    # [7316.0, 7592.05999999999, 7854.039999999994, 7912.0, 6428.079999999987, 6414.0]

    ################ calculte eu, china, usa all samples: Building height
    main_stats_buildheight_merge(bhlist=('bh_stats_china', 'bh_stats_usa', 'bh_stats_eu'),
                                 savepath='datastatsglobe',
                                 savename='bh_stats_globe')

    # convert height in China to meters
    '''
    datapath = os.path.join(ipath, 'bhchina')
    respath = os.path.join(ipath, 'bhchina_height')
    os.makedirs(respath, exist_ok=True)

    filelist = [i for i in Path(datapath).glob('*.tif')]
    print(len(filelist))
    t0= time.time()
    pool = multiprocessing.Pool(10)
    pfunc = partial(floor2height, resdir=respath)
    pool.starmap(pfunc, enumerate(filelist))
    pool.close()
    t1 = time.time()
    print('time: %.3f'%(t1-t0))
    # 16550
    # time: 220.632
    '''
