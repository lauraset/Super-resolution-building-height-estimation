import os
import os.path as osp
from os.path import join
import pandas as pd
from glob import glob
import shutil
from multiprocessing import Pool
import cv2
from functools import partial
import time
from pathlib import Path


def get_file(ipath, respath, dirs=('sen2', 'bh', 'cbra')):
    '''
    :param ipath: subdir includes: sen2, bh, cbra
    :param respath:
    :return:
    '''
    imglist = os.listdir(join(ipath, dirs[0]))
    sen2 = [join(ipath, dirs[0], i) for i in imglist]
    bh = [join(ipath, dirs[1], i) for i in imglist]
    cbra = [join(ipath, dirs[2], i) for i in imglist]

    df = pd.DataFrame({dirs[0]:sen2, dirs[1]:bh, dirs[2]:cbra})
    df.to_csv(respath, sep=',', index=False, header=False)


def split_df(df, value=0, split_rate=0.9):
    df1 = df.loc[df[1]==value].sample(frac=1, random_state=1) # extract and shuffle
    num_train = int(len(df1)*split_rate)
    return df1[:num_train], df1[num_train:]


def split_data(datalist_path, split_rate=0.9, id='2', n1='train', n2='test'):
    data_dir = os.path.dirname(datalist_path)
    base_name = os.path.basename(datalist_path)[:-4]
    train_path = join(data_dir, base_name+'_'+ n1 + id +'.csv')
    test_path = join(data_dir, base_name+'_'+ n2 + id +'.csv')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('train and test list exist')
        return
    else:
        df = pd.read_csv(datalist_path, sep=',', header=None)
        # random sampling
        df1 = df.sample(frac=1, random_state=1) # shuffle
        num_train = int(len(df1) * split_rate)
        df_train = df1[:num_train]
        df_test = df1[num_train:]
        df_train.to_csv(train_path, index=False, sep=',', header=None)
        df_test.to_csv(test_path, index=False, sep=',', header=None)
        print('success')


def generate_allfile(ipath=r'D:\data\Landcover\samples62', subdir='s1_vvvhratio',
                     invalid='shenzhen', suffix='',
                     numsample=0):
    if not isinstance(subdir, list):
        subdir = [subdir,]

    # cat all tif images
    fplist = []
    for i in subdir:
        tmp = Path(os.path.join(ipath, i)).rglob('*.tif')
        fplist.extend(tmp)

    imglist = []
    for i in fplist:
        iname = i.stem+'.tif'
        if invalid is not None:
            if invalid not in iname:
                imglist.append(iname)
        else:
            imglist.append(iname)

    print(len(imglist))
    # if invalid is None:
    #     invalid=''
    # else:
    #     invalid='_del'+invalid

    df = pd.DataFrame({'imglist': imglist})
    # random select
    if numsample != 0:
        df = df.sample(n=numsample, random_state=1)

    respath = os.path.join(ipath, 'datalist_'+suffix+'.csv') #'datalist'+invalid+'_'+suffix+'.csv')
    if not os.path.exists(respath):
        df.to_csv(respath, header=False, index=False)

    # Train/Test
    # train/test=0.6:0.4
    split_data(respath, split_rate=0.7, id='_0.7', n1='train', n2='test')
    # val/test = 0.1:0.3
    respath = os.path.join(ipath, 'datalist_'+suffix+'_test_0.7.csv') #'datalist'+invalid+'_'+suffix+'_test_0.7.csv')
    split_data(respath, split_rate=0.33, id='_0.3', n1='val', n2='test')


# absolute path
def generate_allfile_abspath(ipath=r'D:\data\Landcover\samples62',
                             flist=('china', 'eu', 'usa'),
                             suffix='globe',
                             mergetype='',
                            numsample=0):
    # cat all tif images
    imglist = []
    citylist = []
    for city in flist:
        file = os.path.join(ipath, 'datalist_'+city+mergetype+'.csv')
        # read
        df = pd.read_csv(file, header=None)
        dflist = df[0].values.tolist()
        imglist.extend(dflist)
        num = len(dflist)
        citylist.extend([city]*num)

    print(len(imglist))
    s1list = ['s1'+i+'_check' for i in citylist]
    s2list = ['s2'+i+ '_check' for i in citylist]
    bhlist = ['bh'+i for i in citylist]

    df = pd.DataFrame({'imglist': imglist,
                       's1dir':s1list, 's2dir': s2list,
                       'bhdir': bhlist })
    # random select
    if numsample != 0:
        df = df.sample(n=numsample, random_state=1)

    respath = os.path.join(ipath, 'datalist_'+suffix+mergetype+'.csv') #'datalist'+invalid+'_'+suffix+'.csv')
    if not os.path.exists(respath):
        df.to_csv(respath, header=False, index=False)

    # Train/Test
    # train/test=0.6:0.4
    # split_data(respath, split_rate=0.7, id='_0.7', n1='train', n2='test')
    # # val/test = 0.1:0.3
    # respath = os.path.join(ipath, 'datalist_'+suffix+'_test_0.7.csv') #'datalist'+invalid+'_'+suffix+'_test_0.7.csv')
    # split_data(respath, split_rate=0.33, id='_0.3', n1='val', n2='test')


def concat_allfile(ipath=r'D:\data\Landcover\samples62',
                             flist=('china', 'eu', 'usa'),
                             suffix='globe',
                             mergetype='',
                            ):
    # cat all tif images
    df = []
    for city in flist:
        file = os.path.join(ipath, 'datalist_'+city+mergetype+'.csv')
        # read
        tmp = pd.read_csv(file, header=None)
        df.append(tmp)

    df = pd.concat(df)

    respath = os.path.join(ipath, 'datalist_'+suffix+mergetype+'.csv') #'datalist'+invalid+'_'+suffix+'.csv')
    if not os.path.exists(respath):
        df.to_csv(respath, header=False, index=False)

    # Train/Test
    # train/test=0.6:0.4
    # split_data(respath, split_rate=0.7, id='_0.7', n1='train', n2='test')
    # # val/test = 0.1:0.3
    # respath = os.path.join(ipath, 'datalist_'+suffix+'_test_0.7.csv') #'datalist'+invalid+'_'+suffix+'_test_0.7.csv')
    # split_data(respath, split_rate=0.33, id='_0.3', n1='val', n2='test')


def addabspath(ipath=r'D:\data\buildheight\samples',
        city='china', flist=None):
    if flist is None:
        fpath = os.path.join(ipath, 'datalistcopy')
        flist = [i for i in Path(fpath).glob('*'+city+'*.csv')]
    print(len(flist))
    subdir = {'s1': f's1{city}_check',
              's2': f's2{city}_check',
              'bh': f'bh{city}',
              'ge': f'ge{city}_check',
              'dem': f'dem{city}',
              'dsm': f'dsm{city}',
              }
    for file in flist:
        fname = file.name
        file = str(file)
        df = pd.read_csv(file, header=None)
        for k, v in subdir.items():
            df[k] = v
        resfile = os.path.join(ipath, fname)
        df.to_csv(resfile, header=False, index=False)


if __name__=="__main__":
    ipath = r'.\data'

    # 2023.11.07: generate samples for usa
    generate_allfile(ipath=ipath,
                     subdir='s1usa_check', invalid=None, suffix='usa')

    # 2023.11.9: generate samples for China,
    # delete shenzhen and randomly select 15000
    generate_allfile(ipath=ipath, subdir='s1china_check',
                     invalid='shenzhen', suffix='china',
                     numsample=15000)

    # 2023.11.9: generate samples for EU
    generate_allfile(ipath=ipath, subdir='s1eu_check',
                     invalid=None, suffix='eu',
                     )

    # 2023.11.9: generate samples for China, EU, USA
    # should merge the training and test data directly from the three countries
    # add s1,s2,bh dir to datalist
    addabspath(ipath, 'china')
    addabspath(ipath, 'eu')
    addabspath(ipath, 'usa')

    mergelist =['', '_test_0.7', '_train_0.7', '_test_0.7_test_0.3', '_test_0.7_val_0.3']
    for i in mergelist:
        concat_allfile(ipath=ipath,
            flist=('china', 'eu', 'usa'),
            suffix = 'globe',
            mergetype=i,
            )

