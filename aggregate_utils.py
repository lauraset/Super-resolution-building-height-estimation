import os
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Parameter
import rasterio as rio

def aggregate(data, scale):
    r, c = data.shape
    nr, nc = int(r*scale), int(r*scale)
    step = int(1/scale)
    res = np.zeros((nr, nc))
    data = data.astype('float')
    k=0
    for i in range(0, r, step):
        for j in range(0, c,step):
            m = int(i/step)
            n = int(j/step)
            patch = data[i:i+step, j:j+step]
            patch_area = (patch>0).sum()
            res[m, n] = patch.sum()/(patch_area+1e-6)
            k=k+1
    return res


def aggregate_torch(data, scale):
    step = int(1/scale)
    conv = torch.nn.Conv2d(1, 1,
                           kernel_size=step, stride=step, bias=False,
                           )
    conv.weight.requires_grad = False
    conv.weight = Parameter(torch.ones((1, 1, step, step)), requires_grad=False)
    s1 = conv(data)
    data_area = (data>=0).float() # changed from 1.0 to 0.0
    s2 = conv(data_area)
    res = s1/(s2+1e-10)
    res = res.squeeze() #.numpy()
    return res


def aggregate_torch_gpu(data, scale, device='cuda'):
    # h, w = data.shape
    # data = np.reshape(data, (1, 1, h, w))
    # data = torch.from_numpy(data).float()
    step = int(1/scale)
    conv = torch.nn.Conv2d(1, 1,
                           kernel_size=step, stride=step, bias=False,
                           )
    conv.weight.requires_grad = False
    conv.weight = Parameter(torch.ones((1, 1, step, step)), requires_grad=False)
    conv = conv.to(device)
    s1 = conv(data)
    data_area = (data>1.0).float()
    s2 = conv(data_area)
    res = s1/(s2+1e-10)
    return res


if __name__=="__main__":
    iname = 'Beijing_47.tif'
    datapath = os.path.join(r'D:\data\Landcover\samples62\bh', iname)
    respath = os.path.join('tmp', iname)
    scale = 0.25
    with rio.open(datapath, 'r') as src:
        data = src.read(1)
        t0 = time.time()
        res1 = aggregate(data, scale)
        t1 = time.time()
        print('%.6f'%(t1-t0))

        t0 = time.time()
        res2 = aggregate_torch(data, scale)
        t1 = time.time()
        print('%.6f' % (t1 - t0))
        # profile = src.profile
        # profile.update(dtype=np.float32, count=2)
        # with rio.open(respath, 'w', **profile) as dst:
        #     dst.write(res1, 1)
        #     dst.write(res2, 2)
    # res1 = aggregate(data, scale)
    # res2 = aggregate_torch(data, scale)
    #
    # diff = (res1-res2)
    # print(diff.min(), diff.max())
    # plt.subplot(1,3,1)
    # plt.imshow(data)
    # plt.subplot(1, 3, 2)
    # plt.imshow(res1)
    # plt.subplot(1, 3, 3)
    # plt.imshow(res2)
    # plt.show()
