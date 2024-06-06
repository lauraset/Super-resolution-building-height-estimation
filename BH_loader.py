import os
import torch
import pandas as pd
import numpy as np
import tifffile as tif
import cv2
import torch.utils.data as data
import albumentations as A
from imbalance.label_smooth import prepare_weights
from aggregate_utils import aggregate_torch
import math
from osgeo import gdal
from time import time
from mygeoinfo import create_latlon_mask
from mytrans import obtain_cutmix_box
from copy import deepcopy
import geopandas as gpd


image_transform = A.Compose([
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)

image_transform_strong = A.Compose([
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                  hue=0.25, always_apply=False, p=0.8)
]
)
# apply hierarchical weight
def hierweight(stats, hir):
    num_hier = len(hir)-1
    stats = stats/stats.sum()
    preweight = np.zeros((num_hier,))
    for i in range(num_hier):
        preweight[i] = stats[hir[i]:hir[i + 1]].sum()
    preweight = 1 / np.sqrt(preweight)  # inverse frequency
    # preweight = 1 / (preweight)
    preweight /= preweight.sum()
    scaling = num_hier / np.sum(preweight)
    preweight = scaling * preweight
    return preweight

# 2024.03.12: use simple frequency-inverse
def hierweight_simple(stats, hir):
    num_hier = len(hir)-1
    stats = stats/stats.sum()
    preweight = np.zeros((num_hier,))
    for i in range(num_hier):
        preweight[i] = stats[hir[i]:hir[i + 1]].sum()
    # preweight = 1 / np.sqrt(preweight)  # inverse frequency
    preweight = 1 / (preweight) # inverse frequency
    preweight /= preweight.sum()
    scaling = num_hier / np.sum(preweight)
    preweight = scaling * preweight
    return preweight

# 2024.3.12: weight equal to 1
def hierweight_equal(stats, hir):
    num_hier = len(hir) - 1
    preweight = np.ones((num_hier,))
    return preweight

# load sentinel-2, building height, footprint.
# add multi-scale building height data
class myImageFloder(data.Dataset):
    def __init__(self, datalist, aug=False, num_sample=0, multi_scale=False):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.multi_scale = multi_scale

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        height_path = self.datalist.iloc[index, 1]
        lab = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        #build_path = self.datalist.iloc[index, 2]
        #build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        #lab = np.stack((height, build), axis=2)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=lab)
            img = transformed["image"]
            lab = transformed["mask"]
        # Normalization
        img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # Multi-scale height maps
        if self.multi_scale:
            h, w = lab.shape
            s1 = cv2.resize(lab, dsize=(h // 4, w // 4), interpolation=cv2.INTER_NEAREST)
            s2 = cv2.resize(lab, dsize=(h // 2, w // 2), interpolation=cv2.INTER_NEAREST)
            lab = {"stage1": torch.from_numpy(s1).float(),
                   "stage2": torch.from_numpy(s2).float(),
                   "stage3": torch.from_numpy(lab).float(),}
        else:
            lab = torch.from_numpy(lab).float() # [0, C-1], 0,1,2 index
        return img, lab

    def __len__(self):
        return len(self.datalist)


# load HR images
class myImageFloder_HR(data.Dataset):
    def __init__(self, datalist, aug=False, num_sample=0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        lr = tif.imread(img_path)
        # hr images
        rootname = os.path.dirname(os.path.dirname(img_path))
        basename = os.path.basename(img_path)
        hr_path = os.path.join(rootname, 'ge2', basename)
        hr = tif.imread(hr_path)
        # height
        height_path = self.datalist.iloc[index, 1]
        height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        # build footprint
        build_path = self.datalist.iloc[index, 2]
        build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        build = build/255 # convert to 0,1
        lab = np.stack((height, build), axis=2)
        h, w = lr.shape[:2]
        lr = cv2.resize(lr, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        img = np.concatenate((lr, hr), axis=-1) # concatenate
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=lab)
            img = transformed["image"]
            lab = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img).permute(2,0,1).float()
        img = img/255.0
        lr = img[:4, :, :]
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=0.25, mode="nearest")
        lr = lr.squeeze(0)
        hr = img[4:, :, :]
        # Multi-scale height maps
        height = torch.from_numpy(lab[:,:,0]).float() # [0, C-1], 0,1,2 index
        build = torch.from_numpy(lab[:,:,1]).long()
        return lr, hr, height, build

    def __len__(self):
        return len(self.datalist)


# load HR images & sentinel-1 images: 2023.9.5
class myImageFloder_S12_HR(data.Dataset):
    def __init__(self, datalist, aug=False, num_sample=0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        rootname = os.path.dirname(os.path.dirname(img_path))
        basename = os.path.basename(img_path)
        # s2 & s1 images
        s2 = tif.imread(img_path)
        s1 = tif.imread(os.path.join(rootname, 'sen1', basename))
        lr = np.concatenate((s2, s1), axis=-1)
        # hr images
        hr_path = os.path.join(rootname, 'ge2', basename)
        hr = tif.imread(hr_path)
        # height
        height_path = self.datalist.iloc[index, 1]
        height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        # build footprint
        build_path = self.datalist.iloc[index, 2]
        build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        build = build/255 # convert to 0,1
        lab = np.stack((height, build), axis=2)
        h, w, bands = lr.shape
        lr = cv2.resize(lr, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        img = np.concatenate((lr, hr), axis=-1) # concatenate
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=lab)
            img = transformed["image"]
            lab = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img).permute(2,0,1).float()
        img = img/255.0
        lr = img[:bands, :, :]
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=0.25, mode="nearest")
        lr = lr.squeeze(0)
        hr = img[bands:, :, :]
        # Multi-scale height maps
        height = torch.from_numpy(lab[:,:,0]).float() # [0, C-1], 0,1,2 index
        build = torch.from_numpy(lab[:,:,1]).long()
        return lr, hr, height, build

    def __len__(self):
        return len(self.datalist)


# 2023.9.19: used for the whole China
class myImageFloder_S12_HR_china(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f's1_vvvhratio_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f's2_rgbnir_{normmethod}.txt'))
        self.normge = np.loadtxt(os.path.join(datastats, f'ge_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
            self.normge[1] -= self.normge[0] # max-min
        self.datarange = datarange

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        img_path = os.path.join(self.rootname, 's2_rgbnir', basename)
        # s2 & s1 images
        s2 = tif.imread(img_path)
        s1 = tif.imread(os.path.join(self.rootname, 's1_vvvhratio', basename))
        lr = np.concatenate((s2, s1), axis=-1)
        # hr images
        hr_path = os.path.join(self.rootname, 'ge', basename)
        hr = tif.imread(hr_path)
        # height
        height_path = os.path.join(self.rootname, 'bh', basename)
        height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        # build footprint
        build_path = os.path.join(self.rootname, 'cbra', basename)
        build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        build = (build==255).astype('uint8') # convert to 0,1
        # cat
        lab = np.stack((height, build), axis=2)
        h, w, bs1 = s1.shape
        _, _, bs2 = s2.shape
        bs12 = bs1+bs2
        lr = cv2.resize(lr, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        img = np.concatenate((lr, hr), axis=-1) # concatenate
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=lab)
            img = transformed["image"]
            lab = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        img[:, :, bs2:bs12] = (img[:, :, bs2:bs12] - self.norms1[0]) / self.norms1[1] # sentinel-1
        img[:, :, bs12:] = (img[:, :, bs12:] - self.normge[0]) / self.normge[1]
        img = img.permute(2, 0, 1)  # C H W
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]

        lr = img[:bs12, :, :]
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=0.25, mode="nearest")
        lr = lr.squeeze(0)
        hr = img[bs12:, :, :]
        # Multi-scale height maps
        height = torch.from_numpy(lab[:,:,0]).float() # [0, C-1], 0,1,2 index
        build = torch.from_numpy(lab[:,:,1]).long()
        if self.aug:
            return lr, hr, height, build
        else:
            return lr, hr, height, build, img_path

    def __len__(self):
        return len(self.datalist)


# 2023.9.25: used for the whole China, only read s1 & s2
class myImageFloder_S12_china(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 isheight=3.0, preweight=None,
                 max_target=21,
                 normheight=1.0,
                 isaggre=False, ishir=False, hir=(0, 1, 4, 7, 10, 20, 30, 255)):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]# self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f's1_vvvhratio_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f's2_rgbnir_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
        self.datarange = datarange
        # self.isheight = isheight
        num_hier = len(hir)-1
        self.heightweight = np.ones((num_hier,))
        if preweight is not None:
            stats = np.loadtxt(preweight)
            self.heightweight = hierweight(stats, hir)
            # self.preweight = prepare_weights(stats, reweight='sqrt_inv', max_target=max_target,
            #                             lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        # self.max_target = max_target
        # self.normheight = normheight
        self.isaggre = isaggre
        self.ishir = ishir
        if ishir:
            self.buildhir = np.zeros((255,), dtype='uint8')
            for i in range(num_hier):
                self.buildhir[hir[i]:hir[i+1]] = i

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        img_path = os.path.join(self.rootname, 's2_rgbnir', basename)
        # s2 & s1 images
        s2 = tif.imread(img_path)
        s1 = tif.imread(os.path.join(self.rootname, 's1_vvvhratio', basename))
        img = np.concatenate((s2, s1), axis=-1)
        # height
        height_path = os.path.join(self.rootname, 'bh', basename)
        if os.path.exists(height_path):
            height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        else:
            height = np.ones((256,256), dtype=np.uint8) # if do not exist
        # build footprint
        # build_path = os.path.join(self.rootname, 'cbra', basename)
        # build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        # build = (build==255).astype('uint8') # convert to 0,1
        # cat
        # lab = np.stack((height, build), axis=2)
        h, w, bs2 = s2.shape
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=height)
            img = transformed["image"]
            height = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1] # sentinel-1
        img = img.permute(2, 0, 1).unsqueeze(0)  # 1 C H W
        img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode="nearest")
        img = img.squeeze(0)
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]
        # Multi-scale height maps
        height[(height<3) & (height>0)] = 3 # set the minimum to 3 floors
        # height[height>20] = 20 # set the minimum to 20 floors
        if self.ishir:
            build = self.buildhir[height]
            heightweight = self.heightweight[build]
        else:
            build = (height > 0)
            heightweight = np.ones_like(build)

        build = torch.from_numpy(build).long()
        height = torch.from_numpy(height).float()  # [0, C-1], 0,1,2 index
        heightweight = torch.from_numpy(heightweight).float()

        if self.isaggre:
            h, w = height.shape
            height_aggre = aggregate_torch(height.reshape((1, 1, h, w)), scale=0.25)
            height = [height*3, height_aggre*3]
            # weight
            build_aggre = self.buildhir[height_aggre.long().numpy()]
            heightweight_aggre = self.heightweight[build_aggre]
            heightweight_aggre = torch.from_numpy(heightweight_aggre).float()
            heightweight = [heightweight, heightweight_aggre]

        if (self.aug) and (self.heightweight is not None):
            return img, height, build, heightweight
        if (self.aug) and (self.heightweight is None):
            return img, height, build
        else:
            return img, height, build, img_path

    def __len__(self):
        return len(self.datalist)

# 2023.11.7: load globe image
class myImageFloder_S12_globe(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 s1dir='s1', s2dir='s2', heightdir='bh', preweight=None,
                 isaggre=False, ishir=False, hir=(0, 3, 12, 21, 30, 60, 90, 256),
                 nchans=6, weightmethod='sqrt'):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.nchans = nchans
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        count = len(self.datalist.columns)
        if count==1: # add columns
            self.datalist[s1dir] = s1dir
            self.datalist[s2dir] = s2dir
            self.datalist[heightdir] = heightdir
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample] # self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        self.norms2 = self.norms2[:, :self.nchans]
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
        self.datarange = datarange
        # self.isheight = isheight
        num_hier = len(hir)-1
        self.heightweight = np.ones((num_hier,))
        if preweight is not None:
            stats = np.loadtxt(preweight)
            self.heightweight = hierweight(stats, hir)
            if weightmethod=='simple':
                self.heightweight = hierweight_simple(stats, hir)
            elif weightmethod=='equal':
                self.heightweight = hierweight_equal(stats, hir)
            else:
                pass
            # self.preweight = prepare_weights(stats, reweight='sqrt_inv', max_target=max_target,
            #                             lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        # self.max_target = max_target
        # self.normheight = normheight
        self.isaggre = isaggre
        self.ishir = ishir
        if ishir:
            self.buildhir = np.zeros((256,), dtype='uint8')
            for i in range(num_hier):
                self.buildhir[hir[i]:hir[i+1]] = i

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s1dir = self.datalist.iloc[index, 1]
        s2dir = self.datalist.iloc[index, 2]
        bhdir = self.datalist.iloc[index, 3]
        img_path = os.path.join(self.rootname, s2dir, basename)
        # s2 & s1 images
        s2 = tif.imread(os.path.join(self.rootname, s2dir, basename))[:, :, :self.nchans]
        s1 = tif.imread(os.path.join(self.rootname, s1dir, basename))
        img = np.concatenate((s2, s1), axis=-1)
        # height
        height_path = os.path.join(self.rootname, bhdir, basename)
        if os.path.exists(height_path):
            height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        else:
            height = np.ones((256, 256), dtype=np.uint8) # if do not exist
        # build footprint
        # build_path = os.path.join(self.rootname, 'cbra', basename)
        # build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        # build = (build==255).astype('uint8') # convert to 0,1
        # cat
        # lab = np.stack((height, build), axis=2)
        h, w, bs2 = s2.shape
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=height)
            img = transformed["image"]
            height = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1] # sentinel-1
        img = img.permute(2, 0, 1).unsqueeze(0)  # 1 C H W
        img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode="nearest")
        img = img.squeeze(0)
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]
        # Multi-scale height maps
        # height[(height<3) & (height>0)] = 3 # set the minimum to 3 floors
        # height[height>20] = 20 # set the minimum to 20 floors
        if self.ishir:
            build = self.buildhir[height]
            heightweight = self.heightweight[build]
        else:
            build = (height > 0)
            heightweight = np.ones_like(build)

        build = torch.from_numpy(build).long()
        height = torch.from_numpy(height).float()  # [0, C-1], 0,1,2 index
        heightweight = torch.from_numpy(heightweight).float()

        if self.isaggre:
            h, w = height.shape
            height_aggre = aggregate_torch(height.reshape((1, 1, h, w)), scale=0.25)
            height = [height, height_aggre]
            # weight
            build_aggre = self.buildhir[height_aggre.long().numpy()]
            heightweight_aggre = self.heightweight[build_aggre]
            heightweight_aggre = torch.from_numpy(heightweight_aggre).float()
            heightweight = [heightweight, heightweight_aggre]

        if self.aug:
            return img, height, build, heightweight
        else:
            return img, height, build, img_path

    def __len__(self):
        return len(self.datalist)


# 2023.12.10: only consider s2 image
class myImageFloder_S2_globe(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 s1dir='s1', s2dir='s2', heightdir='bh', preweight=None,
                 isaggre=False, ishir=False, hir=(0, 3, 12, 21, 30, 60, 90, 256),
                 nchans=6):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.nchans = nchans
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        count = len(self.datalist.columns)
        if count==1: # add columns
            self.datalist[s1dir] = s1dir
            self.datalist[s2dir] = s2dir
            self.datalist[heightdir] = heightdir
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample] # self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        self.norms2 = self.norms2[:, :self.nchans]
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
        self.datarange = datarange
        # self.isheight = isheight
        num_hier = len(hir)-1
        self.heightweight = np.ones((num_hier,))
        if preweight is not None:
            stats = np.loadtxt(preweight)
            self.heightweight = hierweight(stats, hir)
            # self.preweight = prepare_weights(stats, reweight='sqrt_inv', max_target=max_target,
            #                             lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        # self.max_target = max_target
        # self.normheight = normheight
        self.isaggre = isaggre
        self.ishir = ishir
        if ishir:
            self.buildhir = np.zeros((256,), dtype='uint8')
            for i in range(num_hier):
                self.buildhir[hir[i]:hir[i+1]] = i

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        # s1dir = self.datalist.iloc[index, 1]
        s2dir = self.datalist.iloc[index, 2]
        bhdir = self.datalist.iloc[index, 3]
        img_path = os.path.join(self.rootname, s2dir, basename)
        # s2 & s1 images
        img = tif.imread(os.path.join(self.rootname, s2dir, basename))[:, :, :self.nchans]
        # s1 = tif.imread(os.path.join(self.rootname, s1dir, basename))
        # img = np.concatenate((s2, s1), axis=-1)
        # height
        height_path = os.path.join(self.rootname, bhdir, basename)
        if os.path.exists(height_path):
            height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        else:
            height = np.ones((256, 256), dtype=np.uint8) # if do not exist
        # build footprint
        # build_path = os.path.join(self.rootname, 'cbra', basename)
        # build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        # build = (build==255).astype('uint8') # convert to 0,1
        # cat
        # lab = np.stack((height, build), axis=2)
        h, w, bs2 = img.shape
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=height)
            img = transformed["image"]
            height = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        # img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1] # sentinel-1
        img = img.permute(2, 0, 1).unsqueeze(0)  # 1 C H W
        img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode="nearest")
        img = img.squeeze(0)
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]
        # Multi-scale height maps
        # height[(height<3) & (height>0)] = 3 # set the minimum to 3 floors
        # height[height>20] = 20 # set the minimum to 20 floors
        if self.ishir:
            build = self.buildhir[height]
            heightweight = self.heightweight[build]
        else:
            build = (height > 0)
            heightweight = np.ones_like(build)

        build = torch.from_numpy(build).long()
        height = torch.from_numpy(height).float()  # [0, C-1], 0,1,2 index
        heightweight = torch.from_numpy(heightweight).float()

        if self.isaggre:
            h, w = height.shape
            height_aggre = aggregate_torch(height.reshape((1, 1, h, w)), scale=0.25)
            height = [height, height_aggre]
            # weight
            build_aggre = self.buildhir[height_aggre.long().numpy()]
            heightweight_aggre = self.heightweight[build_aggre]
            heightweight_aggre = torch.from_numpy(heightweight_aggre).float()
            heightweight = [heightweight, heightweight_aggre]

        if self.aug:
            return img, height, build, heightweight
        else:
            return img, height, build, img_path

    def __len__(self):
        return len(self.datalist)


# 2024.3.12: only consider s1 images
class myImageFloder_S1_globe(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 s1dir='s1', s2dir='s2', heightdir='bh', preweight=None,
                 isaggre=False, ishir=False, hir=(0, 3, 12, 21, 30, 60, 90, 256),
                 nchans=6):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.nchans = nchans
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        count = len(self.datalist.columns)
        if count==1: # add columns
            self.datalist[s1dir] = s1dir
            self.datalist[s2dir] = s2dir
            self.datalist[heightdir] = heightdir
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample] # self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        self.norms2 = self.norms2[:, :self.nchans]
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
        self.datarange = datarange
        # self.isheight = isheight
        num_hier = len(hir)-1
        self.heightweight = np.ones((num_hier,))
        if preweight is not None:
            stats = np.loadtxt(preweight)
            self.heightweight = hierweight(stats, hir)
            # self.preweight = prepare_weights(stats, reweight='sqrt_inv', max_target=max_target,
            #                             lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        # self.max_target = max_target
        # self.normheight = normheight
        self.isaggre = isaggre
        self.ishir = ishir
        if ishir:
            self.buildhir = np.zeros((256,), dtype='uint8')
            for i in range(num_hier):
                self.buildhir[hir[i]:hir[i+1]] = i

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s1dir = self.datalist.iloc[index, 1]
        # s2dir = self.datalist.iloc[index, 2]
        bhdir = self.datalist.iloc[index, 3]
        img_path = os.path.join(self.rootname, s1dir, basename)
        # s2 & s1 images
        # s2 = tif.imread(os.path.join(self.rootname, s2dir, basename))[:, :, :self.nchans]
        img = tif.imread(os.path.join(self.rootname, s1dir, basename))
        # img = np.concatenate((s2, s1), axis=-1)
        # height
        height_path = os.path.join(self.rootname, bhdir, basename)
        if os.path.exists(height_path):
            height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        else:
            height = np.ones((256, 256), dtype=np.uint8) # if do not exist
        # build footprint
        # build_path = os.path.join(self.rootname, 'cbra', basename)
        # build = cv2.imread(build_path, cv2.IMREAD_UNCHANGED)
        # build = (build==255).astype('uint8') # convert to 0,1
        # cat
        # lab = np.stack((height, build), axis=2)
        h, w, bs2 = img.shape
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=height)
            img = transformed["image"]
            height = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        # img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        # img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1] # sentinel-1
        img = (img - self.norms1[0]) / self.norms1[1]  # sentinel-1

        img = img.permute(2, 0, 1).unsqueeze(0).float()  # 1 C H W
        img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode="nearest")
        img = img.squeeze(0)
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]
        # Multi-scale height maps
        # height[(height<3) & (height>0)] = 3 # set the minimum to 3 floors
        # height[height>20] = 20 # set the minimum to 20 floors
        if self.ishir:
            build = self.buildhir[height]
            heightweight = self.heightweight[build]
        else:
            build = (height > 0)
            heightweight = np.ones_like(build)

        build = torch.from_numpy(build).long()
        height = torch.from_numpy(height).float()  # [0, C-1], 0,1,2 index
        heightweight = torch.from_numpy(heightweight).float()

        if self.isaggre:
            h, w = height.shape
            height_aggre = aggregate_torch(height.reshape((1, 1, h, w)), scale=0.25)
            height = [height, height_aggre]
            # weight
            build_aggre = self.buildhir[height_aggre.long().numpy()]
            heightweight_aggre = self.heightweight[build_aggre]
            heightweight_aggre = torch.from_numpy(heightweight_aggre).float()
            heightweight = [heightweight, heightweight_aggre]

        if self.aug:
            return img, height, build, heightweight
        else:
            return img, height, build, img_path

    def __len__(self):
        return len(self.datalist)


# 2023.11.20: add strong augmentation to images
class myImageFloder_S12_globe_strong(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 s1dir='s1', s2dir='s2', heightdir='bh', preweight=None,
                 isaggre=False, ishir=False, hir=(0, 3, 12, 21, 30, 60, 90, 255)):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        count = len(self.datalist.columns)
        if count==1: # add columns
            self.datalist[s1dir] = s1dir
            self.datalist[s2dir] = s2dir
            self.datalist[heightdir] = heightdir
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample] # self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s1dir = self.datalist.iloc[index, 1]
        s2dir = self.datalist.iloc[index, 2]
        img_path = os.path.join(self.rootname, s2dir, basename)
        # s2 & s1 images
        s2 = tif.imread(os.path.join(self.rootname, s2dir, basename))
        s2 = (s2 - self.norms2[0])/self.norms2[1]
        s1 = tif.imread(os.path.join(self.rootname, s1dir, basename))
        s1 = (s1 -self.norms1[0])/self.norms1[1]
        img_w = np.concatenate((s2, s1), axis=-1)
        # Augmentation
         # weak augmentation
        img_w = image_transform(image=img_w)["image"]
        rgb = (img_w[:,:,:3]*255).astype('uint8')
        # strong augmentation
        img_s1 = deepcopy(img_w)
        rgb = image_transform_strong(image=rgb)["image"]
        img_s1[:, :, :3] = rgb/255.
        # img_s2 = image_transform_strong(image=img_w)["image"]
        # Normalization
        img_w = torch.from_numpy(img_w).float().permute(2, 0, 1)  # .permute(2,0,1)
        img_s1 = torch.from_numpy(img_s1).float().permute(2, 0, 1)
        # img_s2 = torch.from_numpy(img_s2).float().permute(2, 0, 1)
        # cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        return img_w, img_s1 #, cutmix_box1

    def __len__(self):
        return len(self.datalist)


# 2023.11.15: add geoinformation
class myImageFloder_S12_globe_geo(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0,
                 s1dir='s1', s2dir='s2', heightdir='bh',
                 preweight=None,
                 isaggre=False, ishir=False, hir=(0, 3, 12, 21, 30, 60, 90, 255)):
        # hir = (0, 1, 4, 7, 10, 20, 30, 255)
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        count = len(self.datalist.columns)
        if count==1: # add columns
            self.datalist[s1dir] = s1dir
            self.datalist[s2dir] = s2dir
            self.datalist[heightdir] = heightdir
            # self.datalist[demdir] = demdir
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample] # self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0] # max-min
        self.datarange = datarange
        # self.isheight = isheight
        num_hier = len(hir)-1
        self.heightweight = np.ones((num_hier,))
        if preweight is not None:
            stats = np.loadtxt(preweight)
            self.heightweight = hierweight(stats, hir)

        self.isaggre = isaggre
        self.ishir = ishir
        if ishir:
            self.buildhir = np.zeros((256,), dtype='uint8')
            for i in range(num_hier):
                self.buildhir[hir[i]:hir[i+1]] = i

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s1dir = self.datalist.iloc[index, 1]
        s2dir = self.datalist.iloc[index, 2]
        bhdir = self.datalist.iloc[index, 3]
        #demdir = self.datalist.iloc[index, 4]
        img_path = os.path.join(self.rootname, s2dir, basename)
        # s2 & s1 images
        s2 = tif.imread(os.path.join(self.rootname, s2dir, basename))
        s2 = (s2 - self.norms2[0])/self.norms2[1]
        s1 = tif.imread(os.path.join(self.rootname, s1dir, basename))
        s1 = (s1 -self.norms1[0])/self.norms1[1]
        # height
        height_path = os.path.join(self.rootname, bhdir, basename)
        if os.path.exists(height_path):
            height = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        else:
            height = np.ones((256, 256), dtype=np.uint8) # if do not exist
        # dem
        # dem = tif.imread(os.path.join(self.rootname, demdir, basename))/10000.0 # 0-1
        lat_mask, lon_mask = create_latlon_mask(img_path)
        lat_mask = (lat_mask+90.0)/180.0 # (-90,90): 0-1
        lon_mask = (lon_mask+180.0)/360.0 # (-180, 180): 0-1
        # dem = np.expand_dims(dem, axis=2)
        lat_mask = np.expand_dims(lat_mask, axis=2)
        lon_mask = np.expand_dims(lon_mask, axis=2)
        img = np.concatenate((s2, s1, lat_mask, lon_mask), axis=-1)
        h, w = s2.shape[:2]
        img = cv2.resize(img, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=height)
            img = transformed["image"]
            height = transformed["mask"]
        # Normalization
        img = torch.from_numpy(img).float()  # .permute(2,0,1)
        img = img.permute(2, 0, 1).unsqueeze(0)  # 1 C H W
        img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode="nearest")
        img = img.squeeze(0)
        if self.ishir:
            build = self.buildhir[height]
            heightweight = self.heightweight[build]
        else:
            build = (height > 0)
            heightweight = np.ones_like(build)

        build = torch.from_numpy(build).long()
        height = torch.from_numpy(height).float()  # [0, C-1], 0,1,2 index
        heightweight = torch.from_numpy(heightweight).float()

        if self.isaggre:
            h, w = height.shape
            height_aggre = aggregate_torch(height.reshape((1, 1, h, w)), scale=0.25)
            height = [height, height_aggre]
            # weight
            build_aggre = self.buildhir[height_aggre.long().numpy()]
            heightweight_aggre = self.heightweight[build_aggre]
            heightweight_aggre = torch.from_numpy(heightweight_aggre).float()
            heightweight = [heightweight, heightweight_aggre]

        if self.aug:
            return img, height, build, heightweight
        else:
            return img, height, build, img_path

    def __len__(self):
        return len(self.datalist)


# 2023.9.12: super-resolution network, input: HR, output: LR
class myImageFloderLRHR(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastats',
                 normmethod='meanstd', datarange=(0, 1),
                 aug=False, num_sample=0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms2 = np.loadtxt(os.path.join(datastats, f's2_rgbnir_{normmethod}.txt'))
        self.normge = np.loadtxt(os.path.join(datastats, f'ge_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms2[1] -= self.norms2[0] # max-min
            self.normge[1] -= self.normge[0] # max-min
        self.image_transform = A.Compose([
                        A.Flip(p=0.5),
                        A.RandomGridShuffle(grid=(2, 2), p=0.5),
                        A.Rotate(p=0.5, interpolation=cv2.INTER_LINEAR),])
        self.datarange = datarange

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        img_path = os.path.join(self.rootname, 's2_rgbnir', basename)
        # s2
        lr = tif.imread(img_path)
        # hr images
        hr_path = os.path.join(self.rootname, 'ge', basename)
        hr = tif.imread(hr_path)

        h, w, bs2 = lr.shape
        lr = cv2.resize(lr, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        img = np.concatenate((lr, hr), axis=-1) # concatenate
        # Augmentation
        if self.aug:
            transformed = self.image_transform(image=img)
            img = transformed["image"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[..., :bs2] = (img[..., :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        img[..., bs2:] = (img[..., bs2:] - self.normge[0]) / self.normge[1]
        img = img.permute(2, 0, 1)  # C H W

        lr = img[:bs2, :, :]
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=0.25, mode="nearest")
        lr = lr.squeeze(0)
        if isinstance(self.datarange, tuple):
            lr[lr < self.datarange[0]] = self.datarange[0]
            lr[lr > self.datarange[1]] = self.datarange[1]
        hr = img[bs2:, :, :]
        if self.aug:
            return lr, hr
        else:
            return lr, hr, img_path

    def __len__(self):
        return len(self.datalist)

# 2023.12.4:
class myImageFloderLRHRglobe(data.Dataset):
    def __init__(self, datalist, rootname, datastats='datastatsglobe',
                 normmethod='minmax', datarange=(0, 1),
                 s2dir='s2', gedir='ge', nchans=3,
                 aug=False, num_sample=0):
        self.nchans = nchans
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist.sample(n=num_sample, random_state=0)
        self.aug = aug # augmentation for images
        self.rootname = rootname
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        self.norms2 = self.norms2[:, :self.nchans] # bug before
        self.normge = np.loadtxt(os.path.join(datastats, f'{gedir}_{normmethod}.txt'))
        if normmethod=='minmax':
            self.norms2[1] -= self.norms2[0] # max-min
            self.normge[1] -= self.normge[0] # max-min
        self.image_transform = A.Compose([
                        A.Flip(p=0.5),
                        A.RandomGridShuffle(grid=(2, 2), p=0.5),
                        A.Rotate(p=0.5, interpolation=cv2.INTER_LINEAR),])
        self.datarange = datarange

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s2dir = self.datalist.iloc[index, 2]
        gedir = self.datalist.iloc[index, 4]
        img_path = os.path.join(self.rootname, s2dir, basename)
        # s2
        # lr = tif.imread(img_path)[:, :, [2, 1, 0]] # convert to RGB
        lr = tif.imread(img_path)[:, :, :self.nchans] # bgr
        # hr images
        hr_path = os.path.join(self.rootname, gedir, basename)
        hr = tif.imread(hr_path)

        h, w, bs2 = lr.shape
        lr = cv2.resize(lr, dsize=(4*h, 4*w), interpolation=cv2.INTER_NEAREST)
        img = np.concatenate((lr, hr), axis=-1) # concatenate
        # Augmentation
        if self.aug:
            transformed = self.image_transform(image=img)
            img = transformed["image"]
        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[..., :bs2] = (img[..., :bs2] - self.norms2[0]) / self.norms2[1] # sentinel-2
        img[..., bs2:] = (img[..., bs2:] - self.normge[0]) / self.normge[1]
        img = img.permute(2, 0, 1)  # C H W

        lr = img[:bs2, :, :]
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=0.25, mode="nearest")
        lr = lr.squeeze(0)
        if isinstance(self.datarange, tuple):
            lr[lr < self.datarange[0]] = self.datarange[0]
            lr[lr > self.datarange[1]] = self.datarange[1]
        hr = img[bs2:, :, :]
        if self.aug:
            return lr, hr
        else:
            return lr, hr, img_path

    def __len__(self):
        return len(self.datalist)


# load VRT images
def load_s12(rootname=r'D:\data\Landcover\s12range',
                 cityname='beijing', datastats='datastats',
                normmethod='minmax', datarange=(0, 1),
                grid=1024, stride=24):
    stride = grid - stride
    norms1 = np.loadtxt(os.path.join(datastats, f's1_vvvhratio_{normmethod}.txt'))
    norms2 = np.loadtxt(os.path.join(datastats, f's2_rgbnir_{normmethod}.txt'))
    if normmethod == 'minmax':
        norms1[1] -= norms1[0]  # max-min
        norms2[1] -= norms2[0]  # max-min

    s2path = os.path.join(rootname, cityname+'_'+'s2_rgbnir'+'_clip.tif')
    s1path = os.path.join(rootname, cityname+'_'+'s1_vvvhratio'+'_clip.tif')
    # convert VRT to tif images
    if not os.path.exists(s2path):
        gdal.Translate(s2path, s2path[:-4]+'.vrt')
    if not os.path.exists(s1path):
        gdal.Translate(s1path, s1path[:-4] + '.vrt')

    # s2 & s1 images:  read sentinel-1/2, and normalization in numpy format
    s2 = tif.imread(s2path)
    bs2 = s2.shape[-1]
    s1 = tif.imread(s1path)

    img = np.concatenate((s2, s1), axis=-1)
    img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
    img[:, :, :bs2] = (img[:, :, :bs2] - norms2[0]) / norms2[1]  # sentinel-2
    img[:, :, bs2:] = (img[:, :, bs2:] - norms1[0]) / norms1[1]  # sentinel-1
    if isinstance(datarange, tuple):
        img[img < datarange[0]] = datarange[0]
        img[img > datarange[1]] = datarange[1]
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, axes=(0, 3, 1, 2)) # N B H W
    # pad img
    n, b, h, w = img.shape
    rows = math.ceil((h - grid) / (stride)) * stride + grid
    rows = int(rows)
    cols = math.ceil((w - grid) / (stride)) * stride + grid
    cols = int(cols)
    print('rows is {}, cols is {}'.format(rows, cols))
    img = np.pad(img, ((0, 0), (0, 0), (0, rows - h), (0, cols - w)), 'symmetric')
    # generate imglist (i,j) denote the position
    x = np.arange(0, rows, step=stride)
    y = np.arange(0, cols, step=stride)
    pos = []
    for i in x:
        for j in y:
            pos.append([i, j])

    return img, (h, w), pos


def get_tif_meta(tif_path):
    dataset = gdal.Open(tif_path, 0)
    # column
    width = dataset.RasterXSize
    # rows
    height = dataset.RasterYSize
    # affine matrix
    geotrans = dataset.GetGeoTransform()
    # projection info
    proj = dataset.GetProjection()
    dataset =None
    return width, height, geotrans, proj


# old version, not use grid, but on the whole images
class wholeimgLoader(data.Dataset):
    def __init__(self,rootname=r'D:\data\Landcover\s12range',
                 cityname='beijing', datastats='datastats',
                normmethod='minmax', datarange=(0, 1),
                grid=1024, stride=1000):
        super().__init__()
        self.s2path = os.path.join(rootname, cityname + '_' + 's2_rgbnir' + '_clip.tif')
        self.s1path = os.path.join(rootname, cityname + '_' + 's1_vvvhratio' + '_clip.tif')
        # convert VRT to tif images
        if not os.path.exists(self.s2path):
            gdal.Translate(self.s2path, self.s2path[:-4] + '.vrt')
        if not os.path.exists(self.s1path):
            gdal.Translate(self.s1path, self.s1path[:-4] + '.vrt')

        # get indices
        self.width, self.height, self.geotrans, self.proj = get_tif_meta(self.s2path)
        width1, height1, _, _ = get_tif_meta(self.s1path)
        if (self.width!=width1):
            raise ValueError('width mismatch in s1 & s2')
        if (self.height!=height1):
            raise  ValueError('height mismatch in s1 & s2')

        x = np.arange(0, self.width, stride) # row
        y = np.arange(0, self.height, stride) # col
        pos = []
        for i in x:
            for j in y:
                pos.append([i, j])
        self.pos = pos

        # define normalization
        self.norms1 = np.loadtxt(os.path.join(datastats, f's1_vvvhratio_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f's2_rgbnir_{normmethod}.txt'))
        if normmethod == 'minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0]  # max-min

        self.grid = grid
        self.datarange = datarange

    def __getitem__(self, index):
        xoff, yoff = self.pos[index]
        xoff = int(xoff)
        yoff = int(yoff)
        # s2 & s1 images:  read sentinel-1/2, and normalization in numpy format
        xcount = min(self.grid, self.width-xoff)
        xcount = int(xcount)
        ycount = min(self.grid, self.height-yoff)
        ycount = int(ycount)

        s2img = gdal.Open(self.s2path, 0)
        s2 = s2img.ReadAsArray(xoff, yoff, xcount, ycount) # xoff, yoff, xcount, ycount
        s2 = s2.transpose((1, 2, 0))
        rows, cols, bs2 = s2.shape
        s2img = None

        s1img = gdal.Open(self.s1path, 0)
        s1 = s1img.ReadAsArray(xoff, yoff, xcount, ycount)
        s1 = s1.transpose((1, 2, 0))
        s1img =None

        img = np.concatenate((s2, s1), axis=-1)
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1]  # sentinel-2
        img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1]  # sentinel-1
        if isinstance(self.datarange, tuple):
            img[img < self.datarange[0]] = self.datarange[0]
            img[img > self.datarange[1]] = self.datarange[1]
        img = np.transpose(img, axes=(2, 0, 1)) # B H W

        # pad the img
        img = np.pad(img, pad_width=((0,0), (0, self.grid-rows), (0, self.grid-cols)),
                     mode='symmetric')
        pos = np.array([xoff, yoff, xcount, ycount])
        return img, pos

    def __len__(self):
        return len(self.pos)


def generateindex(shpfile, transform, validname=None):
    data = gpd.GeoDataFrame.from_file(shpfile)
    if validname is not None:
        data = data[data[validname]>0].copy()
    datadict = data.to_dict(orient='dict')
    pos = []

    # geolocation
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    for k, v in datadict['geometry'].items():
        minX, minY, maxX, maxY = v.bounds
        xoff = round((minX - xOrigin) / pixelWidth)
        yoff = round((yOrigin - maxY) / pixelHeight)
        # yoff = math.fabs(yoff)
        xcount = round((maxX - minX) / pixelWidth)
        ycount = round((maxY - minY) / pixelHeight)
        pos.append((xoff, yoff, xcount, ycount))
    return pos


# 2023.12.1: add grids, and only predict building height on grids
class gridimgLoader(data.Dataset):
    def __init__(self, rootname=r'D:\data\buildheight\s2chn',
                 cityname='Beijing', datastats='datastats',
                normmethod='minmax', datarange=(0, 1),
                 s1dir='s1', s2dir='s2', gridvalid=None,
                 nchans=6):
        super().__init__()
        self.nchans = nchans
        self.s2path = os.path.join(rootname, cityname + '_' + 's2' + '.tif')
        self.s1path = os.path.join(rootname, cityname + '_' + 's1' + '.tif')
        self.gridpath = os.path.join(rootname, cityname + '_' + 's2_grid' + '.shp')
        # get indices
        self.width, self.height, self.geotrans, self.proj = get_tif_meta(self.s2path)
        width1, height1, _, _ = get_tif_meta(self.s1path)
        if (self.width!=width1):
            raise ValueError('width mismatch in s1 & s2')
        if (self.height!=height1):
            raise ValueError('height mismatch in s1 & s2')

        # get the indices of each valid grid
        self.pos = generateindex(self.gridpath, self.geotrans, gridvalid)

        # define normalization
        self.norms1 = np.loadtxt(os.path.join(datastats, f'{s1dir}_{normmethod}.txt'))
        self.norms2 = np.loadtxt(os.path.join(datastats, f'{s2dir}_{normmethod}.txt'))
        self.norms2 = self.norms2[:, :self.nchans]
        if normmethod == 'minmax':
            self.norms1[1] -= self.norms1[0]  # max-min
            self.norms2[1] -= self.norms2[0]  # max-min

        self.datarange = datarange

    def __getitem__(self, index):
        xoff, yoff, xcount, ycount = self.pos[index]

        s2img = gdal.Open(self.s2path, 0)
        s2 = s2img.ReadAsArray(xoff, yoff, xcount, ycount) # xoff, yoff, xcount, ycount
        s2 = s2.transpose((1, 2, 0))
        s2 = s2[:, :, :self.nchans] # H W C
        rows, cols, bs2 = s2.shape
        s2img = None

        s1img = gdal.Open(self.s1path, 0)
        s1 = s1img.ReadAsArray(xoff, yoff, xcount, ycount)
        s1 = s1.transpose((1, 2, 0))
        s1img =None

        img = np.concatenate((s2, s1), axis=-1)
        img = torch.from_numpy(img.astype('float32')).float()  # .permute(2,0,1)
        img[:, :, :bs2] = (img[:, :, :bs2] - self.norms2[0]) / self.norms2[1]  # sentinel-2
        img[:, :, bs2:] = (img[:, :, bs2:] - self.norms1[0]) / self.norms1[1]  # sentinel-1
        # if isinstance(self.datarange, tuple):
        #     img[img < self.datarange[0]] = self.datarange[0]
        #     img[img > self.datarange[1]] = self.datarange[1]

        img = np.transpose(img, axes=(2, 0, 1)) # B H W
        pos = np.array([xoff, yoff, xcount, ycount])
        return img, pos

    def __len__(self):
        return len(self.pos)


if __name__=="__main__":
    data_path = r'D:\data\buildheight\samples'
    city = 'globe'
    # city = 'china'
    suffix = '_check'
    trainlist = os.path.join(data_path, f'datalist_{city}_train_0.7.csv')
    datastats = 'datastatsglobe'
    preweight = os.path.join(datastats, f'bh_stats_{city}.txt')
    # dataloader = myImageFloder_S12_globe(trainlist, data_path, datastats='datastatsglobe',
    #                                          normmethod='minmax', datarange=(0, 1),
    #                                          aug=True, num_sample=0,
    #                                          preweight=preweight,
    #                                          s1dir=f's1{city}' + suffix, s2dir=f's2{city}' + suffix,
    #                                          heightdir=f'bh{city}',
    #                                          # demdir=f'dem{city}',
    #                                         nchans=4,
    #                                          isaggre=True, ishir=True, hir=(0, 3, 12, 21, 30, 60, 90, 255),
    #                                      )
    # lr, hr, build, weight = dataloader.__getitem__(0)
    # lr, hr, height, build = dataloader.__getitem__(0)

    # 2023.12.19: test the LR and HR loader
    '''
    myImageFloderLRHRglobe(trainlist, data_path, datastats=datastats,
                       aug=True, num_sample=0, datarange=(0, 1),
                  s2dir=f's2{city}' + suffix, gedir=f'ge
                  {city}' + suffix,
                  normmethod=args.normmethod)
    '''

    # test on the whole image
    '''
    rootname = r'D:\data\Landcover\s12range'
    cityname = 'shenzhen'
    # s2path = os.path.join(rootname, cityname + '_' + 's2_rgbnir' + '_clip.tif')
    # gdal.Translate(s2path, s2path[:-4]+'.vrt')
    s1path = os.path.join(rootname, cityname + '_' + 's1_vvvhratio' + '_clip.tif')
    gdal.Translate(s1path, s1path[:-4]+'.vrt')
    img, orisize = load_s12(rootname=rootname,
             cityname=cityname, datastats='datastats',
             normmethod='minmax', datarange=(0, 1),
             grid=256, stride=24)
    
    '''
    # generate imglist (i,j) denote the position
    '''
    rows = 5000
    cols = 5000
    stride = 256-24
    x = np.arange(0, rows, step=stride)
    y = np.arange(0, cols, step=stride)
    numx = len(x)
    numy = len(y)
    t0 = time()
    index = np.indices((numx, numy))
    pos = np.moveaxis(index, 0, -1)
    pos = pos*stride
    t1 = time()
    print('%.6f'%(t1-t0))
    # print(pos)

    t0 = time()
    tmp = []
    for i in x:
        for j in y:
            tmp.append([i, j])
    t1 = time()
    print('%.6f'%(t1-t0))
    # print(tmp)
    '''
    '''
    ipath = r'D:\data\Landcover\samples62\s2_rgbnir'
    iname = 'baoding_4.tif'
    tifpath = os.path.join(ipath, iname)
    raster = gdal.Open(tifpath, 0)
    data = raster.ReadAsArray()
    data = data.transpose((1, 2, 0))
    print(data.max())
    data2 = tif.imread(tifpath)
    print(data2.max())
    diff = (data-data2)
    print(diff.max())
    '''
    '''
    dataset = wholeimgLoader(rootname=r'D:\data\Landcover\s12range',
                 cityname='shenzhen', datastats='datastats',
                normmethod='minmax', datarange=(0, 1),
                grid=64, stride=64-6)
    print(len(dataset))
    # batch = dataset.__getitem__(0)
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                                 pin_memory=True)

    for i in dataloader:
        print('success')
    '''
    # 2023.12.1: test on each image with the striction of grid
    # city='globe'
    # suffix='_check'
    # dataset = gridimgLoader(rootname=r'D:\data\buildheight\s2chn',
    #              cityname='Nanchang', datastats='datastatsglobe',
    #              normmethod='minmax', datarange=(0, 1),
    #              s1dir=f's1{city}'+suffix, s2dir=f's2{city}'+suffix,
    #              gridvalid='isv', nchans=6)
    # img, pos = dataset.__getitem__(0)

    # 2024.3.12: only consider the s1 images
    # dataloader = myImageFloder_S1_globe(trainlist, data_path, datastats='datastatsglobe',
    #                                          normmethod='minmax', datarange=(0, 1),
    #                                          aug=True, num_sample=0,
    #                                          preweight=preweight,
    #                                          s1dir=f's1{city}' + suffix, s2dir=f's2{city}' + suffix,
    #                                          heightdir=f'bh{city}',
    #                                          # demdir=f'dem{city}',
    #                                         nchans=4,
    #                                         isaggre=True, ishir=True, hir=(0, 3, 12, 21, 30, 60, 90, 255),
    #                                      )
    # lr, hr, build, weight = dataloader.__getitem__(0)

    # 2024.3.12: consider the weight to simple frequency-inverse, or equal weight
    hier = (0, 3, 12, 21, 30, 60, 90, 255)
    num_hier = len(hier) - 1
    heightweight = np.ones((num_hier,))
    stats = np.loadtxt(preweight)
    w = hierweight(stats, hier)
    print(w)
    # sqrt
    # [0.08743518 0.26821995 0.32067124 0.73515255 0.98135007 1.60267172
    #  3.0044993 ]

    w = hierweight_simple(stats, hier)
    print(w)
    # [4.02924542e-03 3.79169577e-02 5.41965148e-02 2.84843482e-01
    #  5.07573877e-01 1.35375631e+00 4.75768362e+00]
    print(w.sum())

    w = hierweight_equal(stats, hier)
    print(w)
    # [1. 1. 1. 1. 1. 1. 1.]