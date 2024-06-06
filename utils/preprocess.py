import os
import tifffile as tif
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from osgeo import gdal
import rasterio as rio
from matplotlib import cm

def preprocess_imglab(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]
    labpath = testlist.iloc[idx, 1]

    img = tif.imread(img_path)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
    img_tensor = img_tensor/255.0
    img_tensor = img_tensor.unsqueeze(0) # N C H W
    # lab
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W

    return img_tensor, lab_tensor, img


def preprocess_s12lab(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]
    labpath = testlist.iloc[idx, 1]
    # path
    rootname = os.path.dirname(os.path.dirname(img_path))
    basename = os.path.basename(img_path)
    # s2 & s1 images
    s2 = tif.imread(img_path)
    s1 = tif.imread(os.path.join(rootname, 'sen1', basename))
    img = np.concatenate((s2, s1), axis=-1)
    # img to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
    img_tensor = img_tensor/255.0
    img_tensor = img_tensor.unsqueeze(0) # N C H W
    # lab
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W

    return img_tensor, lab_tensor, img


def preprocess_tlclab(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]
    labpath = testlist.iloc[idx, 1]

    img = tif.imread(img_path)
    # tlc
    img_name = os.path.basename(img_path)
    img_dir = os.path.dirname(os.path.dirname(img_path))
    tlc_path = os.path.join(img_dir, 'tlc', 'tlc' + img_name[3:])
    tlc = tif.imread(tlc_path)
    # concat
    img = np.concatenate((img, tlc), axis=2)  # N H (C1+C2)

    img_norm = np.float32(img)/255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # H W C ==> C H W
    img_tensor = img_tensor.unsqueeze(0) # N C H W

    # lab
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W
    #scale
    if scale!=1:
        h,w = img_tensor.shape[2:]
        img_tensor = F.interpolate(img_tensor, size=(int(h*scale), int(w*scale)), mode='bilinear', align_corners=True)
        lab_tensor = F.interpolate(lab_tensor.unsqueeze(0), size=(int(h * scale), int(w * scale)), mode='nearest')
        lab_tensor = lab_tensor[0]
    return img_tensor, lab_tensor, img_norm[:,:,:3]


def preprocess_t1t2(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]

    ibase = os.path.basename(img_path)[:-4]
    idir = os.path.dirname(os.path.dirname(img_path))

    img1 = tif.imread(img_path)
    img2 = tif.imread(os.path.join(idir, 'img2', ibase + '.tif'))
    # tlc: h w 3
    tlc1 = tif.imread(os.path.join(idir, 'tlc1', ibase + '.tif'))
    tlc2 = tif.imread(os.path.join(idir, 'tlc2', ibase + '.tif'))
    # concate
    img = np.concatenate([img1, tlc1, img2, tlc2], axis=2)
    img_norm = np.float32(img)/255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # H W C ==> C H W
    img_tensor = img_tensor.unsqueeze(0) # N C H W

    # mask
    labpath = os.path.join(idir, 'lab', 'lab' + ibase[3:] + '.png')
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W
    #scale
    if scale!=1:
        h,w = img_tensor.shape[2:]
        img_tensor = F.interpolate(img_tensor, size=(int(h*scale), int(w*scale)), mode='bilinear', align_corners=True)
        lab_tensor = F.interpolate(lab_tensor.unsqueeze(0), size=(int(h * scale), int(w * scale)), mode='nearest')
        lab_tensor = lab_tensor[0]
    return img_tensor, lab_tensor, img_norm[:,:,:3]


def array2raster(res_tif, array, src_tif,
                 datatype=gdal.GDT_Byte,
                 nresolution=2.5,
                 compressoption = ['COMPRESS=PACKBITS'],
                 ):
    src = gdal.Open(src_tif,0)
    geotrans = src.GetGeoTransform()
    proj = src.GetProjection()
    src = None
    geotrans = list(geotrans)
    geotrans[1] = nresolution
    geotrans[5] = -nresolution
    rows, cols = array.shape[:2]
    if len(array.shape)==2:
        array = np.expand_dims(array, axis=2)
    nbands= array.shape[2]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(res_tif, cols, rows, nbands, datatype,
                              options=compressoption) # PACKBITS
    outRaster.SetGeoTransform(geotrans)
    outRaster.SetProjection(proj)
    for i in range(nbands):
        outband = outRaster.GetRasterBand(i+1)
        outband.WriteArray(array[:, :, i])
        # outband.SetNoDataValue(nodata) # set nodata
    outRaster.FlushCache()  ##saves to disk!!
    outRaster = None
    outband = None


# write grid by grid, instead of the whole images
def array2raster_grid(res_tif, array, pos, src_tif,
                 datatype=gdal.GDT_Byte,
                 nresolution=2.5,
                 compressoption = ['COMPRESS=PACKBITS'],
                 ):
    src = gdal.Open(src_tif,0)
    geotrans = src.GetGeoTransform()
    proj = src.GetProjection()
    src = None
    geotrans = list(geotrans)
    geotrans[1] = nresolution
    geotrans[5] = -nresolution
    rows, cols = array.shape[:2]
    if len(array.shape)==2:
        array = np.expand_dims(array, axis=2)
    nbands= array.shape[2]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(res_tif, cols, rows, nbands, datatype,
                              options=compressoption) # PACKBITS
    outRaster.SetGeoTransform(geotrans)
    outRaster.SetProjection(proj)
    for i in range(nbands):
        outband = outRaster.GetRasterBand(i+1)
        outband.WriteArray(array[:, :, i])
        # outband.SetNoDataValue(nodata) # set nodata
    outRaster.FlushCache()  ##saves to disk!!
    outRaster = None
    outband = None

# jet
CMAP = {
    0: (0, 0, 0, 255),
    1: (0, 40.5, 255, 255), #blue
    2: (0., 212.5, 255, 255), #cyan
    3: (125, 255, 121.77419355, 255), #green
    4: (255, 229.81481481, 0, 255), # yellow
    5: (255, 70.55555556, 0, 255), # orange
    6: (127.5, 0, 0, 255), #red
}

def array2raster_rio(res_tif, array, src_tif, bands=1,
                     nresolution=2.5, iscmap=True, compress=None):
    with rio.open(src_tif, 'r') as src:
        profile = src.profile
        profile.update(dtype=array.dtype, count=bands,
                       height=array.shape[0], width=array.shape[1])
        trans = src.transform
        newtrans = rio.Affine(nresolution, trans[1], trans[2],
                              trans[3], -nresolution, trans[5],
                              trans[6], trans[7], trans[8])
        profile.update(transform=newtrans)
        if compress is not None:
            profile.update(compress=compress)
        if 'nodata' in profile:
            profile.pop('nodata')
        with rio.open(res_tif, 'w', **profile) as tgt:
            tgt.write(array, 1)
            if iscmap:
                tgt.write_colormap(1, CMAP)


if __name__=="__main__":
    res_tif='../tmp/test_rio.tif'
    array = np.random.randint(0,5,size=(256,256))
    src_tif = r'D:\data\Landcover\samples62\s2_rgbnir\beijing_47.tif'
    array2raster_rio(res_tif, array, src_tif)