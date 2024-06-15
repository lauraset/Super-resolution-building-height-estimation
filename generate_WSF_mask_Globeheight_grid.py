'''
clip WSF to restrict the spatial range of prediction
'''
from osgeo import gdal, osr, ogr
import os
import geopandas as gpd
import unicodedata
from shapely.geometry import mapping
from pathlib  import Path
import math
from demo_preprocess_height_v2 import Count_fishgrid_valid, Fishgrid_stats
import fiona
from shapely.geometry import mapping
import pandas as pd
import rasterio as rio
import numpy as np


def get_tif_meta(tif_path):
    dataset = gdal.Open(tif_path)
    # 栅格矩阵的列数
    width = dataset.RasterXSize
    # 栅格矩阵的行数
    height = dataset.RasterYSize
    # 获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    # 获取投影信息
    proj = dataset.GetProjection()
    # close dataset
    dataset = None
    return width, height, geotrans, proj

def shp2tif(shp_path, refer_tif_path, target_tif_path, attribute_field, nodata_value=0):
    width, height, geotrans, proj = get_tif_meta(refer_tif_path)
    # 读取shp文件
    shp_file = ogr.Open(shp_path)
    # 获取图层文件对象
    shp_layer = shp_file.GetLayer()
    # 创建栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(
        utf8_path=target_tif_path,  # 栅格地址
        xsize=width,  # 栅格宽
        ysize=height,  # 栅格高
        bands=1,  # 栅格波段数
        eType=gdal.GDT_Byte , # 栅格数据类型
        options=['COMPRESS=LZW'],
    )
    # 将参考栅格的仿射变换信息设置为结果栅格仿射变换信息
    target_ds.SetGeoTransform(geotrans)
    # 设置投影坐标信息
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # 设置背景nodata数值
    band.SetNoDataValue(nodata_value)
    band.FlushCache()

    # 栅格化函数
    gdal.RasterizeLayer(
        dataset=target_ds,  # 输出的栅格数据集
        bands=[1],  # 输出波段
        layer=shp_layer,  # 输入待转换的矢量图层
        options=[f"ATTRIBUTE={attribute_field}"],  # 指定字段值为栅格值
        # creationOptions=['COMPRESS=LZW'],# burn=255, # set valid to 1
    )
    target_ds = None  # todo 释放内存，只有强制为None才可以释放干净
    del target_ds



def clip_tiff_to_extent(ref_tif, dst_tif, extent, xres, yres):
    minX, minY, maxX, maxY = extent
    option = gdal.WarpOptions(outputBounds=[minX, minY, maxX, maxY],
                              format='GTiff',
                              creationOptions=['COMPRESS=LZW'],
                              xRes=xres, yRes=yres)
    gdal.Warp(dst_tif, ref_tif, options=option)

def clip_tiff_by_tiff(ref_tif, dst_tif, query_tif, nres=None):
    data = gdal.Open(query_tif, 0)
    geoTransform = data.GetGeoTransform()
    minX = geoTransform[0]
    maxY = geoTransform[3]
    maxX = minX + geoTransform[1] * data.RasterXSize
    minY = maxY + geoTransform[5] * data.RasterYSize
    if nres is None:
        xres = geoTransform[1]
        yres = -geoTransform[5]
    else:
        xres = yres = nres
    geoproj = data.GetProjection()
    srs = osr.SpatialReference(wkt=geoproj)
    data = None
    option = gdal.WarpOptions(outputBounds=[minX, minY, maxX, maxY],
                              format='GTiff',
                              creationOptions=['COMPRESS=LZW'],
                              xRes=xres, yRes=yres,
                              dstSRS=srs)
    gdal.Warp(dst_tif, ref_tif, options=option)


def main_gen_mask(isoname = 'CHN', fieldname='iso_a3', namevalue='CHN',
                fieldpop='ghspop_2_1', popvalue='large metropolitan areas'):
    shppath = r'D:\data\OSMbuild22217038\urban_centers.shp'

    # isoname = 'USA'
    shpdata = gpd.GeoDataFrame.from_file(shppath)
    items = shpdata[(shpdata[fieldname]==namevalue) & (shpdata[fieldpop]==popvalue)]
    isoname = isoname.lower()

    respath = r'D:\data\buildheight\mask' + isoname
    os.makedirs(respath, exist_ok=True)

    reftifpath = r'D:\data\buildheight\s2'+isoname

    num = items.shape[0]
    for index in range(num):
        tmpdata = items.iloc[[index]].copy()

        tmpname = tmpdata['name_main'].iloc[0]
        tmpname = unicodedata.normalize('NFKD', tmpname).encode('ascii', 'ignore')
        tmpname = str(tmpname, 'utf8')
        tmpname = tmpname.replace(' ', '_')
        tmpname = tmpname.replace('[', '_').replace(']', '_')
        tmpname = tmpname.replace('\'', '')

        # save to a new position
        tmpshp = os.path.join(respath, tmpname+'.shp')
        tmpdata['isv'] = 1
        tmpdata.to_file(tmpshp)

        # convert to tif
        tmpref = os.path.join(reftifpath, tmpname+'_s1.tif')
        restif = os.path.join(respath, tmpname+'_mask.tif')
        shp2tif(tmpshp, tmpref, restif, attribute_field='isv', nodata_value=0)


def main_clip_wsf(wsfpath=r'D:\data\boundary\WSF\WSF2019_cog.tif',
                  resdir = 'wsfchn',
                  resuffix = 'wsf',
                  isoname ='CHN', nres=None):
    # clip WSF
    # wsfpath = r'D:\data\boundary\WSF\WSF2019_cog.tif'
    # shp range
    # shppath = r'D:\data\OSMbuild22217038\urban_centers.shp'
    # resolution
    # s2path = r'D:\data\buildheight\s2chn\Zhuhai_s1.tif'
    # s2 = gdal.Open(s2path)
    # _, xres, _, _, _, yres  = s2.GetGeoTransform()
    # yres = -yres
    s2 = None

    # filter name
    # shpdata = gpd.GeoDataFrame.from_file(shppath)

    # isoname = 'USA'
    #items = shpdata[(shpdata['iso_a3']==isoname) & (shpdata['ghspop_2_1']=='large metropolitan areas')]
    # items = shpdata[(shpdata['iso_a3']==isoname) & (shpdata['ghspop_2_1']=='metropolitan areas')]

    respath = os.path.join(r'D:\data\buildheight', resdir)
    os.makedirs(respath, exist_ok=True)

    searchpath = r'D:\data\buildheight\s2' + isoname.lower()
    querylist = [str(i) for i in Path(searchpath).glob('*_s2.tif')]
    num = len(querylist)
    print(num)

    #num=items.shape[0]
    for index in range(num):
        # tmpdata = items.iloc[index]
        #
        # tmpname = tmpdata['name_main']
        # tmpname = unicodedata.normalize('NFKD', tmpname).encode('ascii', 'ignore')
        # tmpname = str(tmpname, 'utf8')
        # tmpname = tmpname.replace(' ', '_')
        # tmpname = tmpname.replace('[', '_').replace(']', '_')

        queryfile = querylist[index]
        tmpname = os.path.basename(queryfile)[:-6]+f'{resuffix}.tif'

        resname = os.path.join(respath, tmpname)
        if os.path.exists(resname):
            #print('exist: %s, skip' % tmpname)
            continue
        print('process: %s' % tmpname)
        # extent = tmpdata.geometry.bounds
        # clip_tiff_to_extent(wsfpath, resname, extent, xres, yres)
        clip_tiff_by_tiff(wsfpath, resname, queryfile, nres=nres)


def Fishgridnew(tif_path, window_size=256, offset=256):
    idir = os.path.dirname(tif_path)
    iname = os.path.basename(tif_path)[:-4]
    outfile = os.path.join(idir, iname + '_grid.shp')

    width, height, geotrans, geoproj = get_tif_meta(tif_path)
    xres = geotrans[1]
    yres = geotrans[5]
    xmin = geotrans[0]  # top left x
    ymax = geotrans[3]  # top left y
    xmax = xmin + xres * width
    ymin = ymax + yres * height

    grid_sizex = (window_size * xres)
    grid_sizey = -(window_size * yres)
    offsetx = offset * xres
    offsety = -offset * yres
    #参数转换到浮点型
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridwidth = float(grid_sizex)
    gridheight = float(grid_sizey)

    #计算行数和列数
    rows = math.floor((ymax-ymin)/offsety)
    cols = math.floor((xmax-xmin)/offsetx)

    #初始化起始格网四角范围
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin+gridwidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridheight

    #创建输出文件
    outdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outfile):
        outdriver.DeleteDataSource(outfile)
    outds = outdriver.CreateDataSource(outfile)
    # create the spatial reference system, WGS84
    srs = osr.SpatialReference(wkt=geoproj)
    # srs.ImportFromEPSG(4326)
    outlayer = outds.CreateLayer(outfile, srs, geom_type = ogr.wkbPolygon)

    #不添加属性信息，获取图层属性
    outfielddefn  = outlayer.GetLayerDefn()
    #遍历列，每一列写入格网
    col = 0
    while col<cols:
        #初始化，每一列写入完成都把上下范围初始化
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        #遍历行，对这一列每一行格子创建和写入
        row = 0
        while row<rows:
            #创建左上角第一个格子
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYbottom)
            ring.AddPoint(ringXleftOrigin,ringYbottom)
            ring.CloseRings()
            #写入几何多边形
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #创建要素，写入多边形
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            #写入图层
            outlayer.CreateFeature(outfeat)
            outfeat = None
            #下一多边形，更新上下范围， 向下移动
            row+=1
            ringYtop = ringYtop - offsety
            ringYbottom = max(ymin, ringYbottom-offsety) # boundary
        #一列写入完成后，下一列，更新左右范围, 向右移动
        col+=1
        ringXleftOrigin = ringXleftOrigin+offsetx
        ringXrightOrigin = min(xmax, ringXrightOrigin+offsetx) # boundary
    #写入后清除缓存
    outds = None
    outlayer = None

# add boundary at the end
def Fishgridnew_bound(tif_path, window_size=256, offset=256):
    idir = os.path.dirname(tif_path)
    iname = os.path.basename(tif_path)[:-4]
    outfile = os.path.join(idir, iname + '_grid.shp')

    width, height, geotrans, geoproj = get_tif_meta(tif_path)
    xres = geotrans[1]
    yres = geotrans[5]
    x0 = geotrans[0]  # top left x
    y0 = geotrans[3]  # top left y
    x1 = x0 + xres * width
    y1 = y0 + yres * height

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    grid_sizex = math.fabs(window_size * xres)
    grid_sizey = math.fabs(window_size * yres)
    offsetx = math.fabs(offset * xres)
    offsety = math.fabs(offset * yres)
    # #参数转换到浮点型
    # xmin = float(xmin)
    # xmax = float(xmax)
    # ymin = float(ymin)
    # ymax = float(ymax)

    #计算行数和列数
    # rows = math.floor((ymax-ymin)/offsety)
    # cols = math.floor((xmax-xmin)/offsetx)
    rows = math.floor((height-window_size)/offset)+1
    cols = math.floor((width-window_size)/offset)+1
    # calculate the residual location
    diff_row = height - ((rows-1)*offset+window_size)
    diff_col = width - ((cols-1)*offset+window_size)

    #初始化起始格网四角范围
    ringXleftOrigin = xmin+0.0
    ringXrightOrigin = xmin+grid_sizex
    ringYtopOrigin = ymax+0.0
    ringYbottomOrigin = ymax-grid_sizey

    #创建输出文件
    outdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outfile):
        outdriver.DeleteDataSource(outfile)
    outds = outdriver.CreateDataSource(outfile)
    # create the spatial reference system, WGS84
    srs = osr.SpatialReference(wkt=geoproj)
    # srs.ImportFromEPSG(4326)
    outlayer = outds.CreateLayer(outfile, srs, geom_type = ogr.wkbPolygon)

    #不添加属性信息，获取图层属性
    outfielddefn  = outlayer.GetLayerDefn()
    #遍历列，每一列写入格网
    col = 0
    while col<cols:
        #初始化，每一列写入完成都把上下范围初始化
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        #遍历行，对这一列每一行格子创建和写入
        row = 0
        while row<rows:
            #创建左上角第一个格子
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYbottom)
            ring.AddPoint(ringXleftOrigin,ringYbottom)
            ring.CloseRings()
            #写入几何多边形
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #创建要素，写入多边形
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            #写入图层
            outlayer.CreateFeature(outfeat)
            outfeat = None
            #下一多边形，更新上下范围， 向下移动
            row+=1
            ringYtop = ringYtop - offsety
            ringYbottom = max(ymin, ringYbottom-offsety) # boundary
        #一列写入完成后，下一列，更新左右范围, 向右移动
        col+=1
        ringXleftOrigin = ringXleftOrigin+offsetx
        ringXrightOrigin = min(xmax, ringXrightOrigin+offsetx) # boundary

    # generate the last col: top-right point
    if diff_col>0:
        ringYtop = ymax+0.0
        ringYbottom = ymax-grid_sizey
        ringXleftOrigin = xmax-grid_sizex
        ringXrightOrigin = xmax+0.0
        #遍历行，对这一列每一行格子创建和写入
        row = 0
        while row<rows:
            #创建左上角第一个格子
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYbottom)
            ring.AddPoint(ringXleftOrigin,ringYbottom)
            ring.CloseRings()
            #写入几何多边形
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #创建要素，写入多边形
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            #写入图层
            outlayer.CreateFeature(outfeat)
            outfeat = None
            #下一多边形，更新上下范围， 向下移动
            row+=1
            ringYtop = ringYtop - offsety
            ringYbottom = max(ymin, ringYbottom-offsety) # boundary

    # generate the last row: down-left
    if diff_row>0:
        ringYtop = ymin+grid_sizey
        ringYbottom = ymin+0.0
        ringXleftOrigin = xmin+0.0
        ringXrightOrigin = xmin+grid_sizex
        #遍历行，对这一列每一行格子创建和写入
        col = 0
        while col<cols:
            #创建左上角第一个格子
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYbottom)
            ring.AddPoint(ringXleftOrigin,ringYbottom)
            ring.CloseRings()
            #写入几何多边形
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #创建要素，写入多边形
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            #写入图层
            outlayer.CreateFeature(outfeat)
            outfeat = None
            #下一多边形，更新上下范围， 向下移动
            col+=1
            ringXleftOrigin = ringXleftOrigin + offsetx
            ringXrightOrigin = min(xmax, ringXrightOrigin+offsetx) # boundary

    # generate the last grid: : down-right point
    if (diff_col>0) or (diff_row>0):
        ringXleftOrigin = xmax-grid_sizex
        ringXrightOrigin = xmax+0.0
        ringYtop = ymin+grid_sizey
        ringYbottom = ymin+0.0
        # 创建左上角第一个格子
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(ringXleftOrigin, ringYtop)
        ring.AddPoint(ringXrightOrigin, ringYtop)
        ring.AddPoint(ringXrightOrigin, ringYbottom)
        ring.AddPoint(ringXleftOrigin, ringYbottom)
        ring.CloseRings()
        # 写入几何多边形
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # 创建要素，写入多边形
        outfeat = ogr.Feature(outfielddefn)
        outfeat.SetGeometry(poly)
        # 写入图层
        outlayer.CreateFeature(outfeat)
        outfeat = None

    #写入后清除缓存
    outds = None
    outlayer = None


# generate grid
def generate_validgrid(dirname='CHN', isoname = 'CHN', namefield= 'iso_a3', typefield='large metropolitan areas'):
    searchpath = r'D:\data\buildheight\s2' + dirname.lower()
    querylist = [str(i) for i in Path(searchpath).glob('*_s2.tif')]
    num = len(querylist)
    print(num)
    wsfpath =  r'D:\data\buildheight\wsf' + dirname.lower()

    urbanpath = r'D:\data\OSMbuild22217038\urban_centers.shp'
    # isoname = 'USA'
    shpdata = gpd.GeoDataFrame.from_file(urbanpath)
    items = shpdata[(shpdata[namefield]==isoname) & (shpdata['ghspop_2_1']==typefield)]

    num = items.shape[0]
    print(num)
    # num=1 # only test baotou, overlap set to 1/4
    # generate grid
    for i in range(num):
        tifpath = querylist[i]
        gridpath = tifpath[:-4]+'_grid.shp'
        if not os.path.exists(gridpath):
            Fishgridnew_bound(tifpath, window_size=64, offset=56) # 56 48

    for index in range(num):
        tmpdata = items.iloc[[index]].copy()

        tmpname = tmpdata['name_main'].iloc[0]
        tmpname = unicodedata.normalize('NFKD', tmpname).encode('ascii', 'ignore')
        tmpname = str(tmpname, 'utf8')
        tmpname = tmpname.replace(' ', '_')
        tmpname = tmpname.replace('[', '_').replace(']', '_')
        tmpname = tmpname.replace('\'','')

        # if 'Baotou' not in tmpname:
        #     continue
        # filter with the range

        query = os.path.join(searchpath, tmpname+'_s2_grid.shp')
        data = gpd.GeoDataFrame.from_file(query)
        datainter = data[data.intersects(tmpdata.iloc[0].geometry)]
        # # intershp = os.path.join(searchpath, tmpname + '_s2_inter.shp')
        datainter.to_file(query)

        # filter with WSF
        print(tmpname)
        query = os.path.join(searchpath, tmpname + '_s2_grid.shp')
        basename = tmpname+'_wsf.tif'
        iwsfpath = os.path.join(wsfpath, basename)
        Fishgrid_stats(iwsfpath, query,
                   fieldname=('sum', 'count', 'isv'),
                   condition=(0, 20, 4096))


def count_fishgrid(shp_file):
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()
    return lyr.GetFeatureCount()

def main_sample_stats(isoname):
    # Stats valid samples
    savename = f'{isoname}_num_grid'
    datapath = r'D:\data\buildheight\s2'+isoname
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    num_valid = []
    name_valid = []
    num2_valid = []
    for file in filelist:
        file = str(file)
        print(file)
        iname = os.path.basename(file)
        iname = iname.split('_')[0]
        num = count_fishgrid(file) #('isv2', 'isv3'))
        num2 = Count_fishgrid_valid(file, fieldname='isv')
        num_valid.append(num)
        name_valid.append(iname)
        num2_valid.append(num2)
        print('number of valid: %d'%num)

    print('total number is: %d'%sum(num_valid))
    data = pd.DataFrame(data={'num': num_valid,
                              'num_valid': num2_valid}, index=name_valid)
    data.to_csv(f'samplestats/{savename}.csv')

# select grid: 1) overlap with urban center; 2) building coverage larger than
# def main_grid_wsf():

def generateindex(shpfile):
    data = gpd.GeoDataFrame.from_file(shpfile)
    datadict = data.to_dict(orient='dict')
    pos = []
    for k, v in datadict['geometry'].items():
        tmp = v.bounds
        pos.append(tmp)
    return pos


def convert_geotrans_rio(imgpath, res_tif):
    with rio.open(imgpath, 'r') as src:
        profile = src.profile
        data = src.read()
        trans = src.transform

    yres = trans[4]
    ymax = trans[5] + yres * src.height
    print(ymax, yres)

    newtrans = rio.Affine(trans[0], trans[1], trans[2],
                          trans[3], -trans[4], ymax,
                          trans[6], trans[7], trans[8])

    # res_tif = imgpath[:-4] + '_tras.tif'

    datanew = np.flip(data, axis=1)
    profile.update(transform=newtrans)
    with rio.open(res_tif, 'w', **profile) as tgt:
        tgt.write(datanew)


def convert_geotrans_gdal(src_tif, res_tif):
    src = gdal.Open(src_tif,0)
    trans = src.GetGeoTransform()
    proj = src.GetProjection()
    array = src.ReadAsArray()
    width, height = src.RasterXSize, src.RasterYSize
    src = None

    yres = trans[5]
    ymin = trans[3]
    ymax = ymin + yres * height
    # print(ymax, yres)

    newtrans = (trans[0], trans[1], trans[2],
                ymax, trans[4], -yres)

    # res_tif = src_tif[:-4] + '_tras.tif'
    array = np.flip(array, axis=1)

    rows, cols, nbands= array.shape
    datatype = array.dtype
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(res_tif, cols, rows, nbands, datatype,
                              options=['COMPRESS=DEFLATE', 'TILED=YES']) # PACKBITS
    outRaster.SetGeoTransform(newtrans)
    outRaster.SetProjection(proj)
    for i in range(nbands):
        outband = outRaster.GetRasterBand(i+1)
        outband.WriteArray(array[:, :, i])
        # outband.SetNoDataValue(nodata) # set nodata
    outRaster.FlushCache()  ##saves to disk!!
    outRaster = None
    outband = None


if __name__=="__main__":
    isoname = 'CHN'
    # isoname = 'USA'
    # dirname = isoname+'_metro'
    # # isoname = 'USA'
    namefield = 'iso_a3'
    typefield = 'metropolitan areas' #'large metropolitan areas'

    # isoname = 'Europe'
    # dirname = isoname + '_metro'
    # namefield = 'continent'
    # typefield = 'metropolitan areas'
    main_clip_wsf(isoname=dirname)
    # main_gen_mask()
    ## 2023.12.8: generate mask for each region,
    main_gen_mask(isoname='CHN', fieldname='iso_a3', namevalue='CHN',
                  fieldpop='ghspop_2_1', popvalue='large metropolitan areas')
    main_gen_mask(isoname='USA', fieldname='iso_a3', namevalue='USA',
                  fieldpop='ghspop_2_1', popvalue='large metropolitan areas')
    main_gen_mask(isoname='Europe', fieldname='continent', namevalue='Europe',
                  fieldpop='ghspop_2_1', popvalue='large metropolitan areas')
    # metro
    main_gen_mask(isoname='CHN_metro', fieldname='iso_a3', namevalue='CHN',
                  fieldpop='ghspop_2_1', popvalue='metropolitan areas')
    main_gen_mask(isoname='USA_metro', fieldname='iso_a3', namevalue='USA',
                  fieldpop='ghspop_2_1', popvalue='metropolitan areas')
    main_gen_mask(isoname='Europe_metro', fieldname='continent', namevalue='Europe',
                  fieldpop='ghspop_2_1', popvalue='metropolitan areas')
    # generate validgrid
    generate_validgrid(dirname, isoname, namefield=namefield, typefield=typefield)
    # main_sample_stats(dirname)


    # 2023.12.19: generate globe height for each region to adjust the value.
    # isonamelist = ['usa', 'europe_metro', 'usa_metro']
    # for isoname in isonamelist:
    #     main_clip_wsf(wsfpath=r'D:\data\Landcover\reference\GHS100\GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0.tif',
    #                   resdir = 'ghs',
    #                   resuffix='ghs',
    #                   isoname =isoname,
    #                   nres=None)