'''
function: convert all building height in vector to raster format at 2.5 m
update: should be consistent with CNBH 10m dataset, 2023.10.6
'''
import cv2
import numpy as np
import os
import shutil
from osgeo import gdal, ogr, osr, gdalconst
import datetime
from xpinyin import Pinyin
from time import time
from pathlib import Path
import math
from glob import glob
import sys
import numpy
import rasterio
from mask_revised import mask
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib

def shp_to_tiff(shp_file, output_tiff, attribute='class', nresolution=2.5):
    """
    :param shp_file:
    :param output_tiff:
    :param attribute: 定义栅格值的矢量属性
    :return:
    """
    start_time = datetime.datetime.now()
    print("start :" + str(start_time))
    # 读取shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_file, 1)
    # 获取图层文件对象
    shp_layer = data_source.GetLayer()
    lon_min, lon_max, lat_min, lat_max = shp_layer.GetExtent()
    s_projection = str(shp_layer.GetSpatialRef())

    # (0,0,:,0,:,0)表示旋转系数
    # 自定义仿射矩阵系数 ， 1表示分辨率大小，决定了栅格像元的大小
    dst_transform = (lon_min, nresolution, 0, lat_max, 0, -nresolution)
    d_lon = int(abs((lon_max - lon_min) / dst_transform[1])) # 除以横向分辨率
    d_lat = int(abs((lat_max - lat_min) / dst_transform[5]))

    # 根据模板tif属性信息创建对应标准的目标栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(output_tiff, d_lon, d_lat, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(dst_transform)
    target_ds.SetProjection(s_projection)

    band = target_ds.GetRasterBand(1)
    # 设置背景数值
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    # 调用栅格化函数。gdal.RasterizeLayer函数有四个参数，分别有栅格对象，波段，矢量对象，value的属性值将为栅格值
    option = ['ATTRIBUTE=%s' % (attribute)]
    gdal.RasterizeLayer(target_ds, [1], shp_layer, options=option)
    # 直接写入
    y_buffer = band.ReadAsArray()
    target_ds.WriteRaster(0, 0, d_lon, d_lat, y_buffer.tobytes())
    start_time = datetime.datetime.now()
    print("end :" + str(start_time))
    target_ds = None  # todo 释放内存，只有强制为None才可以释放干净
    del target_ds, shp_layer


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


def shp2tif(shp_path, refer_tif_path, target_tif_path, attribute_field="class", nodata_value=0):
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
        eType=gdal.GDT_Byte  # 栅格数据类型
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
        options=[f"ATTRIBUTE={attribute_field}"]  # 指定字段值为栅格值
    )
    target_ds = None  # todo 释放内存，只有强制为None才可以释放干净
    del target_ds


def addField_byExpression(shp_file, newFieldName='FloorNum', oldFieldName='elevation'):
    # from osgeo import ogr
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shp_file, 1)  # 1 is read/write
    layer = dataSource.GetLayer()
    defn = layer.GetLayerDefn()
    fieldIndex = defn.GetFieldIndex(newFieldName)
    if fieldIndex < 0:
        # define floating point field named DistFld and 16-character string field named Name:
        fldDef = ogr.FieldDefn(newFieldName, ogr.OFTInteger)
        # fldDef2 = ogr.FieldDefn('Name', ogr.OFTString)
        # fldDef2.SetWidth(16)  # 16 char string width
        layer.CreateField(fldDef)
    fieldIndex2 = defn.GetFieldIndex(newFieldName)
    if fieldIndex2>0:
        print('create success!')

    # field expression
    feature = layer.GetNextFeature()
    indexA = defn.GetFieldIndex(oldFieldName)
    indexB = defn.GetFieldIndex(newFieldName)
    oField = defn.GetFieldDefn(indexB)
    fieldName = oField.GetNameRef()
    while feature is not None:
        valueA = feature.GetFieldAsInteger(indexA) #
        if valueA is None:
            feature.SetFieldNull(indexB)
            continue
        feature.SetField2(fieldName, valueA/3) # floor number
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    # feature.Destroy()
    del layer, dataSource

# fishgrid
def Fishgrid(outfile, xmin, xmax, ymin, ymax, gridwidth, gridheight,
             geoproj):
    #参数转换到浮点型
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridwidth = float(gridwidth)
    gridheight = float(gridheight)

    #计算行数和列数
    rows = math.ceil((ymax-ymin)/gridheight)
    cols = math.ceil((xmax-xmin)/gridwidth)

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
            ringYtop = ringYtop - gridheight
            ringYbottom = max(ymin, ringYbottom-gridheight) # boundary
        #一列写入完成后，下一列，更新左右范围, 向右移动
        col+=1
        ringXleftOrigin = ringXleftOrigin+gridwidth
        ringXrightOrigin = min(xmax, ringXrightOrigin+gridwidth) # boundary
    #写入后清除缓存
    outds = None
    outlayer = None


def Fishgridnew(tif_path, window_size=256):
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

    grid_size = int(window_size * xres)
    #参数转换到浮点型
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridwidth = float(grid_size)
    gridheight = float(grid_size)

    #计算行数和列数
    rows = math.ceil((ymax-ymin)/gridheight)
    cols = math.ceil((xmax-xmin)/gridwidth)

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
            ringYtop = ringYtop - gridheight
            ringYbottom = max(ymin, ringYbottom-gridheight) # boundary
        #一列写入完成后，下一列，更新左右范围, 向右移动
        col+=1
        ringXleftOrigin = ringXleftOrigin+gridwidth
        ringXrightOrigin = min(xmax, ringXrightOrigin+gridwidth) # boundary
    #写入后清除缓存
    outds = None
    outlayer = None


def Raster_extent(filelist, outfile,
                  locName='location', locType=ogr.OFTString,
                  yearName='2020'):
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        shpDriver.DeleteDataSource(outfile)
    outDataSource = shpDriver.CreateDataSource(outfile)
    # Set the spatial reference
    _, _, _, proj = get_tif_meta(filelist[0])
    srs = osr.SpatialReference(wkt=proj)
    basename = os.path.basename(outfile)[:-4]
    outlayer = outDataSource.CreateLayer(basename, srs, geom_type=ogr.wkbPolygon)
    # Create fields
    outlayer.CreateField(ogr.FieldDefn(locName, locType))
    outlayer.CreateField(ogr.FieldDefn(yearName, ogr.OFTInteger64))
    # Loop over all rasters
    for file in filelist:
        width, height, geotrans, proj = get_tif_meta(file)
        filename = os.path.basename(file)
        filename = filename.split('_')
        year = filename[1]
        loc = filename[2]+'_'+filename[3]
        xres = geotrans[1]
        yres = geotrans[5]
        xmin = geotrans[0] # top left x
        ymax = geotrans[3] # top left y
        xmax = xmin + xres*width
        ymin = ymax + yres*height
        # 创建左上角第一个格子
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xmin, ymax) # top left
        ring.AddPoint(xmax, ymax) # top right
        ring.AddPoint(xmax, ymin) # down right
        ring.AddPoint(xmin, ymin) # down left
        ring.CloseRings()
        # 写入几何多边形
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # 创建要素，写入多边形
        outfeat = ogr.Feature(outlayer.GetLayerDefn())
        outfeat.SetGeometry(poly)
        outfeat.SetField(locName, loc)
        outfeat.SetField(yearName, year)
        # Create the feature in the layer (shapefile)
        outlayer.CreateFeature(outfeat)
        # Dereference the featur
        outfeat = None
    # Save and close the data source
    outDataSource = None


# 2023.9.28: need reproject all images to the same projection
def Raster_extent_prj(filelist, outfile,
                  locName='location', locType=ogr.OFTString,
                  yearName='2020',
                target_crs=4326):
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        shpDriver.DeleteDataSource(outfile)
    outDataSource = shpDriver.CreateDataSource(outfile)

    # Set the spatial reference
    # if target_crs==None:
    _, _, _, proj = get_tif_meta(filelist[0])
    targetSR = osr.SpatialReference(wkt=proj)
    # else:
    # create the spatial reference, WGS84
    #     targetSR = osr.SpatialReference()
    #     targetSR.ImportFromEPSG(target_crs)

    basename = os.path.basename(outfile)[:-4]
    outlayer = outDataSource.CreateLayer(basename, targetSR, geom_type=ogr.wkbPolygon)
    # Create fields
    outlayer.CreateField(ogr.FieldDefn(locName, locType))
    outlayer.CreateField(ogr.FieldDefn(yearName, ogr.OFTInteger64))
    # Loop over all rasters
    for file in filelist:
        print(file)
        width, height, geotrans, proj = get_tif_meta(file)
        filename = os.path.basename(file)[:-4]
        filename = filename.split('_')
        year = '2020' # filename[1]
        loc = filename[1] # filename[2]+'_'+filename[3]

        # Reproject vector geometry to same projection as raster
        sourceSR  = osr.SpatialReference(wkt=proj)
        coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

        xres = geotrans[1]
        yres = geotrans[5]
        xmin = geotrans[0] # top left x
        ymax = geotrans[3] # top left y
        xmax = xmin + xres*width
        ymin = ymax + yres*height
        # 创建左上角第一个格子
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xmin, ymax) # top left
        ring.AddPoint(xmax, ymax) # top right
        ring.AddPoint(xmax, ymin) # down right
        ring.AddPoint(xmin, ymin) # down left
        ring.CloseRings()
        # 写入几何多边形
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        poly.Transform(coordTrans)
        # 创建要素，写入多边形
        outfeat = ogr.Feature(outlayer.GetLayerDefn())
        outfeat.SetGeometry(poly)
        outfeat.SetField(locName, loc)
        outfeat.SetField(yearName, year)

        # Create the feature in the layer (shapefile)
        outlayer.CreateFeature(outfeat)
        # Dereference the feature
        outfeat = None
    # Save and close the data source
    outDataSource = None


# def Zone_stats(raster_file, shp_file, fieldName='area', fieldType=ogr.OFTReal):
#     width, height, geotrans, proj = get_tif_meta(raster_file)
#     # Create the output shapefile
#     shpDriver = ogr.GetDriverByName("ESRI Shapefile")
#     dataSource = shpDriver.Open(shp_file, 1)  # 1 is read/write
#     shplayer = dataSource.GetLayer()
#     # Create the field
#     defn = shplayer.GetLayerDefn()
#     if defn.GetFieldIndex(fieldName) == -1:
#         shplayer.CreateField(ogr.FieldDefn(fieldName, fieldType))
#     # Statistics for each raster
#     for feature in shplayer:
#         value =
#         feature.SetField(fieldName, value)
#         shplayer.SetFeature(feature)
#
#     # Close
#     dataSource = None


def zonal_stats(input_zone_polygon, input_value_raster, fieldName=('sum','count'),
                fieldType=ogr.OFTInteger64):
    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon, 1) # 1 read & write
    lyr = shp.GetLayer()

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldName:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, fieldType))

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    sumstats=[]
    countstats=[]
    # feat = lyr.GetNextFeature() # the first feature
    for feat in lyr:
        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)
        # Get extent of feat
        geom = feat.GetGeometryRef()
        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            pointsX = []; pointsY = []
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                        lon, lat, z = ring.GetPoint(p)
                        pointsX.append(lon)
                        pointsY.append(lat)
                count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            pointsX = []; pointsY = []
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)

        else:
            sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)

        # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth) # +1
        ycount = int((ymax - ymin)/pixelWidth) # +1

        # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((
            xmin, pixelWidth, 0,
            ymax, 0, pixelHeight,
        ))
        target_ds.SetProjection(raster.GetProjectionRef())

        # Create memory target vector layer
        shp_name = 'temp'
        mem_driver = ogr.GetDriverByName("Memory")
        if os.path.exists(shp_name):
            mem_driver.DeleteDataSource(shp_name)
        shp_ds = mem_driver.CreateDataSource(shp_name)
        target_lyr = shp_ds.CreateLayer('polygons', targetSR, ogr.wkbPolygon)
        target_lyr.CreateFeature(feat.Clone())

        # Create for target raster the same projection as for the value raster
        # raster_srs = osr.SpatialReference()
        # raster_srs.ImportFromWkt(raster.GetProjectionRef())
        # target_ds.SetProjection(raster_srs.ExportToWkt())

        # Rasterize zone polygon to raster
        gdal.RasterizeLayer(target_ds, [1], target_lyr, burn_values=[1])

        # Read raster as arrays, grid from the original raster
        banddataraster = raster.GetRasterBand(1)
        dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.float64)

        bandmask = target_ds.GetRasterBand(1)
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(numpy.float64)

        # Mask zone of raster
        zoneraster = numpy.ma.masked_array(dataraster,  numpy.logical_not(datamask))
        zoneraster_foot = (zoneraster>0).astype(numpy.uint8)

        # Set field
        feat.SetField2(fieldName[0], zoneraster_foot.sum())
        feat.SetField2(fieldName[1], zoneraster.count())
        lyr.SetFeature(feat)
        # sumstats.append(zoneraster.sum())
        # countstats.append(zoneraster.count())

        # Close
        target_ds = None
        target_lyr = None
        shp_ds = None

    shp = None
    lyr = None
    raster = None
    return True


# def loop_zonal_stats(input_zone_polygon, input_value_raster):
#
#     shp = ogr.Open(input_zone_polygon)
#     lyr = shp.GetLayer()
#     featList = range(lyr.GetFeatureCount())
#     statDict = {}
#
#     for FID in featList:
#         feat = lyr.GetFeature(FID)
#         meanValue = zonal_stats(feat, input_zone_polygon, input_value_raster)
#         statDict[FID] = meanValue
#     return statDict

def merge_alltif(imglist, outfile, srcNodata=0, VRTNodata=0):
    tifs = imglist  # [:10]
    t0 = time()
    iroot = os.path.dirname(outfile)
    iname = os.path.basename(outfile)
    if '.' in iname:
        iname = iname[:-4]+'.vrt'
    else:
        iname  = iname+'.vrt'
    vrt_file = os.path.join(iroot, iname)
    # tif_file = os.path.join(iroot, iname+'.tif')
    gdal.BuildVRT(vrt_file, tifs,
                  options=gdal.BuildVRTOptions(srcNodata=srcNodata,
                                               VRTNodata=VRTNodata)) #options=gdal.BuildVRTOptions()srcNodata=0, VRTNodata=0))
    # ds = gdal.Open(vrt_file)
    # translateoptions = gdal.TranslateOptions(
    #     gdal.ParseCommandLine("-of GTiff -ot Byte -co COMPRESS=LZW -a_nodata 255"))
    # gdal.Translate(tif_file, ds, options=translateoptions)
    print('time elaps: %.2f' % (time() - t0))
    ds = None


def clip_vrt(vrt_file, shp_file, out_file, proj,
             fieldname=('vrt_sum', 'vrt_count'), nresolution=2.5):
    # raster = gdal.Open(vrt_file, gdal.GA_ReadOnly)  # read raster
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1)
    lyr = VectorDataset.GetLayer()

    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    num_feature = lyr.GetFeatureCount()
    for i in range(num_feature):
        feature = lyr.GetNextFeature()
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        # Get extent of feat
        minX, maxX, minY, maxY = geom.GetEnvelope()

        # Clip & warp the original images
        out_file = out_file[:-4]+'.vrt'
        OutTile = gdal.Warp(out_file, vrt_file, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=proj)

        OutTile = None
        out_source = gdal.Open(out_file, 0)
        out_data = out_source.GetRasterBand(1)
        out_data = out_data.ReadAsArray()
        out_data = (out_data==255).astype(numpy.uint8)

        feature.SetField2(fieldname[0], out_data.sum())
        feature.SetField2(fieldname[1], out_data.size)
        lyr.SetFeature(feature)
        out_source = None

    VectorDataset = None
    lyr = None


# compare two tiff images using the same shp_file
def compare_twotiff(tif_file1, tif_file2, shp_file, target_proj,
                    fieldname=('sum', 'count', 'vrt_sum', 'vrt_count', 'absdiff'),
                    nresolution=2.5,
                    condition=(0, 0)):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()

    # Coordinate transformation to the same projection
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=target_proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all

    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Tif 1
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent
        # Clip & warp the original images
        OutTile2 = gdal.Warp("tmp1.vrt", tif_file1, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj)
        out_data1 = OutTile2.GetRasterBand(1)
        out_data1 = out_data1.ReadAsArray()
        out_data1 = (out_data1>condition[0]).astype(numpy.uint8)
        OutTile2 = None

        # Tif 2
        # geom = feature.GetGeometryRef()
        # geom.Transform(coordTrans)
        # minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent
        # Clip & warp the original images
        OutTile = gdal.Warp("tmp2.vrt", tif_file2, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj)
        out_data2 = OutTile.GetRasterBand(1)
        out_data2 = out_data2.ReadAsArray()
        out_data2 = (out_data2>condition[1]).astype(numpy.uint8)
        OutTile = None

        # Absolute difference: check dimension
        if out_data1.shape == out_data2.shape:
            diff = ((out_data1 -out_data2)!=0).astype('uint8')
            diff = diff.sum()
        else:
            diff = 65536 # means delete
        # Create fields: sum & count
        feature.SetField2(fieldname[0], out_data1.sum())
        feature.SetField2(fieldname[1], out_data1.size)
        feature.SetField2(fieldname[2], out_data2.sum())
        feature.SetField2(fieldname[3], out_data2.size)
        feature.SetField2(fieldname[4], diff)
        lyr.SetFeature(feature)
        # out_source = None
        feature = None

    # Close dataset
    VectorDataset = None
    lyr = None


def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou


# compare two tiff images using the same shp_file
def compare_twotiff_valid(tif_ref, vrt_file, shp_file,
                    fieldname=('vrt_sum', 'vrt_count', 'absdiff'),
                    validname=('isv', 'isv2', 'isv3', 'isv4'),
                    nresolution=2.5,
                    condition=(0, 2000, 65536, 0.3)):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()

    # Referenced tif
    raster = gdal.Open(tif_ref)
    target_proj = raster.GetProjectionRef()
    rasterband = raster.GetRasterBand(1)
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Coordinate transformation to the same projection
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=target_proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname+validname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))
    index_isv = defn.GetFieldIndex(validname[0])

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Check if the feature is valid (e.g., meet the condition 1)
        value_isv = feature.GetFieldAsInteger(index_isv)
        if value_isv == 0:
            continue
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent

        # Tif 1: the reference file
        # Specify offset and rows and columns to read
        xoff = int((minX - xOrigin)/pixelWidth)
        yoff = int((yOrigin - maxY)/pixelHeight)
        xcount = int((maxX - minX)/pixelWidth) # +1
        ycount = int((maxY - minY)/pixelHeight) # +1
        out_data1 = rasterband.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint8)
        out_data1 = (out_data1>condition[0]).astype(numpy.uint8)

        #Tif 2: Clip & warp the reference images
        OutTile2 = gdal.Warp("tmp.vrt", vrt_file, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj)
        out_data2 = OutTile2.GetRasterBand(1)
        out_data2 = out_data2.ReadAsArray()
        out_data2 = (out_data2>condition[0]).astype(numpy.uint8)
        OutTile2 = None
        isum = out_data2.sum()
        icount = out_data2.size
        ivalid2 = 1 if (isum >= condition[1]) and (icount >= condition[2]) else 0

        # Absolute difference: check dimension
        if out_data1.shape == out_data2.shape:
            diff = (out_data1!=out_data2).astype('uint8')
            diff = diff.sum()
        else:
            diff = 65536 # means delete

        # Condition 2: difference should be lower than T2 (0.3)
        ivalid3 = ((diff/icount) <= condition[3]).astype(numpy.uint8)
        ivalid4 = 1 if (ivalid2==1) and (ivalid3==1) else 0
        # Create fields: sum & count
        feature.SetField2(fieldname[0], isum)
        feature.SetField2(fieldname[1], icount)
        feature.SetField2(fieldname[2], diff)

        feature.SetField2(validname[1], ivalid2)
        feature.SetField2(validname[2], ivalid3)
        feature.SetField2(validname[3], ivalid4)
        lyr.SetFeature(feature)
        feature = None

    # Close dataset
    raster = None
    VectorDataset = None
    lyr = None

def compare_twotiff_valid_iou(tif_ref, vrt_file, shp_file,
                    fieldname=('vrt_sum', 'vrt_count', 'absdiff'),
                    validname=('isv', 'isv2', 'isv3', 'isv4'),
                    nresolution=2.5,
                    condition=(0, 2000, 65536, 0.3)):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()

    # Referenced tif
    raster = gdal.Open(tif_ref)
    target_proj = raster.GetProjectionRef()
    rasterband = raster.GetRasterBand(1)
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Coordinate transformation to the same projection
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=target_proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname+validname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))
    index_isv = defn.GetFieldIndex(validname[0])
    # create iou
    if defn.GetFieldIndex('diou') == -1:
        lyr.CreateField(ogr.FieldDefn('diou', ogr.OFTReal))

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Check if the feature is valid (e.g., meet the condition 1)
        value_isv = feature.GetFieldAsInteger(index_isv)
        if value_isv == 0:
            continue
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent

        # Tif 1: the reference file
        # Specify offset and rows and columns to read
        xoff = int((minX - xOrigin)/pixelWidth)
        yoff = int((yOrigin - maxY)/pixelHeight)
        xcount = int((maxX - minX)/pixelWidth) # +1
        ycount = int((maxY - minY)/pixelHeight) # +1
        out_data1 = rasterband.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint8)
        out_data1 = (out_data1>condition[0]).astype(numpy.uint8)

        #Tif 2: Clip & warp the reference images
        OutTile2 = gdal.Warp("tmp.vrt", vrt_file, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj)
        out_data2 = OutTile2.GetRasterBand(1)
        out_data2 = out_data2.ReadAsArray()
        out_data2 = (out_data2>condition[0]).astype(numpy.uint8)
        OutTile2 = None
        isum = out_data2.sum()
        icount = out_data2.size
        ivalid2 = 1 if (isum >= condition[1]) and (icount >= condition[2]) else 0

        # Absolute difference: check dimension
        if out_data1.shape == out_data2.shape:
            diou = 1-calculate_iou(out_data1, out_data2)
            diff = (out_data1!=out_data2).astype(numpy.uint8)
            diff = diff.sum()
        else:
            diff = 65536 # means delete
            diou = 1

        # Condition 2: difference should be lower than T2 (0.3)
        # ivalid3 = (diff/icount <= condition[3]).astype(numpy.uint8)
        ivalid3 = (diou <= condition[3]) # and ((diff/icount <= 0.3))
        ivalid3 = ivalid3.astype(numpy.uint8)

        ivalid4 = 1 if (ivalid2==1) and (ivalid3==1) else 0
        # Create fields: sum & count
        feature.SetField2(fieldname[0], isum)
        feature.SetField2(fieldname[1], icount)
        feature.SetField2(fieldname[2], diff)

        feature.SetField2(validname[1], ivalid2)
        feature.SetField2(validname[2], ivalid3)
        feature.SetField2(validname[3], ivalid4)

        feature.SetField2('diou', diou)

        lyr.SetFeature(feature)
        feature = None

    # Close dataset
    raster = None
    VectorDataset = None
    lyr = None


def compare_twotiff_valid_rmse(tif_ref, vrt_file, shp_file,
                    fieldname=('vrt_sum', 'vrt_count', 'absdiff'),
                    validname=('isv', 'isv2', 'isv3', 'isv4'),
                    nresolution=2.5,
                    condition=(0, 2000, 65536, 0.3)):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()

    # Referenced tif
    raster = gdal.Open(tif_ref)
    target_proj = raster.GetProjectionRef()
    rasterband = raster.GetRasterBand(1)
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Coordinate transformation to the same projection
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=target_proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname+validname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))
    index_isv = defn.GetFieldIndex(validname[0])
    # create iou
    if defn.GetFieldIndex('diou') == -1:
        lyr.CreateField(ogr.FieldDefn('diou', ogr.OFTReal))

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Check if the feature is valid (e.g., meet the condition 1)
        value_isv = feature.GetFieldAsInteger(index_isv)
        if value_isv == 0:
            continue
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent

        # Tif 1: the reference file
        # Specify offset and rows and columns to read
        xoff = int((minX - xOrigin)/pixelWidth)
        yoff = int((yOrigin - maxY)/pixelHeight)
        xcount = int((maxX - minX)/pixelWidth) # +1
        ycount = int((maxY - minY)/pixelHeight) # +1
        out_data1 = rasterband.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint8)
        out_data1 = (out_data1>condition[0]).astype(numpy.uint8)

        #Tif 2: Clip & warp the reference images
        OutTile2 = gdal.Warp("tmp.vrt", vrt_file, format='VRT',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj)
        out_data2 = OutTile2.GetRasterBand(1)
        out_data2 = out_data2.ReadAsArray()
        out_data2 = (out_data2>condition[0]).astype(numpy.uint8)
        OutTile2 = None
        isum = out_data2.sum()
        icount = out_data2.size
        ivalid2 = 1 if (isum >= condition[1]) and (icount >= condition[2]) else 0

        # Absolute difference: check dimension
        if out_data1.shape == out_data2.shape:
            diou = 1-calculate_iou(out_data1, out_data2)
            diff = (out_data1!=out_data2).astype(numpy.uint8)
            diff = diff.sum()
        else:
            diff = 65536 # means delete
            diou = 1

        # Condition 2: difference should be lower than T2 (0.3)
        # ivalid3 = (diff/icount <= condition[3]).astype(numpy.uint8)
        ivalid3 = (diou <= condition[3]) # and ((diff/icount <= 0.3))
        ivalid3 = ivalid3.astype(numpy.uint8)

        ivalid4 = 1 if (ivalid2==1) and (ivalid3==1) else 0
        # Create fields: sum & count
        feature.SetField2(fieldname[0], isum)
        feature.SetField2(fieldname[1], icount)
        feature.SetField2(fieldname[2], diff)

        feature.SetField2(validname[1], ivalid2)
        feature.SetField2(validname[2], ivalid3)
        feature.SetField2(validname[3], ivalid4)

        feature.SetField2('diou', diou)

        lyr.SetFeature(feature)
        feature = None

    # Close dataset
    raster = None
    VectorDataset = None
    lyr = None

def array2raster(newRasterfn, array, originX, originY,
                 pixelWidth, pixelHeight, proj,
                 datatype=gdal.GDT_Byte,
                 nodata=0):
    rows, cols = array.shape[:2]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, datatype,
                              options=['COMPRESS=PACKBITS'])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outRaster.SetProjection(proj)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.SetNoDataValue(nodata) # set nodata
    outband.FlushCache() ##saves to disk!!
    outRaster = None
    outband = None


# clip valid samples, 2023.9.18
def clip_twotiff_valid(tif_ref, vrt_file, shp_file,
                       respath, savename='FID',
                       subdir=('bh', 'cbra'),
                       validname='isv4',
                       nresolution=2.5,):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()
    citycode = os.path.basename(shp_file)
    citycode = citycode.split('_')[0]

    # Referenced tif
    raster = gdal.Open(tif_ref, 0)
    target_proj = raster.GetProjectionRef()
    rasterband = raster.GetRasterBand(1)
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Coordinate transformation to the same projection
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference(wkt=target_proj)
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

    # Check whether field exist
    defn = lyr.GetLayerDefn()
    if defn.GetFieldIndex(savename) == -1:
        return 0
    if defn.GetFieldIndex(validname) == -1:
        return 0
    index_isv = defn.GetFieldIndex(validname)
    index_id = defn.GetFieldIndex(savename)

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Check if the feature is valid (e.g., meet the condition 1)
        value_isv = feature.GetFieldAsInteger(index_isv)
        value_id = feature.GetFieldAsInteger(index_id)
        if value_isv == 0:
            continue
        basename = citycode+'_'+str(value_id) + '.tif'
        savepath1 = os.path.join(respath, subdir[0], basename)
        savepath2 = os.path.join(respath, subdir[1], basename)

        # Obtain the extent
        geom = feature.GetGeometryRef()
        geom.Transform(coordTrans)
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent

        # Tif 1: the reference file
        # Specify offset and rows and columns to read
        xoff = int((minX - xOrigin)/pixelWidth)
        yoff = int((yOrigin - maxY)/pixelHeight)
        xcount = int((maxX - minX)/pixelWidth) # +1
        ycount = int((maxY - minY)/pixelHeight) # +1
        out_data1 = rasterband.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint8)
        # Save to file
        array2raster(savepath1, out_data1, minX, maxY,
                     pixelWidth, -pixelHeight, target_proj,
                     datatype=rasterband.DataType,
                     nodata=0)

        #Tif 2: Clip & warp the reference images
        gdal.Warp(savepath2, vrt_file, format='GTiff',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj,
                            creationOptions=['COMPRESS=PACKBITS'])
        # translateoptions = gdal.TranslateOptions(format='GTiff',
        #                                   creationOptions=['COMPRESS=PACKBITS'])
        # gdal.Translate(savepath2, "tmp.vrt", options=translateoptions)
        feature = None

    # Close dataset
    raster = None
    VectorDataset = None
    lyr = None


# calculate the mean and std of raster according to each grid
def Fishgrid_stats(tif_file, shp_file, fieldname=('sum', 'count', 'isv'),
                   condition=(0, 2000, 65536)):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()
    # Create field
    defn = lyr.GetLayerDefn()
    for ifield in fieldname:
        if defn.GetFieldIndex(ifield) == -1:
            lyr.CreateField(ogr.FieldDefn(ifield, ogr.OFTInteger64))
    # Raster
    raster = gdal.Open(tif_file, 0)
    rasterband = raster.GetRasterBand(1)
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    for feature in lyr:
        geom = feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()
        # Specify offset and rows and columns to read
        xoff = int((minX - xOrigin)/pixelWidth)
        yoff = int((yOrigin - maxY)/pixelHeight)
        xcount = int((maxX - minX)/pixelWidth) # +1
        ycount = int((maxY - minY)/pixelHeight) # +1
        # Read data
        out_data = rasterband.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint8)
        out_data = (out_data>condition[0]).astype(numpy.uint8)
        isum = out_data.sum()
        icount = out_data.size
        ivalid = 1 if (isum>=condition[1]) and (icount>=condition[2]) else 0
        # Set field
        feature.SetField2(fieldname[0], isum)
        feature.SetField2(fieldname[1], icount)
        feature.SetField2(fieldname[2], ivalid)
        lyr.SetFeature(feature)

    # Close
    raster = None
    VectorDataset = None
    lyr = None


def Count_fishgrid_valid(shp_file, fieldname='isv4'):#('isv2', 'isv3')):
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 1) # read & write
    lyr = VectorDataset.GetLayer()
    # FieldName
    defn = lyr.GetLayerDefn()
    index_isv2 = defn.GetFieldIndex(fieldname) # area of CBRA
    # index_isv3 = defn.GetFieldIndex(fieldname[1]) # difference
    num = 0
    if (index_isv2!=-1): # and (index_isv3!=-1):
        for feature in lyr:
            value2 = feature.GetFieldAsInteger(index_isv2)
            # value3 = feature.GetFieldAsInteger(index_isv3)
            if (value2==1): #and (value3==1): # should meet the two conditions
                num += 1

    VectorDataset = None
    lyr = None
    return num


def download_sentinel12(query_shp, ref_shp, resroot, typelist=['s1_vvvhratio', 's2_rgbnir'], year='2020'):
    '''
    :param query_shp: the range of each city
    :param ref_shp: the reference. esa_worldcover_grid_composites.fgb
    :param type: "landcover, s1_vvvhratio, s2_rgbnir, s2_swir, s2_ndvi"
    :param resroot: the save path
    :return: the overlapped sentinel-1/2 images
    '''

    imagelist = []
    # Load data grid and region of interest (ROI)
    grid = gpd.read_file(ref_shp)

    query = gpd.read_file(query_shp)
    query = query.to_crs(grid.crs)
    minX, minY, maxX, maxY = query.total_bounds
    geom = Polygon([(minX, maxY), (maxX, maxY), (maxX, minY), (minX, minY), (minX, maxY)])
    # get grid tiles intersecting AOI
    tiles = grid[grid.intersects(geom)]

    fplist = list()
    # download
    for type in typelist:
        respath = os.path.join(resroot, type)
        if type == "landcover":
            if year == "2020":
                for tile in tiles.ll_tile:
                    url = f"s3://esa-worldcover/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
                    fp = os.path.join(respath, os.path.basename(url))
                    fplist.append(fp)
                    if os.path.exists(fp):
                        continue
                    os.system(f"aws s3 cp {url} {respath} --no-sign-request")
            else:
                for tile in tiles.ll_tile:
                    url = f"s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
                    fp = os.path.join(respath, os.path.basename(url))
                    fplist.append(fp)
                    if os.path.exists(fp):
                        continue
                    os.system(f"aws s3 cp {url} {respath} --no-sign-request")
        else:
            field = type + '_' + year
            if field in grid.columns:
                for tile in tiles[field]:
                    fp = os.path.join(respath, os.path.basename(tile))
                    fplist.append(fp)
                    if os.path.exists(fp):
                        continue
                    os.system(f"aws s3 cp {tile} {respath} --no-sign-request")

    return fplist


def clip_tiff_valid(tif_file, shp_file,
                    resroot, subdir='sen1',
                    validname='isv4',
                    savename='FID',
                    nresolution=2.5,):
    # Vector
    VectorDriver = ogr.GetDriverByName('ESRI Shapefile')  # intialize vector
    VectorDataset = VectorDriver.Open(shp_file, 0) # read & write
    lyr = VectorDataset.GetLayer()
    citycode = os.path.basename(shp_file)
    citycode = citycode.split('_')[0]
    target_proj = str(lyr.GetSpatialRef())

    # Check whether field exist
    defn = lyr.GetLayerDefn()
    if defn.GetFieldIndex(validname) == -1:
        return 0
    if defn.GetFieldIndex(savename) == -1:
        return 0
    index_isv = defn.GetFieldIndex(validname)
    index_id = defn.GetFieldIndex(savename)

    # feature = lyr.GetNetFeature()  # select the first polygon (the circle shown in image)
    # Loop over all
    for feature in lyr:
        # feature = lyr.GetNextFeature()
        # Check if the feature is valid (e.g., meet the condition 1)
        value_isv = feature.GetFieldAsInteger(index_isv)
        value_id = feature.GetFieldAsInteger(index_id)
        if value_isv == 0:
            continue
        basename = citycode+'_'+str(value_id) + '.tif'
        savepath = os.path.join(resroot, subdir, basename)

        # Obtain the extent
        geom = feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope() # get the extent

        #Tif 2: Clip & warp the reference images
        gdal.Warp(savepath, tif_file, format='GTiff',
                            outputBounds=[minX, minY, maxX, maxY],
                            xRes=nresolution, yRes=nresolution,
                            dstSRS=target_proj,
                            creationOptions=['COMPRESS=PACKBITS']
                  )
        feature = None

    # Close dataset
    VectorDataset = None
    lyr = None


# reproject tiles, then merge, and finally clip
def clip_tiff_whole(query_shp, ref_grid, resroot, tif_path,
                    suffix='CNBH10m', nresolution=10,
                    srcnodata='nan', VRTNodata='nan'):
    citycode = os.path.basename(query_shp)
    citycode = citycode.split('_')[0]
    out_file = os.path.join(resroot, citycode+'.vrt')
    if os.path.exists(out_file):
        return
    # process
    queryOri = gpd.read_file(query_shp)
    query_srs = queryOri.crs.srs

    query = queryOri.to_crs(ref_grid.crs, inplace=False)
    minX, minY, maxX, maxY = query.total_bounds
    geom = Polygon([(minX, maxY), (maxX, maxY), (maxX, minY), (minX, minY), (minX, maxY)])
    # get grid tiles intersecting AOI
    tiles = ref_grid[ref_grid.intersects(geom)]
    # merge & clip to the range
    imglist = []
    for iname in tiles.location:
        if iname.endswith('.tif'):
            iname = iname[:-4]
        imglist.append(os.path.join(tif_path, f'{suffix}_{iname}.tif'))

    t0 = time()
    # merge to one large file in VRT form
    minX, minY, maxX, maxY = queryOri.total_bounds
    if len(imglist)==1:
        tif_file = imglist[0]
        gdal.Warp(out_file, tif_file, format='VRT',
                  outputBounds=[minX, minY, maxX, maxY],
                  xRes=nresolution, yRes=nresolution,
                  # srcNodata=srcnodata, dstNodata=VRTNodata,
                  dstSRS=query_srs)
    else:
        # check the projection
        tif_plist = []
        for i in imglist:
            _, _, _, tmp_proj = get_tif_meta(i)
            if tmp_proj not in tif_plist:
                tif_plist.append(tmp_proj)
        if len(tif_plist)>1:
            print('need to reproject to the same projection')
            warp_list = []
            for i in imglist:
                iname = os.path.basename(i)[:-4]
                tmp_file = os.path.join(resroot, citycode+'_'+iname+'_warp.vrt')
                warpoption = gdal.WarpOptions(format='VRT',
                         srcNodata=srcnodata, dstNodata=VRTNodata,
                          dstSRS=query_srs)
                gdal.Warp(tmp_file, i, options=warpoption)
                warp_list.append(tmp_file)
        else:
            warp_list = imglist

        merge_file = os.path.join(resroot, citycode+'_merge.vrt')
        merge_alltif(warp_list, merge_file, srcNodata=srcnodata, VRTNodata=VRTNodata)
        gdal.Warp(out_file, merge_file, format='VRT',
                  outputBounds=[minX, minY, maxX, maxY],
                  xRes=nresolution, yRes=nresolution,
                  # srcNodata=srcnodata, dstNodata=VRTNodata,
                  dstSRS=query_srs)

    # clip whole images
    # clip_tiff_valid(tif_file, query_shp,
    #                 sampleroot, subdir=itype,
    #                 validname='isv4',
    #                 savename='FID',
    #                 nresolution=10)
    print('time elaps: %.2f'%(time()-t0))


def cal_rmse(cbra_path, cnbh_path, bh_path, iname):
    # CNBH 10m height: [6, 40]
    cnbh = cv2.imread(os.path.join(cnbh_path, iname), cv2.IMREAD_UNCHANGED)
    cnbh = np.nan_to_num(cnbh)
    cnbh = cv2.resize(cnbh, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    # CBRA 2.5m building area: [0, 1]
    cbra = cv2.imread(os.path.join(cbra_path, iname), cv2.IMREAD_UNCHANGED)
    cbra[cbra == 255] = 1
    # referenced building height at 2.5 m
    bh = cv2.imread(os.path.join(bh_path, iname), cv2.IMREAD_UNCHANGED)
    bh[(bh<=2) & (bh>0)] = 3  # set these buildings to 9m
    bh = bh.astype('float')*3 # convert to building height
    # CNBH vs BH
    cnbh_mask = cnbh * cbra
    diff = (cnbh_mask - bh).flatten()
    diff_valid = diff[diff!=0]
    rmse = np.sqrt((diff_valid**2).mean())
    return rmse

def main_select_heightvalid():
    cnbh_path = r'D:\data\Landcover\samples62\cnbh'
    bh_path = r'D:\data\Landcover\samples62\bh'
    cbra_path = r'D:\data\Landcover\samples62\cbra'
    iname = 'baoding_130.tif'
    rmse = cal_rmse(cbra_path, cnbh_path, bh_path, iname)
    print(rmse)


## merge CNBH 10m height for comparison
def main_proc_CNBH10m():
    '''
    # merge for each city and convert the projection to the same
    ipath = r'D:\data\Landcover\CNBH_10m'
    respath = os.path.join(ipath, 'range62')
    os.makedirs(respath, exist_ok=True)

    filelist = Path(ipath).glob('*.tif')
    filelist = [str(i) for i in filelist]
    print(len(filelist))
    # generate grid for each region
    # the projection is set to the same
    nresolution = 10
    outfile = os.path.join(respath, 'all_grid.shp')
    Raster_extent_prj(filelist, outfile,
                      locName='location', locType=ogr.OFTString,
                      yearName='2020',
                      target_crs=4326)
    '''
    # warp and merge
    # nresolution = 10
    # _,_,_,proj = get_tif_meta(filelist[0])
    # iname = os.path.basename(filelist[0])
    # out_file = os.path.join(respath, iname[:-4]+'.vrt')
    # gdal.Warp(out_file, filelist[0], format='VRT',
    #                             # outputBounds=[minX, minY, maxX, maxY],
    #                             xRes=nresolution, yRes=nresolution,
    #                             dstSRS=proj)

    # Clip CNBH 10 m data
    '''
    cnbh_path = r'D:\data\Landcover\CNBH_10m'
    ref_shp = os.path.join(cnbh_path, 'range62', 'all_grid.shp')
    ref_grid = gpd.read_file(ref_shp)
    resroot = os.path.join(cnbh_path, 'range62')
    itype = 'CNBH10m'
    nresolution = 10

    # Loop over other regions, clip to the range of query_shp
    datapath = r'D:\data\Landcover\city62'
    filelist = [str(file) for file in Path(datapath).rglob("*_grid.shp")]
    print(len(filelist))

    # for query_shp in filelist:
    # query_shp = r'D:\data\Landcover\city77\beijing\Beijing_grid.shp'
    # query_shp = r'D:\data\Landcover\city77\baotou\Baotou_grid.shp'#
    for query_shp in filelist:
        # clip the whole image
        clip_tiff_whole(query_shp, ref_grid, resroot, cnbh_path,
                    suffix=itype, nresolution=nresolution,
                    srcnodata='nan', VRTNodata='nan')
    '''


## merge CNBH 10m height for comparison
def main_proc_CBRA():
    '''
    # merge for each city and convert the projection to the same
    ipath = r'D:\data\Landcover\CNBH_10m'
    respath = os.path.join(ipath, 'range62')
    os.makedirs(respath, exist_ok=True)

    filelist = Path(ipath).glob('*.tif')
    filelist = [str(i) for i in filelist]
    print(len(filelist))
    # generate grid for each region
    # the projection is set to the same
    nresolution = 10
    outfile = os.path.join(respath, 'all_grid.shp')
    Raster_extent_prj(filelist, outfile,
                      locName='location', locType=ogr.OFTString,
                      yearName='2020',
                      target_crs=4326)
    '''
    # warp and merge
    # nresolution = 10
    # _,_,_,proj = get_tif_meta(filelist[0])
    # iname = os.path.basename(filelist[0])
    # out_file = os.path.join(respath, iname[:-4]+'.vrt')
    # gdal.Warp(out_file, filelist[0], format='VRT',
    #                             # outputBounds=[minX, minY, maxX, maxY],
    #                             xRes=nresolution, yRes=nresolution,
    #                             dstSRS=proj)

    # Clip CBRA
    cnbh_path = r'D:\data\Landcover\CBRA_2020'
    ref_shp = os.path.join(cnbh_path, 'all_grid.shp')
    ref_grid = gpd.read_file(ref_shp)
    resroot = os.path.join(cnbh_path, 'range62')
    os.makedirs(resroot, exist_ok=True)
    itype = 'CBRA_2020'
    nresolution = 10

    # Loop over other regions, clip to the range of query_shp
    datapath = r'D:\data\Landcover\city62'
    filelist = [str(file) for file in Path(datapath).rglob("*_grid.shp")]
    print(len(filelist))

    # for query_shp in filelist:
    # query_shp = r'D:\data\Landcover\city77\beijing\Beijing_grid.shp'
    # query_shp = r'D:\data\Landcover\city77\baotou\Baotou_grid.shp'#
    for query_shp in filelist:
        # clip the whole image
        clip_tiff_whole(query_shp, ref_grid, resroot, cnbh_path,
                    suffix=itype, nresolution=nresolution,
                    srcnodata=0, VRTNodata=0)


## download and clip sentinel-1/2 images
def main_proc_sentinel12():
    # Dowload sentinel-1/2 images
    '''
    datapath = r'D:\data\Landcover\city77'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    ref_shp = r'D:\data\googleimg\esa_worldcover_grid_composites.fgb'
    resroot = r'D:\data\Landcover'
    typelist = ['s1_vvvhratio', 's2_rgbnir']
    for type in typelist:
        respath = os.path.join(resroot, type)
        os.makedirs(respath, exist_ok=True)

    for query_shp in filelist:
        print(query_shp)
        download_sentinel12(query_shp, ref_shp, resroot, typelist=typelist, year='2020')
    '''
    # Clip sentinel-1/2 images
    datapath = r'D:\data\Landcover\city77'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    ref_shp = r'D:\data\googleimg\esa_worldcover_grid_composites.fgb'
    resroot = r'D:\data\Landcover'
    sampleroot = os.path.join(resroot, 'samples')
    typelist = ['s1_vvvhratio', 's2_rgbnir']
    # make dirs
    for type in typelist:
        tmp = os.path.join(sampleroot, type)
        os.makedirs(tmp, exist_ok=True)
    # Loop over all files
    for query_shp in filelist[1:]:
        query_shp = str(query_shp)
        print(query_shp)
        fplist = download_sentinel12(query_shp, ref_shp, resroot, typelist=typelist, year='2020')
        for type in typelist:
            imglist = list(filter(lambda x: type in x, fplist))
            print(imglist)
            # merge to one large file in VRT form
            if len(imglist)==1:
                tif_file = imglist[0]
            else:
                citycode = os.path.basename(query_shp)
                citycode = citycode.split('_')[0]
                # merge
                tif_file = os.path.join(resroot, citycode+'_'+type)
                merge_alltif(imglist, tif_file)
                tif_file = tif_file+'.vrt'
            # clip samples, datatype is set to uint16
            t0 = time()
            clip_tiff_valid(tif_file, query_shp,
                            sampleroot, subdir=type,
                            validname='isv4',
                            savename='FID',
                            nresolution=10,)
            print('time elaps: %.2f s'%(time()-t0))


## stats valid samples & clip samples
def main_sample_stats():
    # Stats valid samples
    '''
    datapath = r'D:\data\Landcover\city62'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    num_valid = []
    name_valid = []
    for file in filelist:
        file = str(file)
        print(file)
        iname = os.path.basename(file)
        iname = iname.split('_')[0]
        num = Count_fishgrid_valid(file, fieldname='isv4') #('isv2', 'isv3'))
        num_valid.append(num)
        name_valid.append(iname)
        print('number of valid: %d'%num)

    print('total number is: %d'%sum(num_valid))
    data = pd.DataFrame(data=num_valid, index=name_valid, columns=['num_valid'])
    data.to_csv('tmp_num_valid.csv')
    # total number is: 18137
    '''


def main_clip_bh_sample():
    # Clip samples
    # vrt_file = r'D:\data\Landcover\CBRA_2020\all_tif.vrt'
    datapath = r'D:\data\Landcover\city62'
    respath = r'D:\data\Landcover\samples62'
    subdir = 'bh'
    os.makedirs(os.path.join(respath, subdir), exist_ok=True)
    # for i in subdir:
    #     os.makedirs(os.path.join(respath, i), exist_ok=True)
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*.tif")]
    print(len(filelist))
    for out_tiff in filelist:
        out_tiff = str(out_tiff)
        print('process: %s'% out_tiff)
        t0 = time()
        shp_file = out_tiff[:-4] + '_grid.shp'
        clip_tiff_valid(out_tiff, shp_file,
                        respath, subdir=subdir,
                        validname='isv4',
                        savename='FID',
                        nresolution=2.5,
                        dstNodata=0)
        print('time elaps: %.2f s' % (time() - t0))


# CNBH 10 m
def main_clip_cnbh_sample():
    # Clip samples
    # vrt_file = r'D:\data\Landcover\CBRA_2020\all_tif.vrt'
    tifpath = r'D:\data\Landcover\CNBH_10m\range62'
    respath = r'D:\data\Landcover\samples62'
    shppath = r'D:\data\Landcover\city62'
    subdir = 'cnbh'
    os.makedirs(os.path.join(respath, subdir), exist_ok=True)
    # for i in subdir:
    #     os.makedirs(os.path.join(respath, i), exist_ok=True)
    ipath = Path(shppath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    for query_shp in filelist[:1]:
        query_shp = str(query_shp)
        print('process: %s'% query_shp)
        t0 = time()
        iname = Path(query_shp).stem
        iname = iname.split('_')[0]
        tif_file = os.path.join(tifpath, iname+'.vrt')
        clip_tiff_valid(tif_file, query_shp,
                        respath, subdir=subdir,
                        validname='isv4',
                        savename='FID',
                        nresolution=10)
        print('time elaps: %.2f s' % (time() - t0))


# CNBH 10 m
def main_clip_cbra_sample():
    # Clip samples
    # vrt_file = r'D:\data\Landcover\CBRA_2020\all_tif.vrt'
    tifpath = r'D:\data\Landcover\CBRA_2020\range62'
    respath = r'D:\data\Landcover\samples62'
    shppath = r'D:\data\Landcover\city62'
    subdir = 'cbra'
    os.makedirs(os.path.join(respath, subdir), exist_ok=True)
    # for i in subdir:
    #     os.makedirs(os.path.join(respath, i), exist_ok=True)
    ipath = Path(shppath)
    filelist = [file for file in ipath.rglob("*_grid.shp")]
    print(len(filelist))
    for query_shp in filelist[:1]:
        query_shp = str(query_shp)
        print('process: %s'% query_shp)
        t0 = time()
        iname = Path(query_shp).stem
        iname = iname.split('_')[0]
        tif_file = os.path.join(tifpath, iname+'.vrt')
        clip_tiff_valid(tif_file, query_shp,
                        respath, subdir=subdir,
                        validname='isv4',
                        savename='FID',
                        nresolution=2.5)
        print('time elaps: %.2f s' % (time() - t0))


## sample selection
def main_sample_selection():
    # Grid generation & stats for each grid
    '''
    datapath = r'D:\data\Landcover\city62'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*.tif")]
    print(len(filelist))
    for out_tiff in filelist:
        #print(out_tiff)
        out_tiff = str(out_tiff)
        grid_shp = out_tiff[:-4] + '_grid.shp'
        if not os.path.exists(grid_shp):
            print('create grid: %s'%grid_shp)
            Fishgridnew(out_tiff, window_size=256)
        # grid statisti
        print('stats: %s'%grid_shp)
        Fishgrid_stats(out_tiff, out_tiff[:-4]+'_grid.shp',
                           fieldname=('sum', 'count', 'isv'),
                           condition=(0, 4000, 65536))
    '''
    # Merge all
    '''
    cbra_path = r'D:\ziliao\polyu\data\CBRA_2020'
    imglist = glob(os.path.join(cbra_path, '*.tif'))
    print(len(imglist))
    outfile = os.path.join(cbra_path, 'all_tif.vrt')
    merge_alltif(imglist, outfile)
    '''
    # Test clip vrt
    # georef_tif = r'D:\data\Landcover\city62\baoding\baoding.tif'
    # grid_path = georef_tif[:-4]+'_grid.shp'
    # _, _, _, proj = get_tif_meta(georef_tif)
    # ref_tif = r'D:\data\Landcover\CBRA_2020\all_tif.vrt'
    # clip_vrt(ref_tif, grid_path, "tmp/tmp_clip.tif", proj)

    # Compare two tiffs: CBRA & height
    '''
    vrt_file = r'D:\data\Landcover\CBRA_2020\all_tif.vrt'
    datapath = r'D:\data\Landcover\city62'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*.tif")]
    print(len(filelist))

    for out_tiff in filelist:
        out_tiff = str(out_tiff)
        print(out_tiff)
        # if 'Chongqing' not in out_tiff:
        #     continue
        t0 = time()
        shp_file = out_tiff[:-4]+'_grid.shp'
        compare_twotiff_valid_iou(out_tiff, vrt_file, shp_file,
                          fieldname=('vrt_sum', 'vrt_count', 'absdiff'),
                          validname=('isv', 'isv2', 'isv3', 'isv4'),
                          nresolution=2.5,
                          condition=(0, 4000, 65536, 0.65))
        print('time eplas: %.2f s'%(time()-t0))
    '''

def main2():
    t0 = time()
    shp_file = r'D:\ziliao\polyu\data\height\aomen\Macao_Buildings_DWG-Polygon.shp'
    idir = os.path.dirname(shp_file)
    iname = os.path.basename(shp_file)
    iname = iname.split('_')[0]
    out_tiff = os.path.join(idir, iname+'.tif')
    print('process: %s' % shp_file)
    # add floornum field
    # addField_byExpression(shp_file, newFieldName='FloorNum')
    # # vector to raster
    # shp_to_tiff(shp_file=shp_file, output_tiff=out_tiff,
    #             attribute='FloorNum', nresolution=2.5)  # 2.5 meters
    # print('time eplas: %.2f'%(time()-t0))
    # Generate grid
    # Fishgridnew(out_tiff, window_size=256)
    #
    Fishgrid_stats(out_tiff, out_tiff[:-4]+'_grid.shp',
                   fieldname=('sum', 'count', 'isv'),
                   condition=(0, 2000, 65536))


def main_shp2tif():
    # convert vector to raster
    # Loop over all shp files
    # datapath = r'D:\data\Landcover\city62'
    # ipath = Path(datapath)
    # filelist = [file for file in ipath.rglob("*_clip.shp")]
    # print(len(filelist))
    # for file in filelist:
    #     shp_file = str(file)
    #     out_tiff = os.path.join(str(file.parent), file.stem.split('_')[0]+'.tif')
    #     if not os.path.exists(out_tiff):
    #         print('process: %s' % shp_file)
    #         # reproject
    #         data = gpd.read_file(shp_file)
    #         espg = data.estimate_utm_crs()
    #         data = data.to_crs(crs=espg)
    #         data.to_file(shp_file[:-4]+'_utm.shp')
    #         shp_file = shp_file[:-4]+'_utm.shp'
    #         # vector to raster
    #         shp_to_tiff(shp_file=shp_file, output_tiff=out_tiff,
    #                     attribute='Floor', nresolution=2.5)  # 2.5 meters
    '''
    # Generate Fishnet with grids of 256 x 256 pixels each
    # first get the range of raster, and then generate 256 x 256 grid.
    datapath = r'D:\data\Landcover\city62'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*.tif")]
    print(len(filelist))
    # print(filelist[:5])
    window_size = 256

    for file in filelist:
        tif_path = str(file)
        outfile = os.path.join(str(file.parent), file.stem.split('_')[0]+'_grid.shp')

        width, height, geotrans, proj = get_tif_meta(tif_path)
        xres = geotrans[1]
        yres = geotrans[5]
        xmin = geotrans[0] # top left x
        ymax = geotrans[3] # top left y
        xmax = xmin + xres*width
        ymin = ymax + yres*height

        grid_size = int(window_size*xres)
        Fishgrid(outfile, xmin, xmax, ymin, ymax,
                 grid_size, grid_size,
                 proj)

        # print(grid_size)
        # print(geotrans)
        # print(proj)
    '''

    '''
    # Generate the exent of raster (CBRA)
    raster_path = r'D:\data\Landcover\CBRA_2020'
    filelist = glob(os.path.join(raster_path, "*.tif"))
    outfile = os.path.join(raster_path, 'all_grid.shp')
    Raster_extent(filelist, outfile,
                  locName='location', locType=ogr.OFTString,
                  yearName='2020')
    '''
    # Seclect samples according to two conditions grid by grid.
    # Condition 1: The area of CBRA and height should both be larger than T1 (4000 pixels)
    # Condition 2: The difference of CBRA and height should be lower than T2 (30%)
    # Methods: write attributes into the field of grid shapefile

    # zonal stats over height data
    # raster_file = r'D:\data\Landcover\city62\baoding\baoding.tif'
    # shp_file = r'D:\data\Landcover\city62\baoding\baoding_grid.shp'
    # zonal_stats(shp_file, raster_file)
    # zonal stats over building area

    # 1. merge all patches 2. clip
    # cbra_path = r'D:\data\Land cover\CBRA_2020'
    # imglist = glob(os.path.join(cbra_path, '*.tif'))
    # print(len(imglist))
    # outfile = os.path.join(cbra_path, 'all_tif.vrt')
    # merge_alltif(imglist, outfile)

    # raster_vrt = gdal.Open(outfile+'.vrt')
    # print(raster_vrt.GetGeoTransform())
    # print(raster_vrt.GetProjection())
    # grid_path = r'D:\data\Land cover\city77\aomen\Macao_grid.shp'
    # grid = gpd.read_file(grid_path)
    # polygon = grid['geometry']
    # tmpfile = 'clip.tif'

    # Clip CBRA files using VRT
    # georef_tif = r'D:\data\Land cover\city77\aomen\Macao.tif'
    # _, _, _, proj = get_tif_meta(georef_tif)
    # srs = osr.SpatialReference(wkt=proj)
    # clip_vrt(outfile, grid_path, "tmp_clip.tif", proj)

    # Compare two tiffs
    # tif_file1 = r'D:\data\Land cover\city77\baoding\Baoding.tif'
    # tif_file2 = r'D:\data\Land cover\CBRA_2020\all_tif.vrt'
    # shp_file = r'D:\data\Land cover\city77\baoding\Baoding_grid.shp'
    # _, _, _, target_proj = get_tif_meta(tif_file1)
    # compare_twotiff(tif_file1, tif_file2, shp_file, target_proj,
    #                 fieldname=('sum', 'count', 'vrt_sum', 'vrt_count',
    #                            'absdiff'),
    #                 nresolution=2.5,
    #                 condition=(0, 0))

    # Compare two tiffs over all cities
    '''
    datapath = r'D:\data\Land cover\city77'
    ipath = Path(datapath)
    filelist = [file for file in ipath.rglob("*.tif")]
    print(len(filelist))
    print(filelist[:5])
    # CBRA
    tif_file2 = r'D:\data\Land cover\CBRA_2020\all_tif.vrt'
    # target_obj
    ref_file = r'D:\data\Land cover\city77\aomen\Macao.tif'
    _, _, _, target_proj = get_tif_meta(ref_file)

    for file in filelist:
        tif_file1 = str(file)
        print(tif_file1)
        shp_file = os.path.join(str(file.parent), file.stem+'_grid.shp')
        compare_twotiff(tif_file1, tif_file2, shp_file, target_proj,
                        fieldname=('sum', 'count', 'vrt_sum', 'vrt_count',
                                   'absdiff'),
                        nresolution=2.5,
                        condition=(0, 0))
    '''
    '''
    src_raster = None
    with rasterio.open(outfile, 'r') as src:
        profile = src.profile
        print(profile)
        out_image, out_transform = mask(src, [polygon[10]], crop=True)
        print(out_image.shape)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1], # C H W
                         "width": out_image.shape[2],
                         "transform": out_transform})
        with rasterio.open(tmpfile, "w", **out_meta) as dest:
            dest.write(out_image)
    with rasterio.open(tmpfile, 'r') as src:

    '''
    # zonal_stats(shp_file, outfile, fieldName=['vrt_sum', 'vrt_count'])


if __name__=="__main__":
    # step 1: shp2raster
    # main_shp2tif()

    # step 2: sample selection
    # main_sample_selection()
    main_select_heightvalid()

    # step 3: sample stats
    # main_sample_stats()

    # clip sample
    # main_clip_bh_sample()
    # main_clip_cnbh_sample()
    # main_proc_CBRA()
    # main_clip_cbra_sample()

    # step 4: clip CNBH10m image
    # main_proc_CNBH10m()

    # main6()
    # datapath = r'D:\data\Landcover\CNBH_10m\CNBH10m_X135Y49.tif'
    # raster = gdal.Open(datapath, 0)
    # band1 = raster.GetRasterBand(1)
    # v = band1.ReadAsArray()
    # print(v)
    # raster = None
    # datapath = r'D:\data\Landcover\CNBH_10m\range\Baotou_CNBH10m_merge.vrt'
    # respath = datapath[:-4]+'_trans.tif'
    # gdal.Translate(respath, datapath, options=gdal.TranslateOptions())
    # gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
    # gdal.Translate(respath, datapath, format='GTiff',
    #                outputType=gdal.GDT_Byte, creationOptions=['COMPRESS=LZW'])
    # gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', None)

    # iname = 'baoding_5.tif'
    # cnbh_path = r'D:\data\Landcover\samples62\cnbh'
    # bh_path = r'D:\data\Landcover\samples62\bh'
    #
    # cnbh = cv2.imread(os.path.join(cnbh_path, iname), cv2.IMREAD_UNCHANGED)
    # bh = cv2.imread(os.path.join(bh_path, iname), cv2.IMREAD_UNCHANGED)
    #
    # plt.subplot(1,2,1)
    # plt.imshow(cnbh)
    # plt.subplot(1,2,2)
    # plt.imshow(bh)
    # plt.show()

    # # heatmap = cv2.applyColorMap(data, cv2.COLORMAP_JET)
    # # a colormap and a normalization instance
    # cmap = matplotlib.colormaps['jet']
    # imax = data.max()
    # print(imax)
    # norm = plt.Normalize(vmin=0, vmax=20)
    # # map the normalized data to colors
    # # image is now RGBA (512x512x4)
    # image = cmap(norm(data))
    # # save the image
    # plt.imsave('tmp/test.png', image)
    # plt.imshow(image)
    # plt.show()