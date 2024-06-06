# for prediction
import torch
import numpy as np
import time
# bug: weight zeros instead of ones

# two output
def predict_whole_image_over2(model, image, r, c, num_class=2, grid=512, stride=256, device='cuda'):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    overlay prediction
    change pad to the out space
    r,c: original shape
    rows, cols: changed shape
    '''
    model.eval()
    _, _, rows, cols = image.shape
#     n,b,r,c = image.shape
#     rows= math.ceil((r-grid)/(stride))*stride+grid
#     cols= math.ceil((c-grid)/(stride))*stride+grid
#     rows=math.ceil(rows)
#     cols=math.ceil(cols)
    print('rows is {}, cols is {}'.format(rows,cols))
    # image_= np.pad(image,((0,0),(0,0),(0,rows-r), (0,cols-c), ),'symmetric')
    weight = np.zeros((rows, cols), dtype=np.int8)
    res = np.zeros((num_class, rows, cols),dtype=np.float32)
    num_patch= len(range(0,rows,stride))*len(range(0,cols,stride))
    print('num of patch is',num_patch)
    k=0
    for i in range(0,rows, stride):
        for j in range(0, cols, stride):
            start=time.time()
            patch = image[0:,0:,i:i+grid,j:j+grid]
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred, pred2 = model(patch.to(device))
            pred = pred.cpu().numpy()
            pred2 = pred2.cpu().numpy()
            res[0, i:i+grid,j:j+grid] += np.squeeze(pred) # H W
            res[1, i:i+grid,j:j+grid] += np.squeeze(pred2) # H W
            weight[i:i+grid,j:j+grid] += 1
            end=time.time()
            k=k+1
            if k % 500 ==0:
                print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
                #tif.imsave(os.path.join(ipath,'height{}_{}.tif'.format(i,j)),pred,dtype=np.float32)
    res= res/weight
    # res=np.argmax(res, axis=0)
    res = res[:,0:r,0:c].astype(np.float32)
    res = np.squeeze(res)
    return res

# one output
def predict_whole_image_over(model, image, r, c, num_class=1, grid=512, stride=256, device='cuda'):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    overlay prediction
    change pad to the out space
    r,c: original shape
    rows, cols: changed shape
    '''
    model.eval()
    _, _, rows, cols = image.shape
    print('rows is {}, cols is {}'.format(rows,cols))
    weight = np.zeros((rows, cols), dtype=np.int8) # start from zeros
    res = np.zeros((num_class, rows, cols),dtype=np.float32) # start from zeros
    num_patch= len(range(0,rows,stride))*len(range(0,cols,stride))
    print('num of patch is',num_patch)
    k=0
    for i in range(0,rows, stride):
        for j in range(0, cols, stride):
            start=time.time()
            patch = image[0:,0:,i:i+grid,j:j+grid]
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred = model(patch.to(device))
            pred = pred.cpu().numpy()
            res[:, i:i+grid,j:j+grid] += np.squeeze(pred) # H W
            weight[i:i+grid,j:j+grid] += 1
            end=time.time()
            k=k+1
            if k % 500 ==0:
                print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
    res= res/weight
    res = res[:,0:r,0:c].astype(np.float32)
    res = np.squeeze(res)
    return res


# three output
def predict_whole_image_over3(model, image, r, c, num_class=1, grid=512, stride=256, device='cuda'):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    overlay prediction
    change pad to the out space
    r,c: original shape
    rows, cols: changed shape
    '''
    model.eval()
    _, _, rows, cols = image.shape
    print('rows is {}, cols is {}'.format(rows,cols))
    weight = np.zeros((rows, cols), dtype=np.int8) # start from zeros
    res = np.zeros((num_class, rows, cols),dtype=np.float32) # start from zeros
    num_patch= len(range(0,rows,stride))*len(range(0,cols,stride))
    print('num of patch is',num_patch)
    k=0
    for i in range(0,rows, stride):
        for j in range(0, cols, stride):
            start=time.time()
            patch = image[0:,0:,i:i+grid,j:j+grid]
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred,_,_ = model(patch.to(device))
            pred = pred.cpu().numpy()
            res[:, i:i+grid,j:j+grid] += np.squeeze(pred) # H W
            weight[i:i+grid,j:j+grid] += 1
            end=time.time()
            k=k+1
            if k % 500 ==0:
                print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
    res= res/weight
    res = res[:,0:r,0:c].astype(np.float32)
    res = np.squeeze(res)
    return res
