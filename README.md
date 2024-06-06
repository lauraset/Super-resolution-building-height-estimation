# A deep learning-based super-resolution method for building height estimation at 2.5 m spatial resolution in the Northern Hemisphere
Author: Yinxia Cao, Qihao Weng* | [Paper link](https://www.sciencedirect.com/science/article/pii/S0034425724002591) | Date: August 2024

## Dataset
- Download link in [google drive](https://drive.google.com/drive/folders/1ngeSyPWOUkj0DTS4M1zuSsbIdYeAHHhS?usp=drive_link) or[Onedrive](https://1drv.ms/f/s!AsLBo0q3zUjCgYRfYHgM8oSZqFsiFg?e=YfJTuf). The total size is 2.72G    
- Unzip them to a path (e.g., `data`)
- Split dataset into train/val/test set, see the directory `data`
 The specific spliting file is put in `BH_dataset.py`
- Obtain the statistics of the dataset, see the directory  `datasteglobe` 
The specifi file is `stats_dataset_globe.py`    

## Training
```commandline
python train.py
```

## Pretrained weights
available soon