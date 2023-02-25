import os
import csv
import cv2
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, datasets

def MakeCSV(data_dir, csv_name):
    data_folders = os.listdir(data_dir)
    data_folders.sort(key=len)
    f = open('./' + csv_name, 'w', newline='') 
    wr = csv.writer(f)


    for folder in data_folders:
        
        folder_dir = os.path.join(data_dir, folder) 

        if(os.path.isdir(folder_dir)):
            data_list = os.listdir(folder_dir)

            for data_name in data_list:
                wr.writerow([os.path.join(folder_dir, data_name), folder])


    f.close()


def CalcMeanStdFromCSV(filename):
    data_info = pd.read_csv('./'+filename+'.csv', header=None)

    image_arr = np.asarray(data_info.iloc[:, 0])

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    aver_mean = 0.0
    aver_std = 0.0
    counter = 0

    for dir in image_arr:

        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32)
        
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        
        std, mean = cv2.meanStdDev(img)
        aver_std += std[0][0]
        aver_mean += mean[0][0]
        counter += 1
    

    f = open(filename +'_std_mean.csv', 'w', newline='') 
    wr = csv.writer(f)
    wr.writerow([aver_std / counter, aver_mean / counter])
    f.close()

