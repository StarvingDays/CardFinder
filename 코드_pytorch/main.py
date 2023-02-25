from number_model import torch, nn, optim, F, Variable
from number_area_model import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from number_model import CNN
from MakeCSV import MakeCSV, CalcMeanStdFromCSV
from train_val import train, evaluate
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CustomDataset import CustomDataset
from torchvision import transforms
import sys
import cv2
import os
import pandas as pd
import numpy as np
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == "__main__":

    MakeCSV('./number_img', 'number_img.csv')
    CalcMeanStdFromCSV('./number_img')
    
    data_info = pd.read_csv('./number_img_std_mean.csv', header=None)
    std = np.asarray(data_info.iloc[:, 0])[0]
    mean = np.asarray(data_info.iloc[:, 1])[0]
          
    number_dataset = CustomDataset(
        './number_img.csv', 
        transforms.Compose([
            transforms.Normalize([std], [mean])
            ]))
    
    number_dataset_loader = torch.utils.data.DataLoader(
        dataset=number_dataset,
        batch_size=20,
        shuffle=True,
        num_workers = 4
        )
 

    number_dataset_size = number_dataset_loader.dataset.data_len

    model_number = CNN().to(DEVICE)

    optimizer_adam = torch.optim.Adam(model_number.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    max_mse = sys.maxsize

    for i in range(1, 81):
        train(i, model_number, DEVICE, 
            number_dataset_loader, optimizer_adam, 
            number_dataset_size, max_mse, 'number_script')