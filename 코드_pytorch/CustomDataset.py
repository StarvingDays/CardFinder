import pandas as pd
import numpy as np
import cv2
from torch import from_numpy
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset  
class CustomDataset(Dataset):

    def __init__(self, csv_path, trans):

        self.data_info = pd.read_csv(csv_path, header=None)         # csv 파일 읽기
    
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])      # 0열에 있는 모든 행 데이터를(이미지 경로) 가져온다

        self.label_arr = np.asarray(self.data_info.iloc[:, 1])      # 1열에 있는 모든 행 데이터를(레이블값 : 폴더명) 가져온다
 
        self.data_len = len(self.data_info.index)                   # 데이터들의 인덱스 길이를 가져온다

        self.transform = trans                                      # transform의 Normalize 값을 가져온다

    def __getitem__(self, index):
   
        image_name = self.image_arr[index]                          # index에 위치한 파일경로를 가져온다

        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)          # 가져온 파일경로에서 이미지를 읽어온다

        img = cv2.normalize(img, None,                              # 이미지를 원소값이 0과 1사이로 갖는 float 타입 형태로 변환한다
            0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

        img_to_tensor = from_numpy(img).unsqueeze(0)                # 이미지의 channel 앞에 batch 1을 추가한다

        img_to_tensor = self.transform(img_to_tensor)               # transform Normalize 값 설정

        image_label = self.label_arr[index]                         # 인덱스에 위치한 레이블 값 가져오기

        return (img_to_tensor, image_label)                         # 텐서로 변환한 이미지와 레이블 값을 튜플 형태로 반환

    def __len__(self):
        return self.data_len

