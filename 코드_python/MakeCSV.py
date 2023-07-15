import os
import csv
import cv2
import pandas as pd
import numpy as np


def MakeCSV(data_dir, csv_name):                                                    # n-Samples행 2열(data_dir, label)을 지닌 .csv 파일을 생성하는 함수
    data_folders = os.listdir(data_dir)                                             # data 디렉토리 내부에 있는 폴더 이름들을 저장한다

    f = open('./' + csv_name, 'w', newline='')                                      
    wr = csv.writer(f)                                                              # .csv 파일 open


    for folder in data_folders:
        
        folder_dir = os.path.join(data_dir, folder)                                 # data 디렉토리명과 폴더명을 합쳐 폴더 디렉토리 경로를 생성

        if(os.path.isdir(folder_dir)):                                              # 디렉토리 유효성 확인
            data_list = os.listdir(folder_dir)                                      # 폴더 디렉토리 내부 데이터 이미지들의 이름들을 모은 배열을 생성

            for data_name in data_list:                                             # 데이터 배열 순회
                wr.writerow([os.path.join(folder_dir, data_name), folder])          # n-Samples행 2열(data_dir, label)을 지닌 .csv 파일을 생성 


    f.close()                                                                       # csv 파일 닫기


def CalcMeanStdFromCSV(filename):                                                   # 데이터의 평균값과 표준편차값을 구하는 함수
    data_info = pd.read_csv('./' + filename + '.csv', header=None)

    image_arr = np.asarray(data_info.iloc[:, 0])                                    # csv 파일에서 데이터 디렉토리를 모두 획득
    
    aver_mean = 0.0                                                                 # 모든 데이터들의 평균값이 저장되는 결과변수
    aver_std = 0.0                                                                  # 모든 데이터들의 표준편차값이 저장되는 결과변수
    counter = 0                                                                     # 데이터들의 개수를 세는 변수

    for dir in image_arr:

        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)                                 # 디렉토리 경로를 입력받아 영상을 생성한다

        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)   # ndarray를 float 타입의 0과 1사이 값으로 정규화                               
        
        std, mean = cv2.meanStdDev(img)                                             # 영상에서 평균 및 표준편차값을 획득
        aver_std += std[0][0]                                                       # 획득한 편차값을 결과변수에 더한다
        aver_mean += mean[0][0]                                                     # 획득한 평균값을 결과변수에 더한다
        counter += 1                                                                # 카운트 증가
    

    f = open(filename +'_std_mean.csv', 'w', newline='')                            
    wr = csv.writer(f)                                                              # .csv 파일 open
    wr.writerow([aver_std / counter, aver_mean / counter])                          # .csv 파일에 전체 결과값을 데이터 개수로 나누어 평균값과 표준편차값을 저장한다
    f.close()                                                                       # csv 파일 닫기

