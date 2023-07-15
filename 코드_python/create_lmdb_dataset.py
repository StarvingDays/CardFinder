import os
import lmdb
import cv2

import numpy as np
import pandas as pd




def writeCache(env, cache):                                             # lmdb에 데이터를 쓰는 함수
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, outputPath, checkValid=False):
    os.makedirs(outputPath, exist_ok=True)                              # 디렉토리 생성
    env = lmdb.open(outputPath)                                         # lmdb 생성
    cache = {}                                                          # 캐시 배열 생성
    cnt = 1 

    data_info = pd.read_csv(inputPath, header=None)                     # 데이터 및 레이블 csv 읽어오기
    dirs = np.asarray(data_info.iloc[:, 0])                             # 전체 데이터의 경로 획득
    labels = np.asarray(data_info.iloc[:, 1])                           # 전체 데이터의 레이블 획득

    nSamples = len(data_info)                                           # 전체 데이터 개수
    
    for i in range(nSamples):                                           # 전체 데이터 개수 만큼 for문 순회
        imagePath = dirs[i]     
        if not os.path.exists(imagePath):                               # 이미지 경로가 유효하지 않은 경우
            print('%s does not exist' % dirs[i])
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()                                         # bytebuffer 읽어오기
         

        imageKey = 'image-%09d'.encode() % cnt                          # 데이터 key 값
        labelKey = 'label-%09d'.encode() % cnt                          # 레이블 key 값

        cache[imageKey] = imageBin                                      # imageKey값 위치에 bytebuffer를 value 값으로 저장
        cache[labelKey] = str(labels[i]).encode()                       # labelKey값 위치에 labels를 value 값으로 저장
        cnt += 1                                                        # 카운트 증가

    nSamples = cnt-1                                                    # nSamples에 전체 데이터 개수를 저장
    cache['num-samples'.encode()] = str(nSamples).encode()              # 샘플 개수를 바이터 문자열로 저장
    writeCache(env, cache)                                              # lmdb 데이터 셋 저장