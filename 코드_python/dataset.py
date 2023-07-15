from torch.utils.data import Dataset
from torch.utils.data import sampler
import lmdb
import sys
import numpy as np
import cv2

class lmdbDataset(Dataset):

    def __init__(self, root=None):
        self.env = lmdb.open(                                                                           # lmdb dataset open
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))                                      # num-samples항목에서 ladb 내부에 저장된 샘플들의 개수를 획득
            self.nSamples = nSamples

  


    def __len__(self):
        return self.nSamples                                                                            # 데이터셋 크기 반환

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index                                                              # lmdb dataset의 data key값 
            imgbuf = txn.get(img_key.encode('utf-8'))                                                   # key값을 utf-8형식으로 인코딩하여 lmdb 내부 value값 가져오기

            img = cv2.imdecode(np.fromstring(imgbuf, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)           # 획득한 bytebuffer를 ndarray로 변환
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)                   # ndarray를 float 타입의 0과 1사이 값으로 정규화 
                
            label_key = 'label-%09d' % index                                                            # lmdb dataset의 label key값
            label = txn.get(label_key.encode('utf-8'))                                                  
            
            label = int(label.decode())                                                                 # byte 문자열을 일반문자열로 decode 한 뒤 int 타입으로 변환


        return (img, label)                                                                             # img와 label을 튜플 타입으로 반환

