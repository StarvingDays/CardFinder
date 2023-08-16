from enum import Enum
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import base64
import cv2
import easyocr
import asyncio
import easyocr
from pydantic import BaseModel
import json

app = FastAPI()                                                                                     # FastAPI 인스턴스 생성                                                                                     

class DataInput(BaseModel):
    Checkcard: str

reader = easyocr.Reader(['en'])                                                                     # easyocr 인스턴스 생성

def AnalizeImage(checkcard_data):                                                                   # 이미지 분석 함수
    image_np = cv2.imdecode(                                                                        # base64 포멧 str 디코딩
        np.fromstring(base64.b64decode(checkcard_data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    texts = reader.readtext(image_np, detail=0)                                                     # easyocr 이미지 분석
    
    card_number = ''                                                                                # 체크카드 숫자
    name = ''                                                                                       # 체크카드 영문이름
    valid = ''                                                                                      # 체크카드 유효기한
    
    for i in texts:
        if len(i) == 19:
            card_number = i
        if len(i) == 11:
            name = i
        if i.find('/') != -1:
            valid = i

    results = {                                                                                     # 결과를 json type으로 저장
        'Result': {
            'CardNumber': card_number,
            'Name': name,
            'Valid': valid
        }
    }

    return results

@app.post('/OCR')
async def OCR(image_data: DataInput):                                                               # FastAPI로 호출하는 비동기 함수                            
    try:
        if image_data.Checkcard:                                                                            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, AnalizeImage, image_data.Checkcard)          # base64 포멧 스트링을 파라미터로 받는 AnalizeImage 함수를 실행하여 결과값을 받을 때 까지 대기  
            print(results)
            
            return JSONResponse(results)
        else:
            return JSONResponse({'error': 'No image data received'})

    except Exception as e:
        return JSONResponse({'error': str(e)})

if __name__ == "__main__":
    uvicorn.run("Server:app", host='192.168.0.21', port=2005, reload=True)

