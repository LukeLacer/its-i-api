from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
import cv2
import base64


class ImageData(BaseModel):
    imageData: str


app = FastAPI()


origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    return {"message": "The api is running"}


@app.post("/itsi")
def guessifitsi(data: ImageData):
    img = decode_img(data)
    size = 28
    img = img_to_bw(
        resize_img(img, size),
        size
    )
    return {"message": str(predict_img(img)[0])}


def img_to_bw(img, size):
    return np.reshape(cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1], size * size).reshape(1, -1)


def resize_img(img, size):
    return cv2.resize(
        img,
        (size, size)
    )


def predict_img(img):
    model = load(pathlib.Path('svm-its-i.joblib'))
    return model.predict(img)


def decode_img(msg):
    b64_img = msg.imageData.split(',')[1]
    bytes_img = base64.b64decode(b64_img)
    np_img = np.frombuffer(bytes_img, dtype=np.uint8)
    img = cv2.imdecode(np_img, flags=cv2.IMREAD_GRAYSCALE)
    return img
