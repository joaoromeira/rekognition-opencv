from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import sys
from base64 import b64decode, b64encode
import numpy as np

cvClassifier = cv.CascadeClassifier(
    'opencv/haarcascade/haarcascade_frontalface_alt.xml')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Access-Control-Allow-Headers",
        "Origin",
        "Accept",
        "X-Requested-With",
        "Content-Type",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods"
        "Authorization",
        "X-Amz-Date",
        "X-Api-Key",
        "X-Amz-Security-Token"
    ]
)


@app.get("/")
def index():
    return {"API Version V1.0.0"}


@app.post("/rekognition/faces")
def face_rekognition(base64: str):
    try:
        type, base = base64.split(',')

        img_bytes = b64decode(base, validate=True)
        img_unit = np.fromstring(img_bytes, np.uint8)
        img_np = cv.imdecode(img_unit, cv.IMREAD_COLOR)

        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)

        detections = cvClassifier.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=6)

        for(x, y, w, h) in detections:
            cv.rectangle(img_np, (x, y), (x+w, y+h), (0, 0, 255), 4)

        retval, buffer = cv.imencode('.jpg', img_np)
        pic_str = b64encode(buffer)
        pic_str = pic_str.decode()

        return jsonify({
            "base64": type+','+pic_str
        })
    except:
        print("Unexpected error:", sys.exc_info()[0])

        return {"message": "Error"}, 500
