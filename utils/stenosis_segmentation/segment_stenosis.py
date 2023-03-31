from ultralytics import YOLO
import numpy as np
import cv2

def segment_stenosis(model, images):
    model = YOLO(model)
    res = model.predict(images)[0]
    mat = np.zeros((640,640))
    boxes = res.boxes.xyxy.numpy().astype(int)
    for i in boxes:
        mat = cv2.rectangle(mat, i[:2], i[2:5], (1), cv2.FILLED)
    return mat