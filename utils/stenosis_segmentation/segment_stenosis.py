from ultralytics import YOLO
import numpy as np
import cv2

def segment_stenosis(model, images):
    model = YOLO(model)
    res = model.predict(images)[0]
    cls = res.boxes.cls.numpy()
    mat = np.zeros((640,640))
    boxes = res.boxes.xyxy.numpy().astype(int)
    rects = []
    for i in boxes:
        rects.append(cv2.rectangle(np.zeros((640,640)), i[:2], i[2:5], (1), cv2.FILLED))
        mat = cv2.rectangle(mat, i[:2], i[2:5], (1), cv2.FILLED)
    return list(zip([model.names[cl] for cl in cls], rects, res.boxes.conf.numpy())), mat