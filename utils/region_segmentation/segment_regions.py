from ultralytics import YOLO


def segment_regions(model, images):
    model = YOLO(model)
    pred = model.predict(images)[0]
    return pred, model.names