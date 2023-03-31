import numpy as np   
from keras.models import load_model

def extract_frame(source:np.array, model, n_best=3)->np.ndarray:
    model = load_model(model)
    pred = model.predict(source/255)
    res = pred.argmax(axis=3).sum(axis=1).sum(axis=1)
    res = source[np.array(res).argpartition(-n_best, axis=0)[-n_best:]]
    return res, np.array(res).argpartition(-n_best, axis=0)[-n_best:]


