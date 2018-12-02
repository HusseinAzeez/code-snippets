import keras
from keras.models import Model, load_model
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
import glob
from custom_layers import PoolHelper, LRN2D

img_rows, img_cols = 64, 64

model = load_model('./models/single.h5',
                   custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})

for file in glob.glob("./data/preprocessed/*.png"):
    # # read the image
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if K.image_data_format() == 'channels_first':
        img = img.reshape(1, 1, img_rows, img_cols)
    else:
        img = img.reshape(1, img_rows, img_cols, 1)

    # Standardization training set
    mean_px = img.mean().astype(np.float32)
    std_px = img.std().astype(np.float32)
    img = (img - mean_px)/(std_px)

    predicaion = model.predict(img)
    # print(predicaion)
    print("File=%s, Predicted=%s" % (file, np.argmax(predicaion, axis=1)))
