import os
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
from custom_layers import PoolHelper, LRN2D


def create_file_list(path, extension='.png'):
    # Useful function
    file_list = []
    print(path)
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(extension):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list


IMG_ROWS, IMG_COLS = 64, 64

LENGTH_MODEL = load_model('./models/length3.h5',
                          custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
SINGLE_MODEL = load_model('./models/single_mix3.h5',
                          custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
DOUBLE_MODEL = load_model(
    './models/double.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})

files = create_file_list("./data/preprocessed/")
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in files:
    # # read the image
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if K.image_data_format() == 'channels_first':
        img = img.reshape(1, 1, IMG_ROWS, IMG_COLS)
    else:
        img = img.reshape(1, IMG_ROWS, IMG_COLS, 1)

    # Standardization training set
    mean_px = img.mean().astype(np.float32)
    std_px = img.std().astype(np.float32)
    img = (img - mean_px)/(std_px)

    length_predicaion = LENGTH_MODEL.predict(img)
    # print("File=%s, Predicted=%s" %
    #       (file, np.argmax(length_predicaion, axis=1)))
    # print(length_predicaion)
    if np.argmax(length_predicaion, axis=1) == 0:
        single_predicaion = SINGLE_MODEL.predict(img)
        print("File=%s, Predicted=%s" %
              (file, np.argmax(single_predicaion, axis=1)))
    else:
        double_predication = DOUBLE_MODEL.predict(img)
        if np.argmax(double_predication, axis=1) < 10:
            print("File=%s, Predicted=%s" %
                  (file, np.argmax(double_predication, axis=1)))
        else:
            print("File=%s, Predicted=%s" %
                  (file, np.argmax(double_predication, axis=1)))
