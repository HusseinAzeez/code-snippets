from keras.models import Model, load_model
from keras import backend as K
import numpy as np
import cv2
from tqdm import tqdm
import glob
import os
from custom_layers import PoolHelper, LRN2D


def createFileList(myDir, format='.png'):
    # Useful function
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


img_rows, img_cols = 64, 64

length_model = load_model('./models/length2.h5',
                          custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
single_model = load_model('./models/single_mix2.h5',
                          custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
myFileList = createFileList("./data/preprocessed/")
myFileList.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in myFileList:
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

    # length_predicaion = length_model.predict(img)
    # print("File=%s, Predicted=%s" %
    #       (file, np.argmax(length_predicaion, axis=1)))
    # if (np.argmax(length_predicaion, axis=1) == 0):
    single_predicaion = single_model.predict(img)
    print("File=%s, Predicted=%s" %
          (file, np.argmax(single_predicaion, axis=1)))
