"""
    Autour: Eraser (ตะวัน)
"""
# Standard library imports
import os
import re

# Third-party imports
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2

# Local imports
from custom_layers import PoolHelper, LRN2D


def natural_sort(s):
    """
        Sort the given list in the way that humans expect.
    """
    numbers = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(numbers, s)]


def create_file_list(path, extension='.png'):
    # Useful function
    file_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(extension):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list


def predict(path):
    """
        Prediction method using the pretrained models (length, single digit, double digit)
    """
    img_rows, img_cols = 64, 64

    length_model = load_model(
        './models/length_mix.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    single_model = load_model(
        './models/single_mix.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    double_model = load_model(
        './models/double_mix.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})

    files = create_file_list(path)
    files.sort(key=natural_sort)

    for file in files:
        # read the image
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        if K.image_data_format() == 'channels_first':
            img = img.reshape(1, 1, img_rows, img_cols)
        else:
            img = img.reshape(1, img_rows, img_cols, 1)

        # Standardization the image
        mean_px = img.mean().astype(np.float32)
        std_px = img.std().astype(np.float32)
        img = (img - mean_px)/(std_px)

        length_predicaion = length_model.predict(img)

        if np.argmax(length_predicaion, axis=1) == 0:
            single_predicaion = single_model.predict(img)
            print("File={} => Predicted={}".format(
                file, np.argmax(single_predicaion, axis=1)))
        else:
            double_predication = double_model.predict(img)
            if np.argmax(double_predication, axis=1) < 10:
                print("File=%s => Predicted=%.02d" %
                      (file, np.argmax(double_predication, axis=1)))
            else:
                print("File={} => Predicted={}".format(
                    file, np.argmax(double_predication, axis=1)))


if __name__ == '__main__':
    predict('./data/digits')
