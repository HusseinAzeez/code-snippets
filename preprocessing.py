import os, glob
import cv2
import numpy as np
import math
from scipy import ndimage
from PIL import Image

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


# # create an array where we can store our 10 pictures
# images = np.zeros((10, 784))
# # and the correct values
# correct_vals = np.zeros((10, 10))

# i = 0

folder = './data/digits'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        image.append(img)
# for no in [0,1,2,4,5,6,7,9,10,11,12,14,15,25,26,27,28,29,30,31,35,36,37,38,39,40,41,43,44,45]:
    # # read the image
    # image = cv2.imread("./data/digits/"+str(no)+".png", cv2.IMREAD_GRAYSCALE)

    # resize the images and invert it (black background)
    image = cv2.resize(255 - image, (28, 28))

    while np.sum(image[0]) == 0:
        image = image[1:]

    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)

    while np.sum(image[-1]) == 0:
        image = image[:-1]

    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)

    rows, cols = image.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # first cols than rows
        image = cv2.resize(image, (cols, rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # first cols than rows
        image = cv2.resize(image, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),
                   int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),
                   int(math.floor((28-rows)/2.0)))
    image = np.lib.pad(image, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(image)
    shifted = shift(image, shiftx, shifty)
    image = shifted

    # save the processed images
    cv2.imwrite("./data/preprocessed/"+str(file)+".png", image)

    # flatten = image.flatten() / 255.0
    # images[i] = flatten
    # correct_val = np.zeros((10))
    # correct_val[no] = 1
    # correct_vals[i] = correct_val
    # i += 1
