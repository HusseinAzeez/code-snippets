from PIL import Image
import cv2
import numpy as np
import math
from scipy import ndimage


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


# create an array where we can store our 10 pictures
images = np.zeros((10, 784))
# and the correct values
correct_vals = np.zeros((10, 10))

# we want to test our images which you saw at the top of this page
i = 0
for no in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # read the image
    image = cv2.imread("./data/digits/"+str(no)+".png", cv2.IMREAD_UNCHANGED)
    print(image)

    # resize the images and invert it (black background)
    image = cv2.resize(255 - image, (28, 28))
    print(image.shape)
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
    cv2.imwrite("./data/preprocessed/"+str(no)+".png", image)
    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = image.flatten() / 255.0
    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    images[i] = flatten
    correct_val = np.zeros((10))
    correct_val[no] = 1
    correct_vals[i] = correct_val
    i += 1
