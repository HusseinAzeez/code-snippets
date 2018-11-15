'''
    Autor: Eraser
'''
import cv2
import numpy as np
import glob
import pathlib
import math
from scipy import ndimage

# Delete existed images from last PDF


def clear():
    for file in glob.glob("./data/digits/*.png"):
        path = pathlib.Path(file)
        path.unlink()
    for file in glob.glob("./data/roi/*.png"):
        path = pathlib.Path(file)
        path.unlink()

# Crops the full image into 8 regions of interests by using fixed coordinates


def crop(img):
    y1 = 5
    y2 = 977
    x1 = 0
    x2 = 820
    h, w = (0, 0)
    main_region = img[700:1677, 1060:1880]

    for no in range(1, 9):
        roi = main_region[y1 + h:120 + h, x1:x2]
        h += 120
        cv2.imwrite("./data/roi/roi_"+str(no)+".png", roi)

# Segments the 8 regions of interests into digits using OpenCV contour function


def segment():
    for region in range(1, 9):
        # read the image
        image = cv2.imread("./data/roi/roi_" + str(region) +
                           ".png", cv2.IMREAD_UNCHANGED)
        if (image is not None):
            blur = cv2.GaussianBlur(image, (15, 15), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 13)
            bit = cv2.bitwise_not(thresh)
            _, contours, hierarchy = cv2.findContours(
                bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            # Draw a rectangle for each contour
            for i in range(0, len(contours)):
                cnt = contours[i]
                if (cv2.contourArea(cnt) > 120):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if (h > 20 and w > 20):
                        cv2.rectangle(image, (x, y), (x+w, y+h),
                                      (255, 255, 255), 1)
                        digit = image[y:y+h, x:x+w]
                        # cv2.imshow('Regions of Interests', image)
                        cv2.imwrite("./data/digits/" +
                                    str(region) + "_"+str(i)+".png", digit)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


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

def resize():
    i = 0
    for img in glob.glob("./data/digits/*.png"):
        # # read the image
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if (image is not None):
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
            cv2.imwrite("./data/preprocessed/"+str(i)+".png", image)

        i += 1


full = cv2.imread('../raw/full10.tiff', cv2.IMREAD_GRAYSCALE)
crop(full)
segment()
resize()
# clear()
