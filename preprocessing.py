'''
    Author: Eraser
'''
import glob
import pathlib
import math
import cv2
import numpy as np
from scipy import ndimage
# from skimage import morphology


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                         key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, bounding_boxes)


def clear():
    # Delete existed images from last PDF
    for file in glob.glob("./data/digits/*.png"):
        path = pathlib.Path(file)
        path.unlink()
    for file in glob.glob("./data/roi/*.png"):
        path = pathlib.Path(file)
        path.unlink()
    for file in glob.glob("./data/preprocessed/*.png"):
        path = pathlib.Path(file)
        path.unlink()


'''
Crops the full image into 8 regions of interests by using fixed coordinates
'''


def crop(img):
    y1 = 0
    y2 = 977
    x1 = 0
    x2 = 900
    h, w = (0, 0)
    main_region = img[720:1690, 1080:1900]
    cv2.imwrite("./data/roi/main_region.png", main_region)
    # date_region = img[720:1677, 135:385]
    for no in range(1, 9):
        roi = main_region[y1 + h:120 + h, x1:x2]
        h += 124
        cv2.imwrite("./data/roi/roi_"+str(no)+".png", roi)


def segment():
    # Segments the 8 regions of interests into digits using OpenCV contour function
    for region in range(1, 9):
        image = cv2.imread("./data/roi/roi_" + str(region) +
                           ".png", cv2.IMREAD_UNCHANGED)
        if (image is not None):
            blur = cv2.GaussianBlur(image, (15, 15), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 15)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 6))
            # morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow('Morph', morph_img)
            bit = cv2.bitwise_not(thresh)
            _, contours, _ = cv2.findContours(
                bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            # sort contours
            contours, bounding_boxes = sort_contours(
                contours, method="left-to-right")
            # Draw a rectangle for each contour
            for i in range(0, len(contours)):
                cnt = contours[i]
                if (cv2.contourArea(cnt) > 120):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if (h > 20 and w > 20):
                        # cv2.rectangle(image, (x, y), (x+w, y+h),
                        #               (0, 0, 255), 1)
                        cv2.imshow('image', image)
                        digit = image[y:y+h, x:x+w]
                        cv2.imwrite("./data/digits/" +
                                    str(region) + "_"+str(i)+".png", digit)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
        # read the image
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if image is not None:
            # resize the images and invert it (black background)
            image = cv2.resize(image, (64, 64))

            rows, cols = image.shape

            if rows > cols:
                factor = 38.0/rows
                rows = 28
                cols = int(round(cols*factor))
                # first cols than rows
                image = cv2.resize(image, (cols, rows))
            else:
                factor = 38.0/cols
                cols = 28
                rows = int(round(rows*factor))
                # first cols than rows
                image = cv2.resize(image, (cols, rows))

            cols_padding = (int(math.ceil((64-cols)/2.0)),
                            int(math.floor((64-cols)/2.0)))
            rows_padding = (int(math.ceil((64-rows)/2.0)),
                            int(math.floor((64-rows)/2.0)))

            image = np.lib.pad(image, (rows_padding, cols_padding),
                               'constant', constant_values=255)
            # shiftx, shifty = getBestShift(image)
            # shifted = shift(image, shiftx, shifty)
            # image = shifted

            # save the processed images
            cv2.imwrite("./data/preprocessed/62_"+str(i)+".png", image)

        i += 1


FULL = cv2.imread('../raw/Full/full_62.png', cv2.IMREAD_GRAYSCALE)
clear()
crop(FULL)
segment()
resize()
