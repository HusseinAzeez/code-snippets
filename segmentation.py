# # import the necessary packages
# from imutils import contours
# import numpy as np
# import argparse
# import imutils
# import cv2

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-r", "--reference", required=True,
# 	help="path to reference OCR-A image")
# args = vars(ap.parse_args())

# # define a dictionary that maps the first digit of a credit card
# # number to the credit card type
# FIRST_NUMBER = {
# 	"3": "American Express",
# 	"4": "Visa",
# 	"5": "MasterCard",
# 	"6": "Discover Card"
# }

# # load the reference OCR-A image from disk, convert it to grayscale,
# # and threshold it, such that the digits appear as *white* on a
# # *black* background
# # and invert it, such that the digits appear as *white* on a *black*
# ref = cv2.imread(args["reference"])
# ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
import cv2
from PIL import Image
import pytesseract as tes
import numpy as np

import cv2
import numpy as np
from skimage import io

img = cv2.imread('./pdf/19.tiff')
cv2.imshow('Original', img)
imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow('Gray', imgray)

ret,thresh = cv2.threshold(imgray, 120,255,cv2.THRESH_BINARY)
cv2.imshow('Threshold', thresh)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
canvs = np.zeros_like(img)
c_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
c_img2 = cv2.drawContours(canvs, contours, -1, (0, 255, 0), 1)
cv2.imshow('Contour', c_img)
cv2.imshow('Contour outline', c_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,75,10)
img = cv2.bitwise_not(img)
cv2.imwrite('./data/out.jpg', img)