import os
import glob
import cv2
import numpy as np
import math
from scipy import ndimage
from PIL import Image

for img in glob.glob("./data/digits/*.png"):
    
    
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if (image is not None):
        w = 260
        h = 80
        x = 0
        y = 0
        time_begin = image[0:0+h, 0:0+w]
        cv2.imshow("cropped", time_begin)
        w += 200
        h += 80
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()