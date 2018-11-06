import cv2
from PIL import Image, ImageEnhance
import pytesseract as tes
import numpy as np
from skimage import io

image = cv2.imread('./pdf/19.tiff')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
bit = cv2.bitwise_not(thresh)
_, contours, hierarchy = cv2.findContours(bit, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
cv2.contourArea
for i in range(0, len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(bit, (x, y), (x+w, y+h), (255, 255, 255), 1)
    digit = bit[y:y+h, x:x+w]
    # cv2.imwrite("./data/digits/"+str(i)+".png", digit)
  
cv2.imshow('Contour', bit)
cv2.waitKey(0)
cv2.destroyAllWindows()