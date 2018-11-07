import cv2
import numpy as np

image = cv2.imread('./data/testing/out1.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(image, (9, 9), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 40)
bit = cv2.bitwise_not(thresh)
_, contours, hierarchy = cv2.findContours(bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
0
for i in range(0, len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    digit = image[y:y+h, x:x+w]
    cv2.imwrite("./data/digits/"+str(i)+".png", digit)

cv2.imshow('Final Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()