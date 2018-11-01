import cv2 as cv
from PIL import Image
im1 = Image.open("./data/5.jpg")
im2 = im1.point(lambda p: p * 0.9)
# brings up the modified image in a viewer, simply saves the image as
# a bitmap to a temporary file and calls viewer associated with .bmp
# make certain you have associated an image viewer with this file type
im2.show()
im2.save("./data/4.jpg")
