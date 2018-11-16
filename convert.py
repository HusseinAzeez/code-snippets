from PIL import Image
import numpy as np
import sys
import os
import csv
import pandas

# Useful function
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('./data/preprocessed/')

for file in myFileList:
    print(file)
    img = Image.open(file)

    # get original image parameters...
    width, height = img.size
    format = img.format
    mode = img.mode

    # Save Greyscale values
    value = np.asarray(img.getdata(), dtype=np.int).reshape(
        (img.size[1], img.size[0]))
    value = value.flatten()
    with open("training.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)