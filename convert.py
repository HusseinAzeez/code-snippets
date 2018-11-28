from PIL import Image
import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd
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
myFileList = createFileList('../NIST double/00/')
myFileList.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in tqdm(myFileList):
    with open("full_double_files.txt", "a") as txt:
        txt.write(file + "\n")

    img = Image.open(file)

    # get original image parameters...
    width, height = img.size
    format = img.format
    mode = img.mode

    # Save Greyscale values
    value = np.asarray(img.getdata(), dtype=np.int).reshape(
        (img.size[1], img.size[0]))
    value = value.flatten()
    value[0] = 00

    with open("full_double.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

print(value.shape)
print(value)
