import cv2
import glob
from tqdm import tqdm
import pathlib
import math
import os
import numpy as np


def createFileList(myDir, format='.png'):
    # Useful function
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


def resize(mode):
    if (mode == 'double'):
        for i in range(0, 10):
            myFileList = createFileList(
                "../NIST single/"+str(i)+"/train_"+str(i)+"/")
    else:
        for i in range(0, 100):
            if (i < 10):
                # load the original images from 0-9
                myFileList = createFileList("../NIST double/0"+str(i)+"/")
            else:
                # load the original images from 10-99
                myFileList = createFileList("../NIST double/"+str(i)+"/")

    for img in tqdm(myFileList):
        # read the image
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # rows, cols = image.shape

        # if rows > cols:
        #     factor = 38.0/rows
        #     rows = 28
        #     cols = int(round(cols*factor))
        #     # first cols than rows
        #     image = cv2.resize(image, (cols, rows))
        # else:
        #     factor = 38.0/cols
        #     cols = 28
        #     rows = int(round(rows*factor))
        #     # first cols than rows
        #     image = cv2.resize(image, (cols, rows))

        # colsPadding = (int(math.ceil((64-cols)/2.0)),
        #                int(math.floor((64-cols)/2.0)))
        # rowsPadding = (int(math.ceil((64-rows)/2.0)),
        #                int(math.floor((64-rows)/2.0)))

        # image = np.lib.pad(image, (rowsPadding, colsPadding),
        #                    'constant', constant_values=255)

        image = cv2.resize(image, (64, 64))

        # save the processed images
        cv2.imwrite(str(img), image)


def clear(mode, format):
    if (mode == 'single'):
        for i in range(0, 10):
            # if (i < 10):
            for file in tqdm(glob.glob("../NIST single/"+str(i)+"/train_"+str(i)+"/*"+format + "")):
                path = pathlib.Path(file)
                path.unlink()
    else:
        for i in range(0, 100):
            if (i < 10):
                for file in tqdm(glob.glob("../NIST double/0"+str(i)+"/*"+format + "")):
                    path = pathlib.Path(file)
                    path.unlink()
            else:
                for file in tqdm(glob.glob("../NIST double/"+str(i)+"/*"+format + "")):
                    path = pathlib.Path(file)
                    path.unlink()


# clear(mode='single', format='.png')
# resize(mode='single')
