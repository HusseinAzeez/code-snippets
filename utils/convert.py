from PIL import Image
import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd


def rename(dir=None):
    i = 24480
    for file in glob.glob(dir):

        # Read the image
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # Save images
        cv2.imwrite("../Indy data/5/train_35_"+str(i)+".png", img)
        i += 1


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


def convert():
    for i in range(0, 10):
        # if (i < 10):
        #     # load the original images from 0-9
        #     myFileList = createFileList("../NIST double/0"+str(i)+"/")
        # else:
        #     # load the original images from 10-99
        #     myFileList = createFileList("../NIST double/"+str(i)+"/")

        myFileList = createFileList(
            "../NIST single/"+str(i)+"/train_"+str(i)+"/")
        # Sorted files are easier to label
        myFileList.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for file in tqdm(myFileList):
            with open("full_single_mix_files.txt", "a") as txt:
                txt.write(file + "\n")

            # Open the image
            img = Image.open(file)

            # Store the image as numpy array
            value = np.asarray(img.getdata(), dtype=np.int).reshape(
                (img.size[1], img.size[0]))

            # Flatten image into one row
            value = value.flatten()

            # Add the label
            value = np.hstack([i, value])

            # Write the images to csv
            with open("full_single_mix.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)

        print('Current file values:', value)


convert()
