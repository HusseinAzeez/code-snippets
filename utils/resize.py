import cv2
import glob
from tqdm import tqdm
import pathlib


def resize():
    for i in range(00, 100):
        for img in tqdm(glob.glob("../NIST double/99/*.png")):

            # read the image
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

            if (image is not None):

                # resize the images and invert it (black background)
                image = cv2.resize(
                    image, (64, 64))

                # save the processed images
                cv2.imwrite(str(img), image)


def clear():
    for i in range(00, 100):
        for file in tqdm(glob.glob("../NIST double/"+str(i)+"/*.txt")):
            path = pathlib.Path(file)
            path.unlink()


# clear()
# resize()
