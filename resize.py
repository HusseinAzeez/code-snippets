import cv2
import glob
from tqdm import tqdm


def resize():
    for img in tqdm(glob.glob("../NIST single/9/train_9/*.png")):

        # read the image
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        if (image is not None):

            # resize the images and invert it (black background)
            image = cv2.resize(
                image, (64, 64))

            # save the processed images
            cv2.imwrite(str(img), image)


resize()
