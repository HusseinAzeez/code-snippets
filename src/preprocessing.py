'''
    Author: Eraser
'''
import glob
import pathlib
import math
import cv2
import numpy as np


class Preprocessing():
    """
        This class convert the pdf into tiff then crop the required
        region segments them and save the digits
    """
    @classmethod
    def clear_images(cls):
        """
            Delete existed images from last PDF
        """
        for file in glob.glob("./data/digits/*.png"):
            path = pathlib.Path(file)
            path.unlink()
        for file in glob.glob("./data/roi/*.png"):
            path = pathlib.Path(file)
            path.unlink()

    @classmethod
    def text_detection(cls, image):
        """
            Find the largest text area on the scanned document
        """
        rgb = cv2.pyrDown(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        grad = cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(
            grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        # using RETR_EXTERNAL instead of RETR_CCOMP
        _, contours, _ = cv2.findContours(
            connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the largest contour by finding the maximum area of the contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (255, 255, 255), 2)
        rect = rgb[y:y+h, x:x+w]

        return rect

    @classmethod
    def deskew_images(cls, image):
        """
            Rotating the skew images
        """
        binary = cv2.bitwise_not(image)

        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv2.threshold(binary, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # grab the (x, y) coordinates of all pixel values that
        # are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all
        # coordinates
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # draw the correction angle on the image so we can validate it
        cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return rotated

    @classmethod
    def sort_contours(cls, cnts, method="left-to-right"):
        """
            Sort the contours
        """
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes

    @classmethod
    def delete_borders(cls, region):
        """
            Delete the table borders by using vertical and horizontal morpholohical operations
        """
        # Thresholding the image
        (_, img_bin) = cv2.threshold(region, 128, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Invert the image
        img_bin = 255-img_bin

        # Defining a kernel length
        kernel_length = np.array(region).shape[1]//40

        # A verticle kernel of (1 X kernel_length),
        # which will detect all the verticle lines from the image.
        verti_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, kernel_length))

        # A horizontal kernel of (kernel_length X 1),
        # which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_length, 1))

        # Morphological operation to detect verticle lines from an image
        img_temp1 = cv2.erode(img_bin, verti_kernel, iterations=3)
        verticle_lines = cv2.dilate(img_temp1, verti_kernel, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        mask = verticle_lines + horizontal_lines
        mask = cv2.bitwise_not(mask)
        img_bin = cv2.bitwise_and(img_bin, img_bin, mask=mask)
        img_bin = cv2.bitwise_not(img_bin)

        return img_bin

    def crop(self, image):
        '''
            Crops the full image into 8 regions of interests by using fixed coordinates
        '''
        y1 = 0
        x1 = 0
        x2 = 412
        h = 0
        # full_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        main_region = self.text_detection(image=image)
        main_region = self.deskew_images(image=main_region)
        main_region = main_region[305:795, 530:945]
        main_region = self.delete_borders(region=main_region)
        cv2.imwrite('./data/roi/main_region.png', main_region)

        for no in range(1, 9):
            roi = main_region[y1 + h:60 + h, x1:x2]
            h += 67
            cv2.imwrite("./data/roi/roi_"+str(no)+".png", roi)

    @classmethod
    def resize_images(cls, image):
        """
            Resize the images into 64x64 pixels images by adding padding to the images
        """
        # read the image
        if image is not None:
                # resize the images and invert it (black background)
            image = cv2.resize(image, (64, 64))

            rows, cols = image.shape

            if rows > cols:
                factor = 38.0/rows
                rows = 28
                cols = int(round(cols*factor))
                # first cols than rows
                image = cv2.resize(image, (cols, rows))
            else:
                factor = 38.0/cols
                cols = 28
                rows = int(round(rows*factor))
                # first cols than rows
                image = cv2.resize(image, (cols, rows))

            cols_padding = (int(math.ceil((64-cols)/2.0)),
                            int(math.floor((64-cols)/2.0)))
            rows_padding = (int(math.ceil((64-rows)/2.0)),
                            int(math.floor((64-rows)/2.0)))

            image = np.lib.pad(image, (rows_padding, cols_padding),
                               'constant', constant_values=255)

        return image

    @classmethod
    def label_contour(cls, image, contours, i):
        """
            Return the image with the contour number drawn on it
        """
        # compute the center of the contour area and draw a circle
        # representing the center
        M = cv2.moments(contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and label number on the image
        cv2.putText(image, "{}".format(i), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1)

        # return the image with the contour number drawn on it
        return image

    def segment(self):
        """
            Segments the 8 regions of interests into digits using OpenCV contour function
        """
        for region in range(1, 9):
            image = cv2.imread("./data/roi/roi_" + str(region) +
                               ".png", cv2.IMREAD_UNCHANGED)
            if image is not None:
                blur = cv2.GaussianBlur(image, (15, 15), 0)
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 12)
                bit = cv2.bitwise_not(thresh)
                _, contours, _ = cv2.findContours(
                    bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Sort contours
                contours, _ = self.sort_contours(
                    contours, method="left-to-right")

                # For debugging uncomment the below codes to see the sorted contours

                # sort = image.copy()
                # for (i, cnt) in enumerate(contours):
                #     if cv2.contourArea(cnt) > 120:
                #         x, y, w, h = cv2.boundingRect(cnt)
                #         if h > 10 and w > 10:
                #             sort = label_contour(sort, cnt, i)
                #             print(i)
                # cv2.imshow('Sorted Image', sort)

                # Draw a rectangle for each contour
                for (i, cnt) in enumerate(contours):
                    if cv2.contourArea(cnt) > 120:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if h > 13 and w > 13:
                            # For debugging uncomment the below line to see all the contours

                            # cv2.rectangle(image, (x, y), (x+w, y+h),
                            #               (0, 0, 0), 1)
                            # cv2.imshow('Segmented Image', image)
                            digit = image[y:y+h, x:x+w]
                            digit = self.resize_images(digit)
                            cv2.imwrite("./data/digits/" +
                                        str(region) + "."+str(i)+".png", digit)


# For debugging uncomment the below code and run the file


# if __name__ == '__main__':
#     preproessing = Preprocessing()
#     preproessing.clear_images()
#     preproessing.crop(image='../raw/Full/full_54.png')
#     preproessing.segment()
