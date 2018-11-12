import glob
import cv2

full = cv2.imread('./data/raw/full-image.tiff', cv2.IMREAD_GRAYSCALE)
main_region = full[700:1677, 1060:1880]
roi_1 = main_region[5:110, 0:818]
roi_2 = main_region[110:240, 0:818]
roi_3 = main_region[240:360, 0:818]
roi_4 = main_region[360:480, 0:818]
roi_5 = main_region[480:560, 0:818]
roi_6 = main_region[560:660, 0:818]
roi_7 = main_region[660:760, 0:818]
roi_8 = main_region[760:860, 0:818]

# for no in range(1,9):
cv2.imwrite("./data/roi/roi_1.png", roi_1)
cv2.imwrite("./data/roi/roi_2.png", roi_2)
cv2.imshow('Main Region', main_region)
cv2.imshow('ROI 1', roi_1)
cv2.imshow('ROI 2', roi_2)
# cv2.imshow('ROI 3', roi_3)
# cv2.imshow('ROI 4', roi_4)
# cv2.imshow('ROI 5', roi_5)
# cv2.imshow('ROI 6', roi_6)
# cv2.imshow('ROI 7', roi_7)
# cv2.imshow('ROI 8', roi_8)
cv2.waitKey(0)
cv2.destroyAllWindows()