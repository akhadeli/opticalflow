# TEST SCRIPT IGNORE

import cv2
import numpy as np
from opticalFlow import difference, threshold, lowpassFilter

if __name__ == "__main__":
    img1 = cv2.imread('armD32im1/2.png')
    img2 = cv2.imread('armD32im1/3.png')

    # img1 = cv2.imread('armD32im1/4.png')
    # img2 = cv2.imread('armD32im1/5.png')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', difference(img1, img2), 30)
    cv2.imshow('frame2', lowpassFilter(difference(img1, img2), 30))
    cv2.waitKey(0)