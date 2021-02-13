# First step in processing a specific region- producing the filter of the specific region to use.
# Taking the region, I process it to filter out all background noise aside from the bee.
import cv2
import numpy as np
from PIL import Image, ImageOps
import scipy
import sys

class beeFilter:
    # takes the region of the image, blurs it, adds a filter for sharpness, and thresholds this from the region and also the main picture
    def beeFilter(beeImage, beeroi):
        # separate grayscale version of the bee region
        beeroigray = cv2.cvtColor(beeroi, cv2.COLOR_BGR2GRAY)
        beeImagegray = cv2.cvtColor(beeImage, cv2.COLOR_BGR2GRAY)
        # checks to see if the bee image was loaded- checks height
        if beeroi.shape[0] is 0:
            return 0

        # commented original filter- 5x5 kernel that blurred 35 pixels around the main pixel
        # blur = cv2.bilateralFilter(beeroigray, 5, 35, 35)

        # new blur- median 3x3 kernel blur that I used to try to get shadows out, it's okay.
        # bilateral filter to spread the colors out more, tryna get them shadows out, u feel? Anyways it doesn't work well, but it retains bee shape.
        blur = cv2.medianBlur(beeroigray, 3)
        blur2 = cv2.bilateralFilter(blur, 9, 35, 35)
        blur3 = cv2.bilateralFilter(blur2, 9, 35, 35)


        # threshold using super blurred image- range of 0-140, where any pixel strictly above 140 is turned to 0, unless it is in a line where most pixels are under 140.
        # Using binary thresh with Otsu to get the best edge detection possible.
        ret, thresh = cv2.threshold(blur3, 0, 140, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Canny edge detection on the thresholded image, Goes through the regions where the color is 255 instead of 0.
        threshcanny = cv2.Canny(thresh, 20, 60)



        morphKernel = np.array(
            [[0,1,0],
             [1,1,1],
             [0,1,0]], np.uint8)

        # dilate the canny img for better contour detection later
        # I used this kernel for a more x/y axis dilation instead of dilating at the diagonals.
        threshcanny = cv2.dilate(threshcanny,morphKernel, iterations=1)

        # adding new check for a complete threshold by adding 1 pixel black border around image, just to make sure no half bees are captured
        threshcanny = cv2.copyMakeBorder(threshcanny, 5,5,5,5, cv2.BORDER_CONSTANT, value=[0,0,0])



        return threshcanny

