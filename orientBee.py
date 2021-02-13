import cv2
import numpy as np
import os
from matplotlib import pyplot as plt



def orientBee(rotatedImage, beeImage):

	#create mask of one half of bee image
	mask = np.zeros(rotatedImage.shape[:2], np.uint8)
	#anything within the bottom half of the image is set to black to calculate histogram of top half
	mask[(rotatedImage.shape[0])/2:rotatedImage.shape[0], 0 : rotatedImage.shape[1]] = 255

	hist = cv2.calcHist([rotatedImage], [0],mask, [10], [0,255])

	#and again for the other half of the image
	mask2 = np.zeros(rotatedImage.shape[:2], np.uint8)
	mask2[0:(rotatedImage.shape[0])/2, 0:rotatedImage.shape[1]] = 255
	hist2 = cv2.calcHist([rotatedImage], [0], mask2, [10], [0,255])


	print('wwwwwwwwwwwwwwwww')

	#Used to find points in histogram
	#plt.plot(hist, label = 'hist1'), plt.plot(hist2, label = 'hist2')

	#print(hist)
	#print(hist2)

	#plt.legend()
	#plt.show()

	beeImage = beeImage.split(".")
	beeImage = beeImage[0]
	newf = beeImage + "cut" + ".jpg"
	#If bucket 2 within the histogram is 0, then it's probably a shadow
	if hist2[2] <= 0:
		print("das a shadow")
		return
	#Checking the two histograms to find the top of the bee- I need to fix up them buckets yo
	if hist2[3] < hist[3]:
		h, w = rotatedImage.shape[:2]
		#find center
		center = (w/2, h/2)
		#rotate 180
		matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
		rotatedImage2 = cv2.warpAffine(rotatedImage, matrix, (w, h))
		#cv2.imshow('bloop', rotatedImage2)
		#cv2.waitKey(0)


		cv2.imwrite(newf, rotatedImage2)

		return
	else:
		cv2.imwrite(newf, rotatedImage)
		return




