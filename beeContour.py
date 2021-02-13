import cv2
import numpy as np
import sys


def beeContour(cannyImg, fullBeeImg, beeroi):

	#couldbe useful in determination later
	beeroigray = cv2.cvtColor(beeroi, cv2.COLOR_BGR2GRAY)
	beeImagegray = cv2.cvtColor(fullBeeImg, cv2.COLOR_BGR2GRAY)

	#Makes copy of canny image and beeroi image
	contourFinder = cannyImg
	beeroiCopy = np.copy(beeroi)

	#Finds all contours within the canny image, no approximation because of possible hierarchy issues with a moving tracked object
	blah, contours, hierarchy = cv2.findContours(contourFinder, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#Only grabs the 3 largest contours, since most bees will stand out from the background as the largest object in the region.
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]

	#Initializes different variables, if it remains None, it will throw an error in beeSight
	beeBox = None
	beeRect= None
	beeApprox = None
	beeEllipse = None

	#Iterating through the three largest contours to see if issa bee
	for i,c in enumerate(contours):

		#cv2.drawContours(beeroiCopy, contours, i, (255,255,255), 2)

		#contour approximation- epsilon values depend on how approximated you want the curve to be
		#ex- 10% will make a fair estimation, while 1% will form to the contour.
		epsilon = 0.10*cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)

		#tries to fit an eclipse, but sometimes an eclipse will not work with the image. Cant remember the error I get but it's odd. Not too sure of the reasoning.
		try:
			ellipse = cv2.fitEllipse(c)
		except:
			print('contours wobbly')
			continue

		#draws the ellipse on the beeroi copy
		cv2.ellipse(beeroiCopy, ellipse,(150,150,150), 2)

		#Makes a min area rectangle out of the contour to check lengths
		rect = cv2.minAreaRect(c)
		print('rect:')
		print(rect)
		width = rect[1][0]
		height = rect[1][1]
		print(width)
		print(height)

		#Makes box points based on the rectangle and position on the image-used to check if rectangle bounds are outside of the image
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		#draws the box on the copy- kinda useless but could be good for visualization
		cv2.drawContours(beeroiCopy, [box], 0, (0,255,255), 2)

		#cv2.imshow('approx', beeroiCopy)
		#cv2.waitKey(0)

		#Checking rectangle values that would correspond to ***roughly*** what a bee shape would be
		if width > 20 and width < 80 and height > 20 and height < 80:
			#go through each box coordinate to make sure they are within the scope of the image
			for i in box:
				if i[0] > 320 or i[1] > 240:
					print("Not a full bee in image")
					#returns the None-typed objects
					return beeBox, beeRect, beeApprox, beeEllipse
			#If contour requirements and box bounding requirements are met, move on to the next step
			beeBox = box
			beeRect = rect
			beeApprox = approx
			beeEllipse = ellipse
			return beeBox, beeRect, beeApprox, beeEllipse
		else:
			print("No bees found for contour")
	return beeBox, beeRect, beeApprox, beeEllipse


