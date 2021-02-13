import cv2
import numpy as np

#Rotate the bee based on ellipse values- needs to be edited cuz dem shadows
#rotates it and take the image out of the region
def extractBee(imgroi, ellipseFit, rectangle):
	#gets srayscale of the image and the shape- use grayscale because there isn't a dimension entry that you need to worry about
	gray = cv2.cvtColor(imgroi, cv2.COLOR_BGR2GRAY)
	shape = gray.shape
	shape = list(shape)
	#rotates the shape values around to be used in the warping
	uno = shape[0]
	dos = shape[1]
	shape[1] = uno
	shape[0] = dos

	shape = tuple(shape)

	#gets the size of the bee's width and height
	contourSizes = rectangle[1]

	#Flips them depending on the larger of the two, produces a uniform shape I promise
	if contourSizes[0] > contourSizes[1]:
		width = contourSizes[0]
		height = contourSizes[1]
	else:
		width = contourSizes[1]
		height = contourSizes[0]

	#Gets the center of the ellipse because the rectangle isn't good enough? Either or would work, ellipse seems to work better.
	center = ellipseFit[0]

	#Angle of rotation from the ellipse fit as well, if using a rectangular center, use rectangle[2]
	theta = ellipseFit[2]

	#Gets a rotation matrix for the pixture to base it's rotation off of using the center of the contour and the angle offset
	matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
	#Warps the image based on the rotation matrix and the shape of the image
	rotatedImage = cv2.warpAffine(src=imgroi, M=matrix, dsize = shape)

	#Gets the center of contour the bee resides at
	x = int(center[1]) - int(width / 2)
	y = int(center[0]) - int(height / 2)

	#Expands the rotated image by 10 pixels, just made so it can get the bee with shadow messing it up
	rotatedImage2 = rotatedImage[x - 10:x + int(width) + 10, y - 10:y + int(height) + 10]

	return rotatedImage2