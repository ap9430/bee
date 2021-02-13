#main file for sighting the bees, gathers a region of image and then sends what it needs to its helper methods.

import cv2
import sys
import os
import subprocess
import datetime
import numpy
import argparse
from beeFilter import beeFilter
from beeContour import beeContour
from extractBee import extractBee
from orientBee import orientBee



#Used when it was a command line tool, adds the ability to select an image name
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the input image")
args = vars(ap.parse_args())
#beeImage = args["image"]

#arbitrary bee image that is replaced by video
beeImage = "testImg1.jpg"

#check to see if its a video or not- it is
videoYN = True
#w/e video name I picked earlier on, should probably make an input check for it
vidName = "12-20-46.h264"

#Main function that defines the steps needed to take bees out of the region
def beeSight(beeImage):
	#read in image
	img = cv2.imread(beeImage)

	#240- half of the 480 from the 640x480 video scene we typically use in the bee videos
	startPointHeight = 240
	# 320- half of the 640 from the 640x480 video scene we typically use in the bee videos
	startPointWidth = 320

	#Using the start points, take a region that encompasses this middle point of the image. Takes 120-360pixels height wise and 160-480pixels width wise region from the base image
	imgRoi = img[int(startPointHeight) - (int(startPointHeight)/2): int(startPointHeight) + (int(startPointHeight)/2), int(startPointWidth) - (int(startPointWidth)/2) : int(startPointWidth) + (int(startPointWidth)/2)]
	print(imgRoi.shape[0])

	#calls the filter function with the region and the image
	cannyImg = beeFilter.beeFilter(img, imgRoi)
	#Checking if a canny image was produced or if there was no image in the first place
	if cannyImg is 0:
		print('New Video')
		return

	#Call to figure out if the contours found in the region add up to a bee
	box, rectangle, approximateFit, ellipseFit = beeContour(cannyImg, img, imgRoi)
	#If no contours were determined to be a bee, then return
	if rectangle is None:
		print("no bees found in image")
		return

	#Extract the bee out of the region
	rotatedImage = extractBee(imgRoi, ellipseFit, rectangle)
	#Orient the bee based on the color patterns to provide a uniform layout
	orientBee(rotatedImage, str(beeImage))

#Splice- only used for videos, splice takes the frames out of the video file to process separately
def splice(vidName, spliceNum):

	#Checking if user has inputted a number
	if type(spliceNum) is int:
		print(vidName)
		vid = vidName + '.mp4'
		#makes directory to store spliced images and their cutouts
		if not os.path.exists(vidName):
			os.makedirs(vidName)
		#converting splices to hour,minute,second format to add to the ffmpeg calls

		startTime = datetime.timedelta(0,5)


		#removes all files within the current directory that are images- used if the same folder has been ran before
		[os.remove(f) for f in os.listdir('.') if f.startswith('image')]
		#ffmpeg call to cut the video into splices based on spliceNum.
		#-y is to auto confirm the call without a prompt reappearing, -ss is the start time, -f is the format, fps is basically saying 1 image every (splicenum) seconds
		ff = "ffmpeg -y -i " + vid + " -s 640x480 -ss " + str(startTime) + " -f image2 -vf fps=1/" + str(spliceNum) +" -hide_banner -q:v 2 " + vidName + "/image%d.jpg"
		subprocess.call([ff], shell=True)
		os.chdir(vidName)
		#runs beeSight for every image created
		[beeSight(f) for f in os.listdir('.') if f.startswith('image')]



# Converts h264 video to mp4 video type for timestamping and getting how many splices you want
def vidtoMp4(vidName):

	name, extension = vidName.split('.')
	print(name)
	print(extension)
	print(os.getcwd())
	#converts h264 to mp4 b/c no timestamps in h264
	command = "ffmpeg -y -i {0}.h264 -c copy {1}.mp4".format(name, name)
	if extension == "h264":
		subprocess.call([command], shell=True)
	elif extension != "mp4":
		sys.exit("no useable video available")
	#how many seconds to split each image by, could change to an input statement but w/e
	spliceNum = 4
	splice(name, spliceNum)

#Where it all begins, due to my poor code formatting
if __name__ == '__main__':
	# ext check to make sure you arent making a file in the wrong place
	ext = 1
	#It will probably always be video, but in the one in a million chance it isn't
	if videoYN:
		#Arbitrary folder name that I downloaded videos into. I got some ftp code in my entropy repo for downloading folders from the server
		#link to ftp : https://github.com/brothertonjd/pyentropy/blob/master/pyentropy/entropy.py
		#Lines 16-27 provide connection, with auth.txt containing the host, port#, username, and password on separate lines
		os.chdir('rpi11b2018-04-30')
		#temporary change to iterate through all videos in a folder
		for f in os.listdir('.'):
			if f.endswith('.h264'):
				print(f)
				#checking if first file, backs up a directory to stop recursive folders
				if ext == 1:
					ext = 0
				else:
					os.chdir('..')
				vidtoMp4(f)
	else:
		beeSight(beeImage)