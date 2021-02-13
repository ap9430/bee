import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage import morphology as mp
from skimage import img_as_bool
import matplotlib.pyplot as plt
import matplotlib.image as im
import scipy.interpolate as spi
import math

__author__ = 'aaron'


def main():
    # INPUT AND OUTPUT FILEPATHS
    in_path = 'images/input/lowrez/'
    in_path = 'images/input/mt_bees/'
    in_path = 'bees/12-00-32/'
    in_path = 'C:/Users/parkerat2/Desktop/images/'
    out_path = 'images/output/prepro/'

    # TESTING IMAGES
    # some sample images chosen for testing_label/tweaking the file's functionality
    # in_img = "thresholdedframe525_14-12-57_1.png"
    # in_img = "thresholdedframe183_14-12-57.png"
    # in_img = "thresholdedframe750_14-12-57_2.png"
    # in_img = "thresholdedframe750_14-12-57_1.png"
    # in_img = "thresholdedframe225_14-12-57_1.png"
    in_img = 'thresholdedframe1175_14-12-57_2.png'
    in_img = 'Rotated11/11-20-36-1a.jpg'
    in_img = 'image2cut.jpg'
    in_img = '1440_10-00-16.h264.png'
    # in_img = 'mt_bee1.png'
    img = cv2.imread(in_path + in_img)
    cv2.imshow('image', img)
    # this is for those manually cut/rotated images that had black backgrounds
    # it will turn a value below 8 into 255, so black becomes white
    # this is important because the initial thought of this program would that it is easier to determine a white background
    # from the bees, so the goal is to effectively "find the background" and what is left out is the bee.
    # make sure to comment this line if your bees have a white background
    # retval, img = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY_INV)
    # do not mess with anything you see as .copy()
    # object = object assignments are just two variables referring to the same object, so copy is used to create new obj
    img1 = img.copy()

    # testing_label ranges
    # moving it to hsv (hue, saturation, value) space
    # don't remember what this section was for
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # lower = np.array([0, 0, 0])
    # upper = np.array([160, 160, 160])
    #
    # mask = cv2.inRange(hsv, lower, upper)
    # res = cv2.bitwise_and(img, img, mask= mask)
    # mask_inv = cv2.bitwise_not(mask)

    cv2.imshow('image', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask_inv', mask_inv)
    # cv2.imshow('res', res)

    # BLURRING SECTION
    # blurring options, gaussian/median/bilateral
    # blur - standard cv2 blur
    # gaussian - good for reducing gaussian noise
    # median - good for reducing noise such as salt-and-pepper noise
    # bilateral - good for preserving edges

    # ksize is kernal size
    # bordertype tries extrapolate pixels outside the image, 0 is a sentinal value for a bordertype constant
    grayscaled = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img1, ksize=(3, 3), borderType=0)
    gausblur = cv2.GaussianBlur(img1, (3, 3), 0)
    medblur = cv2.medianBlur(img1, ksize=3)
    medblur2 = cv2.medianBlur(medblur, ksize=5)
    # d is diameter of the pixel neighborhood used for filtering
    # d > 5 is considered a large filter and 9 is suggested for offline heavy noise filtering
    d = 9
    # sigmacolor is the range of colors that will be mixed together within the neighborhood
    # large value means colors can mix with nonlike colors (ie: red and green)
    # small value means colors will only mix with like colors (ie: blue and indigo)
    color = d * d
    color = 100
    # sigmaspace is used for determining the pixel neighborhood for filtering
    # space is irrelevant if d (diameter) is > 0, if you specify d it'll use that, if you don't it'll use sigmaspace
    space = d * d
    bil1 = cv2.bilateralFilter(medblur, d, color, space)
    bil2 = cv2.bilateralFilter(bil1, d, color, space)
    bil3 = cv2.bilateralFilter(bil2, d, color, space)
    bil4 = cv2.bilateralFilter(bil3, d, color, space)

    # showing and saving the blurred images
    # cv2.imshow('blur', blur)
    cv2.imshow('medblur', medblur)
    # cv2.imshow('medblur2', medblur2)
    cv2.imshow('bil', bil1)
    # cv2.imwrite(out_path + 'medblue' + in_img, medblur)
    # cv2.imwrite(out_path + 'bilblur' + in_img, bil1)

    # picking the final blurred image for the rest of the program
    blurred = bil1

    # COLOR SCALING SECTION
    # 0's out the nondesired color channels
    # to bluescale for example you zero our red and green
    # remember color format is bgr
    blue = blurred.copy()
    for r in blue:
        for c in r:
            c[1] = 0
            c[2] = 0
    # red = blurred.copy()
    # for r in red:
    #     for c in r:
    #         c[0] = 0
    #         c[1] = 0
    # green = blurred.copy()
    # for r in green:
    #     for c in r:
    #         c[0] = 0
    #         c[2] = 0

    # showing the results of scaling
    # cv2.imshow('img', img)
    cv2.imshow('blue', blue)
    # cv2.imshow('green', green)
    # cv2.imshow('red', red)

    # THRESHOLDING SECTION
    # if you want to use the bluescaled image use the blue threshold
    # note it is considerably lower and much more sensitive than using a grayscaled image for thresholding
    gray_thresh = 105
    thresh = gray_thresh
    # blue_thresh = 10
    # thresh = blue_thresh
    # blurred = blue
    # cv2.imwrite(out_path + 'blue' + in_img, blue)

    # the chosen blue/gray blurred image makes it this section, if grayscaled this line does nothing
    # but if bluescaled you'll need to grayscale the bluescale for thresholding
    grayblur = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(out_path + 'gray' + in_img, grayblur)
    # anything higher than the thresh value is set to 255 (white)
    # these functions return two objects, a value that indicates the optimal threshold if otsu is used (if not otsu than
    # it returns the threshold that was passed to it) and the thresholded image.
    retval, threshold = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    retval2, threshold2 = cv2.threshold(grayblur, thresh, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(out_path + 'threshold2' + in_img, threshold2)
    #  guassian and otsu thresholding
    gaus = cv2.adaptiveThreshold(grayblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    retval3, otsu = cv2.threshold(grayblur, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # showing the thresholded images
    # cv2.imshow('threshold', threshold)
    cv2.imshow('threshold2', threshold2)
    # cv2.imshow('gaus', gaus)
    # cv2.imshow('otsu', otsu)

    # MASKING SECTION
    # since our threshold does a good job of finding backgrounds instead of bees we inverse the mask
    # you could also also inverse the thresholding section, but this follows the thought process of the initial idea of
    # finding the background
    mask_thresh = cv2.bitwise_not(threshold2)
    # cv2.imwrite(out_path + 'mask_thresh' + in_img, mask_thresh)
    img_thresh = cv2.bitwise_and(img, img, mask=mask_thresh)
    cv2.imshow('img_thresh', img_thresh)

    # SKELETON SECTION
    # create a binary image and feed into the medial axis or skeletonize functions
    # ideal output from these is to create a "spine" of the bee with no branches
    binary = img_as_bool(mask_thresh)
    med = mp.medial_axis(binary)
    skele = mp.skeletonize(binary)
    # skele = mp.skeletonize(med)

    # compare the skeletons
    # print(med[0])
    #
    # f, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.imshow(med, cmap='gray', interpolation='nearest')
    # ax1.imshow(skele, cmap='gray', interpolation='nearest')
    # plt.show()
    # plt.clf()

    # creates a skeleton over the image
    # img_copy = grayscaled.copy()
    # for row in range(len(skele)):
    #     for col in range(len(skele[row])):
    #         if skele[row][col]:
    #             img_copy[row][col] = 255
    #
    # # print(med[0][0] == False)+
    # # print(img_copy[0][0])
    # cv2.imshow("copy", img_copy)
    #
    # # print(img_copy[25])
    # arr = []
    #
    # for row in range(len(img_copy)):
    #     for col in range(len(img_copy[row])):
    #         if img_copy[row][col].all() == 0:
    #             arr.append(img[row][col])
    #
    # print(arr)

    # cv2.imwrite(out_path + in_img, img_thresh)

    # turns the "binary" bool array to bgr array
    bw = img.copy()
    sk = skele
    for row in range(len(sk)):
        for col in range(len(sk[row])):
            if sk[row][col]:
                bw[row][col] = [255, 255, 255]
            else:
                bw[row][col] = [0, 0, 0]
    cv2.imshow("bw", bw)
    # cv2.imwrite("skele.png", bw)

    # get the x and y coords of the skeleton
    x = []
    y = []
    for row in range(len(bw)):
        for col in range(len(bw[row])):
            if (bw[row][col] != 0).all():
                y.append(row)
                x.append(col)

    # get the x and y coords of the img
    cv2.imshow("img", img)
    x1 = []
    y1 = []
    for row in range(len(img)):
        for col in range(len(img[row])):
            if (img[row][col] != 255).all():
                y1.append(row)
                x1.append(col)

    # for plt
    img2 = im.imread(in_path + in_img)

    # polyfit
    # attempts to create a line of best fit using the coorinates of the skeleton over the linespace of the images coordinates
    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)
    x_new = np.linspace(np.min(x1), np.max(x1) + 1, 200)
    y_new = f(x_new)
    # print(x_new)
    # print(y_new)

    # plotting shenanigans
    # points = [[xn, yn] for xn, yn in zip(x_new, y_new) if (np.min(y1) <= yn <= np.max(y1))]
    # plt.imshow(threshold2, cmap='Greys')
    plt.imshow(img2)
    plt.plot(x_new, y_new)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig(out_path + in_img)
    plt.xlim(0, len(img2[0]) - 1)
    plt.ylim(len(img2) - 1, 0)
    plt.show()
    plt.clf()


    # GETTING THE COLOR VALUES OF THE LINE OF BEST FIT
    # interpolation stuff
    # coming back to this I think interpolation might be done automatically in getting the right color in python
    # the whole reason to do this is to get the expected color at a point on our line of best fit
    # because the line of best fits' points may not be an exact pixel coordinate, we interpolate the color value from
    # the image at what would be expected at that coordinate
    # in short if given coordinate (2.34, 19.89) we interpolate the color for that coordinate
    points = [[xn, yn] for xn, yn in zip(x_new, y_new) if (np.min(y1) <= yn <= np.max(y1))]
    # print(points)
    x_points = [round(point[0]) for point in points]
    y_points = [round(point[1]) for point in points]
    interpolated_values = []
    for point in points:
        px = point[1]
        py = point[0]
        ipx = int(px)
        ipy = int(py)
        if px != float(ipx) and py == float(ipy):
            pxf = math.floor(px)
            pxc = math.ceil(px)
            c = np.array(img2[pxc][py])
            f = np.array(img2[pxf][py])
            fs = f * (1 - (px - pxf))
            cs = c * (1 - (1 - (px - pxf)))
            clr = fs + cs
            interpolated_values.append(clr)
        elif px == float(ipx) and py != float(ipy):
            pyf = math.floor(py)
            pyc = math.ceil(py)
            c = np.array(img2[px][pyc])
            f = np.array(img2[px][pyf])
            fs = f * (1 - (py - pyf))
            cs = c * (1 - (1 - (py - pyf)))
            clr = fs + cs
            interpolated_values.append(clr)
        elif px != float(ipx) and py != float(ipy):
            pxf = math.floor(px)
            pxc = math.ceil(px)
            pyf = math.floor(py)
            pyc = math.ceil(py)
            c = np.array(img2[pxc][pyf])
            f = np.array(img2[pxf][pyf])
            fs = f * (1 - (px - pxf))
            cs = c * (1 - (1 - (px - pxf)))
            xyf = fs + cs
            c = np.array(img2[pxc][pyc])
            f = np.array(img2[pxf][pyc])
            fs = f * (1 - (px - pxf))
            cs = c * (1 - (1 - (px - pxf)))
            xyc = fs + cs
            fs = xyf * (1 - (py - pyf))
            cs = xyc * (1 - (1 - (py - pyf)))
            # fs = np.round(xyf * (1 - (py - pyf)) * 255)
            # cs = np.round(xyc * (1 - (1 - (py - pyf))) * 255)
            clr = (fs + cs)
            # clr = [int(it) for it in clr]
            interpolated_values.append(clr)
        else:
            interpolated_values.append(img2[px][py])

    # if the threshold values are met add the interpolated values to the list of values
    # if the pixel on the image is black (the bee part) grab the expected color at that pixel coordinate
    threshold_values = []
    t_px = []
    t_py = []
    for i in range(len(points)):
        px = x_points[i]
        py = y_points[i]
        if threshold2[py][px] == 0:
            t_px.append(px)
            t_py.append(py)
            # threshold_values.append(img2[py][px])
            threshold_values.append(interpolated_values[i])

    # print(t_py[0], t_px[0])
    # print(img2[t_py[0]][t_px[0]])
    print(threshold_values)

    # PLOTTING STUFF
    # plots just the pixels grabbed
    plt.scatter(t_px, t_py, facecolors=threshold_values, marker='s')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()

    # plots those pixel in an overlay on image, makes a little box around them
    plt.imshow(img2)
    plt.scatter(t_px, t_py, facecolors=threshold_values, marker='s', edgecolors='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(x_new, y_new)
    plt.xlim(0, len(img2[0]) - 1)
    plt.ylim(len(img2) - 1, 0)
    # plt.gca().invert_yaxis()
    plt.show()
    plt.clf()

    # plots the color ratios of those chosen values
    # pretty sure this ended up being not correct color label wise because plt.show is rbg and
    # the images worked with are in bgr, so it plots them color inversed
    plt.plot(np.array(threshold_values)[:, 0], c='b')
    plt.plot(np.array(threshold_values)[:, 1], c='g')
    plt.plot(np.array(threshold_values)[:, 2], c='r')
    plt.show()
    plt.clf()

    # OVAL ATTEMPT
    # bwg = img.copy()
    # mt = mask_thresh
    # for row in range(len(mt)):
    #     for col in range(len(mt[row])):
    #         if mt[row][col]:
    #             bwg[row][col] = [255, 255, 255]
    #         else:
    #             bwg[row][col] = [0, 0, 0]
    # cv2.imshow("bwg", bwg)
    #
    # # coords of the oval
    # x = []
    # y = []
    # for row in range(len(bwg)):
    #     for col in range(len(bwg[row])):
    #         if (bwg[row][col] != 0).all():
    #             y.append(row)
    #             x.append(col)
    #
    # # for plt
    # img3 = im.imread(in_path + in_img)
    #
    # # polyfit
    # z = np.polyfit(x, y, 2)
    # f = np.poly1d(z)
    # x_new = np.linspace(min(x1), max(x1), 100)
    # y_new = f(x_new)
    # plt.imshow(img3)
    # plt.plot(x_new, y_new)
    # plt.gca().set_aspect('equal', adjustable='box')
    # # plt.savefig(out_path + in_img)
    # plt.show()
    # plt.clf()

    # BACKGROUND SUBTRACTOR
    # vid = '14-12-57'
    # video = cv2.VideoCapture('video/18-01-33.h264')
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    #
    # while True:
    #     ret, frame = video.read()
    #     fgmask = fgbg.apply(frame)
    #
    #     # cv2.imshow('orginal', frame)
    #     cv2.imshow('fg', fgmask)
    #
    #     k = 0xFF & cv2.waitKey(30)  # Wait for a second
    #     if k == 27:
    #         break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
