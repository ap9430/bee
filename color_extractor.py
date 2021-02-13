import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# gets a color line representing a particular bee's stripe
# takes an image location
def line(in_img):
    # image path
    # in_path = 'images/input/mt_bees/'
    in_path = 'conference/'
    # for saving images
    out_path = 'conference/'

    # read img, grayscale it
    img = cv2.imread(in_path + in_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binary the image and anything below a gray value of 30 becomes white
    retval, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    # retval, lower = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    # retval, upper = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    # threshold = cv2.bitwise_and(upper, lower)
    # cv2.imshow('threshold', threshold)

    mask_thresh = threshold
    # threshold2 = cv2.bitwise_not(mask_thresh)

    # get the midpoints of a bee from top to bottom
    midpointsx = []
    midpointsy = []
    # for each row
    for row in range(len(mask_thresh)):
        rowlist = []
        x = 0
        y = 0
        # get all nonzero columns
        for col in range(len(mask_thresh[row])):
            if mask_thresh[row][col] != 0:
                rowlist.append((row, col))
        # add the coord value to the list
        for coord in rowlist:
            y += coord[0]
            x += coord[1]
        # if the row wasn't empty
        if x != 0 and y != 0:
            x /= len(rowlist)
            y /= len(rowlist)
            x = math.floor(x)
            y = math.floor(y)
        # check again to make sure the row is valid
        # append the midpoints
        if x != 0 and y != 0:
            midpointsx.append(x)
            midpointsy.append(y)

    # plotting stuffs
    plt.imshow(mask_thresh)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(midpointsx, midpointsy, c='k', marker='s')
    plt.xlim(0, len(mask_thresh[0]) - 1)
    plt.ylim(len(mask_thresh) - 1, 0)
    # plt.title("midpoints")
    plt.axis('off')
    # plt.savefig(out_path + "midpoints" + in_img)
    plt.show()
    plt.clf()

    # MIDPOINTS METHOD
    # using the midpoints get the midpoint and the 4 pixels to the left and right of it for 9 total
    # this is to average out oddities like highlights
    mp_colors = []
    width = 9
    for i in range(1, len(midpointsy) - 1):
        y = midpointsy[i]
        x = midpointsx[i]
        color = []
        # b = img[y][x + 5][0] if img[y][x + 5][0] else 0
        b = sum([img[y][x - 4][0], img[y][x - 3][0], img[y][x - 2][0], img[y][x - 1][0], img[y][x][0],
                 img[y][x + 1][0], img[y][x + 2][0], img[y][x + 3][0], img[y][x + 4][0]]) / width
        g = sum([img[y][x - 4][1], img[y][x - 3][1], img[y][x - 2][1], img[y][x - 1][1], img[y][x][1],
                 img[y][x + 1][1], img[y][x + 2][1], img[y][x + 3][1], img[y][x + 4][1]]) / width
        r = sum([img[y][x - 4][2], img[y][x - 3][2], img[y][x - 2][2], img[y][x - 1][2], img[y][x][2],
                 img[y][x + 1][2], img[y][x + 2][2], img[y][x + 3][2], img[y][x + 4][2]]) / width
        color.append(b)
        color.append(g)
        color.append(r)
        mp_colors.append(color)

    # color ratios
    plt.plot(np.array(mp_colors)[:, 0], c='b')
    plt.plot(np.array(mp_colors)[:, 1], c='g')
    plt.plot(np.array(mp_colors)[:, 2], c='r')
    plt.title("midpoint color values")
    plt.ylabel('rbg value')
    plt.savefig(out_path + "mcv" + in_img)
    plt.show()
    plt.clf()

    # STRAIGHT LINE METHOD
    # gets the top and bottom midpoint and plots a line of best fit to them
    yt = midpointsy[0]
    xt = midpointsx[0]
    yb = midpointsy[-1]
    xb = midpointsx[-1]

    # fit the line
    p = np.polyfit((yt, yb), (xt, xb), 1)
    f = np.poly1d(p)
    xs = []
    for y in midpointsy:
        xs.append(f(y))

    # plot the line
    plt.imshow(mask_thresh)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(xs, midpointsy, c='k', marker='s')
    plt.xlim(0, len(mask_thresh[0]) - 1)
    plt.ylim(len(mask_thresh) - 1, 0)
    # plt.title("straightline")
    plt.axis('off')
    # plt.savefig(out_path + "straightline" + in_img)
    plt.show()
    plt.clf()

    # get the color values and average them
    s_colors = []
    for i in range(1, len(midpointsy) - 1):
        y = midpointsy[i]
        x = xs[i]
        color = []
        b = sum([img[y][x - 4][0], img[y][x - 3][0], img[y][x - 2][0], img[y][x - 1][0], img[y][x][0],
                 img[y][x + 1][0], img[y][x + 2][0], img[y][x + 3][0], img[y][x + 4][0]]) / width
        g = sum([img[y][x - 4][1], img[y][x - 3][1], img[y][x - 2][1], img[y][x - 1][1], img[y][x][1],
                 img[y][x + 1][1], img[y][x + 2][1], img[y][x + 3][1], img[y][x + 4][1]]) / width
        r = sum([img[y][x - 4][2], img[y][x - 3][2], img[y][x - 2][2], img[y][x - 1][2], img[y][x][2],
                 img[y][x + 1][2], img[y][x + 2][2], img[y][x + 3][2], img[y][x + 4][2]]) / width
        color.append(b)
        color.append(g)
        color.append(r)
        s_colors.append(color)

    # color ratios for straight line
    plt.plot(np.array(s_colors)[:, 0], c='b')
    plt.plot(np.array(s_colors)[:, 1], c='g')
    plt.plot(np.array(s_colors)[:, 2], c='r')
    plt.title("straightline color values")
    plt.ylabel('rbg value')
    plt.savefig(out_path + "slcv" + in_img)
    plt.show()
    plt.clf()

    # raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # plt.clf()

    # plots the color values of the midpoint and straightline methods
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(np.array(mp_colors)[:, 0], c='b')
    ax1.plot(np.array(mp_colors)[:, 1], c='g')
    ax1.plot(np.array(mp_colors)[:, 2], c='r')
    ax1.set_ylabel("rbg value")
    ax2.plot(np.array(s_colors)[:, 0], c='b')
    ax2.plot(np.array(s_colors)[:, 1], c='g')
    ax2.plot(np.array(s_colors)[:, 2], c='r')
    ax1.set_title("midpoints")
    ax2.set_title("straightline")
    fig.set_size_inches(14, 6, forward=True)
    plt.savefig(out_path + "colorvalues" + in_img)
    plt.show()
    plt.clf()

    return mp_colors, s_colors


# iterates a directory and gets all the stripes
# gets the min and max sizes of the stripes which is used for normalization and such
def stripes_dir(directory, max_size, min_size):
    path = 'images/input/mt_bees/'
    pd = path + directory

    stripes = []
    for filename in os.listdir(pd):
        label = filename
        in_img = directory + label
        mpc, sc = line(in_img)
        stripes.append(mpc)
        z = len(mpc)
        if z > max_size:
            max_size = z
        if z < min_size:
            min_size = z

    return stripes, max_size, min_size

# assumes the bee is already segmented from the background before this is run
def main():
    # image path
    # path = 'images/input/mt_bees/'
    # image directory
    # directory = "r24/"
    # pd = path + directory
    # base_directory = "diff24/"
    # bpd = path + base_directory
    # max_size, min_size = 0, 1000

    # pick a bee with its directory
    mp_color, sl_color = line('threshimage2cut.jpg')

    # stripes, max_size, min_size = stripes_dir(directory, max_size, min_size)


if __name__ == '__main__':
    main()