import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy
import scipy.interpolate as interp
import pickle
import colorsys
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances


def rbg_percentage(images):
    # any color below this value is considered a transition pixel from the segmented image to a black background that
    # may be created from saving or any other means.
    threshold = 40
    percentages = []
    #  for every image in our image list
    for img in images:
        rgb = []
        # for each pixel in the image
        for i in range(len(img)):
            for j in range(len(img[i])):
                # check if the color values are above the threshold
                if img[i][j][0] > threshold and img[i][j][1] > threshold and img[i][j][2] > threshold:
                    # if so add the values
                    rgb.append([img[i][j][0], img[i][j][1], img[i][j][2]])

        # normalize rbg values
        rgb = normalize(rgb)
        # sums up each of the images percentage of color
        # sum(), sums a 2d array by column
        partial = sum(rgb)
        # sums up the total percentages
        total = sum(partial)

        # get each color percentage
        red_percentage = partial[0] / total
        green_percentage = partial[1] / total
        blue_percentage = partial[2] / total

        percentages.append([red_percentage, green_percentage, blue_percentage])

    rbg_percentages = np.array(percentages)

    # 3D scatter plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(rbg_percentages[:, 0], rbg_percentages[:, 1], rbg_percentages[:, 2])
    ax.set_xlabel('red')
    ax.set_ylabel('green')
    ax.set_zlabel('blue')

    # 2d scatter plot, make sure to adjust labels appropriately
    # plt.scatter(rbg_percentages[:, 0], rbg_percentages[:, 1])
    # plt.xlabel('red')
    # plt.ylabel('green')

    plt.show()
    plt.clf()

    return rbg_percentages


def directory_to_images(directory):
    filenames = []
    images = []
    # for each file in the directory
    for file in os.listdir(directory):
        # add the file name
        filenames.append(file)
        # get and add the image
        img = cv2.imread(directory + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return filenames, images


# note that the cv2 image is left as a BGR image because any cv2 functions working with a cv2 image assume that the
# cv2 image is BGR
def line(in_img):
    # create a cv2 image, convert it to grayscale and apply a threshold
    img = cv2.imread(in_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    # retval, lower = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    # retval, upper = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    # threshold = cv2.bitwise_and(upper, lower)
    # cv2.imshow('threshold', threshold)

    # create a mask from the threshold
    mask_thresh = threshold
    # threshold2 = cv2.bitwise_not(mask_thresh)

    # logical and the mask with the original image to remove any non bee background
    img = cv2.bitwise_and(img, img, mask=mask_thresh)

    # midpoints method
    # creates a list of x and y coordinates for the midpoints of the honey bee
    midpointsx = []
    midpointsy = []

    # for each row in the image
    for row in range(len(mask_thresh)):
        rowlist = []
        x = 0
        y = 0
        # for each column
        for col in range(len(mask_thresh[row])):
            # if the pixel isn't zero, add its coordinates to the list
            if mask_thresh[row][col] != 0:
                rowlist.append((row, col))
        # sum the x's and y's
        for coord in rowlist:
            y += coord[0]
            x += coord[1]
        # if they are nonzero, divide by the amount of valid pixels that went into the sum
        if x != 0 and y != 0:
            x /= len(rowlist)
            y /= len(rowlist)
            # floor is used so they can be used as indices
            x = math.floor(x)
            y = math.floor(y)
        # if nonzero add them to the midpoints
        if x != 0 and y != 0:
            midpointsx.append(x)
            midpointsy.append(y)

    # preview the midpoints chosen overlayed the mask
    # plt.imshow(mask_thresh)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.scatter(midpointsx, midpointsy, c='k', marker='s')
    # plt.xlim(0, len(mask_thresh[0]) - 1)
    # plt.ylim(len(mask_thresh) - 1, 0)
    # plt.show()
    # plt.clf()

    # get the colors at the midpoint indices
    mp_colors = []
    # any individual colors that fall outside this threshold are considered non honey bee colors
    # the upper threshold looks to remove highlights and the lower threshold removes black background as this indexing
    # refers back to the original image
    upper_thresh = 1.25
    lower_thresh = 0.5
    for i in range(1, len(midpointsy)):
        y = midpointsy[i]
        x = midpointsx[i]
        color = []
        # from the midpoint, expand to the left and right 5 pixels, creating an 11 pixel slice.  Sum the color channels
        # for each respective color, blue, green, red
        # grab the blue color channel for this section
        bc = [img[y][x - 5][0], img[y][x - 4][0], img[y][x - 3][0], img[y][x - 2][0], img[y][x - 1][0], img[y][x][0],
                 img[y][x + 1][0], img[y][x + 2][0], img[y][x + 3][0], img[y][x + 4][0], img[y][x + 5][0]]
        # gets rid of highlights and masked values at locations such as the end of the thorax
        # remove any zero values, this would be if we grabbed background values
        bc = [i for i in bc if i != 0]
        # revome any values that are brighter than the average (highlights), or much lower than the average (transition
        # pixels from honey bee to backgroun)
        b = [i for i in bc if (i <= upper_thresh * np.mean(bc) and i >= lower_thresh * np.mean(bc))]
        # get the blue average
        b = sum(b) / len(b)
        gc = [img[y][x - 5][1], img[y][x - 4][1], img[y][x - 3][1], img[y][x - 2][1], img[y][x - 1][1], img[y][x][1],
                 img[y][x + 1][1], img[y][x + 2][1], img[y][x + 3][1], img[y][x + 4][1], img[y][x + 5][1]]
        gc = [i for i in gc if i != 0]
        g = [i for i in gc if (i <= upper_thresh * np.mean(gc) and i >= lower_thresh * np.mean(gc))]
        g = sum(g) / len(g)
        rc = [img[y][x - 5][2], img[y][x - 4][2], img[y][x - 3][2], img[y][x - 2][2], img[y][x - 1][2], img[y][x][2],
                 img[y][x + 1][2], img[y][x + 2][2], img[y][x + 3][2], img[y][x + 4][2], img[y][x + 5][2]]
        rc = [i for i in rc if i != 0]
        r = [i for i in rc if (i <= upper_thresh * np.mean(rc) and i >= lower_thresh * np.mean(rc))]
        r = sum(r) / len(r)

        color.append(r)
        color.append(g)
        color.append(b)
        mp_colors.append(color)

    # color ratios, plots the colors across the honey bee's midpoints
    # plt.plot(np.array(mp_colors)[:, 0], c='r')
    # plt.plot(np.array(mp_colors)[:, 1], c='g')
    # plt.plot(np.array(mp_colors)[:, 2], c='b')
    # plt.title(in_img)
    # plt.show()
    # plt.clf()

    # raw image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # plt.clf()

    return mp_colors


def stripes_dir(directory):
    max_size = 0
    min_size = 1000
    stripes = []
    for filename in os.listdir(directory):
        label = filename
        in_img = directory + label
        mpc = line(in_img)
        stripes.append(mpc)
        z = len(mpc)
        if z > max_size:
            max_size = z
        if z < min_size:
            min_size = z

    return stripes, max_size, min_size


# takes a list of rbg values and stretches them to the desired length via interpolation
def stretch(color_array, max_size):
    x = np.array(color_array)
    new_array = []
    ca_interp_r = interp.interp1d(np.arange(x[:, 0].size), x[:, 0])
    ca_stretch_r = ca_interp_r(np.linspace(0, x[:, 0].size - 1, max_size))
    ca_interp_g = interp.interp1d(np.arange(x[:, 1].size), x[:, 1])
    ca_stretch_g = ca_interp_g(np.linspace(0, x[:, 1].size - 1, max_size))
    ca_interp_b = interp.interp1d(np.arange(x[:, 2].size), x[:, 2])
    ca_stretch_b = ca_interp_b(np.linspace(0, x[:, 2].size - 1, max_size))

    for i in range(max_size):
        new_array.append([ca_stretch_r[i], ca_stretch_g[i], ca_stretch_b[i]])

    return new_array


# normalizes the stripes within themselves, this is get around the difference in color values of the same honey bee being
# photographed in the sunshine vs the shade
def normalize_stripes_3ch(stripes):
    # this normalized max value we wish to convert all values to
    # 255 is chosen in case we wish to compare images by plotting later
    normval = 255
    # for each stripe
    for s in range(len(stripes)):
        # get the max and min colors for normalization
        sarray = np.array(stripes[s])
        maxr = np.max(sarray[:, 0])
        maxg = np.max(sarray[:, 1])
        maxb = np.max(sarray[:, 2])
        minr = np.min(sarray[:, 0])
        ming = np.min(sarray[:, 1])
        minb = np.min(sarray[:, 2])

        # normalize function is (value - min) / (max - min) * normval
        # for each value in the stripe
        for j in range(len(stripes[s])):
            r = (stripes[s][j][0] - minr) / (maxr - minr) * normval
            g = (stripes[s][j][1] - ming) / (maxg - ming) * normval
            b = (stripes[s][j][2] - minb) / (maxb - minb) * normval
            stripes[s][j][0] = r
            stripes[s][j][1] = g
            stripes[s][j][2] = b

    return stripes


def stripe_processor(stripes, max_size):
    stretched_stripes = []
    # in case individual color channels want to be used or to compared
    stripes_red = []
    stripes_green = []
    stripes_blue = []
    for i in range(len(stripes)):
        # stretch the stripe to the desired length
        x = stretch(copy.deepcopy(stripes[i]), max_size)
        stretched_stripes.append(x)
        # get the red values
        stripes_red.append(np.array(x)[:, 0])
        # get the green values
        stripes_green.append(np.array(x)[:, 1])
        # get the blue values
        stripes_blue.append(np.array(x)[:, 2])

    normalize_stripes = normalize_stripes_3ch(stretched_stripes)

    return normalize_stripes


# turns the rgb values in a list into hsl values and returns the hue metric
def rgb_hls(stripe):
    hues = []
    for i in range(len(stripe)):
        r = stripe[i][0] / 255
        g = stripe[i][1] / 255
        b = stripe[i][2] / 255
        hues.append(colorsys.rgb_to_hls(r, g, b)[0])

    return hues


# given a data set and a window size, makes windows and multiplies them by a hamming function.
def window_creation(data, ws):
    # category creation
    overlap = 0.5
    slide = int(ws * overlap)
    l = len(data[0])
    # number of windows
    num_windows = int((l - ws) / slide + 1)
    # hamming window creation
    hw = np.hamming(ws)

    windows = []
    # for each stripe in the data
    for h in data:
        # for each window
        for s in range(num_windows):
            # get the index ranges that we want to multiply the hamming window by
            x = int(s * slide)
            y = x + ws
            # create a window and append it to our list of windows
            window = [a * b for a, b in zip(h[x:y], hw)]
            windows.append(window)
            # plt.plot(window)
            # plt.show()
            # plt.clf()
    windows = np.array(windows)

    return windows


# computes the inertia or within-cluster sum-of-squares
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


# this is used to analyze certain k values given our base windows
# it uses 3 methods, the elbow, silhouette, and gap method
# with significant k sizes, this function will run very long with the gap method implemented
def window_testing(windows, ws):
    sil_vals = []
    num_iter = 300
    distortions = []
    # this is use to create a range of k's one would like to check
    min_n = 6
    n = 10
    # creates a random distribution of data with the same shape as our windows, comment this out if gap method isn't being used
    gap_reference = np.random.rand(*windows.shape)
    reference_inertia = []
    ondata_inertia = []

    for i in range(min_n, n + 1):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=num_iter,
                    random_state=0)

        # gap method
        local_inertia = []
        for j in range(n):
            KM = copy.copy(km)
            assignments = KM.fit_predict(gap_reference)
            local_inertia.append(compute_inertia(assignments, gap_reference))
        reference_inertia.append(np.mean(local_inertia))
        # comment the previous if you aren't using the gap method, will significantly slow runtime on large k

        # fit our data to the classifier and predict the data
        km.fit(windows)
        y_km = km.predict(windows)
        # gets inertia for the gap method
        ondata_inertia.append(compute_inertia(y_km, windows))
        # gets the distortion and silhouette avg for the elbow and silhouette methods
        distortions.append(km.inertia_)
        sil_avg = silhouette_score(windows, y_km)
        sil_vals.append(sil_avg)

    # elbow method
    plt.plot(range(min_n, n + 1), distortions, marker='o')
    plt.yticks(np.arange(min(distortions), max(distortions) + 1, 150))
    plt.xticks(np.arange(min_n, n + 1, 10))
    plt.xlabel('num classes')
    plt.ylabel('distortion')
    plt.title("Elbow Method, Window Size %s" % (ws))
    plt.show()
    plt.clf()

    # average silhouette method
    plt.plot(range(min_n, n + 1), sil_vals)
    plt.xticks(np.arange(min_n, n + 1, 1))
    plt.xlabel('num classes')
    plt.ylabel('silhouette average')
    plt.title("Silhouette Method, Window Size %s" % (ws))
    plt.show()
    plt.clf()

    # gap statistic plotting
    # comment section if gap method isn't being used
    # first plot shows the inertia change between random and the actual data
    # second shows the gap between this inertia plotting
    # a larger gap is ideal, but keep in mind the scale of change in inertia, sometimes it is trivially small
    plt.plot(range(min_n, n + 1), reference_inertia,
             '-o', label='reference')
    plt.plot(range(min_n, n + 1), ondata_inertia,
             '-o', label='data')
    plt.xticks(np.arange(min_n, n + 1, 1))
    plt.xlabel('num_classes')
    plt.ylabel('inertia')
    plt.title("Inertia Differences, Window Size %s" % (ws))
    plt.show()
    plt.clf()
    gap = [x - y for x,y in zip(reference_inertia, ondata_inertia)]
    plt.plot(range(min_n, n + 1), gap, '-o')
    plt.xticks(np.arange(min_n, n + 1, 1))
    plt.ylabel('gap')
    plt.xlabel('num classes')
    plt.title("Gap Method, Window Size %s" % (ws))
    plt.show()
    plt.clf()


# returns true if two lists are the same within a specified number of differences allowed
def same(list1, list2, num_disagreements=0):
    num = 0
    for i, j in zip(list1, list2):
        # this comparitor is fine for numbers as it != compares value.
        if i != j:
           num += 1
    # if there are too many disagreements return false
    if num > num_disagreements:
        return False
    else:
        return True


# predicts how many different classes are within
def diversity_prediction(prediction_windows, num_disagreements):
    # assign class numbers and a class number to the indice of each series
    class_number = 0
    indices = []
    indices.append(class_number)
    classes = []
    # the first class is the first series
    classes.append(prediction_windows[0])
    # for every other series
    for i in range(1, len(prediction_windows)):
        flag = False
        # for each of our classes
        for j in range(len(classes)):
            # if the series is considered the same as a class
            if same(prediction_windows[i], classes[j], num_disagreements):
                # mark which class it is in the indice list, mark our flag and break out of the loop
                indices.append(j)
                flag = True
                break
        # if we never found a class the series was similar too
        if flag == False:
            # create a new class number
            class_number += 1
            # place that number in the indice list and add the series to our list of classes
            indices.append(class_number)
            classes.append(prediction_windows[i])

    # for each class increment its distribution everytime the respective indice appears in our indice list
    class_distribution = np.zeros(len(classes))
    for ind in indices:
        class_distribution[ind] += 1

    return classes, class_distribution


# for doing a finer prediction on a series of classes that have are in a class by themselves
def refined_diversity_prediction(classes, class_distribution, num_disagreements):
    new_classes = []
    new_indices = []
    new_class_number = 0
    # for each class
    for i in range(len(classes)):
        # if that class has isn't a solo class
        if class_distribution[i] > 1:
            # add that class to our class list and update the indice list and class number
            new_classes.append(classes[i])
            new_indices.append(new_class_number)
            new_class_number += 1
        # else do perform the code from the previous function
        else:
            flag = False
            # for each class
            for j in range(len(new_classes)):
                # compare our current class to the list of new classes and see if they are the same
                if same(classes[i], new_classes[j], num_disagreements):
                    # if the same as a class in our new class, update the indice
                    new_indices.append(j)
                    flag = True
                    break
            # if it isn't the same as any of our other classes after a less strenuous restriction then it is still a unique class
            if flag == False:
                # update the indices, class number, and add it to our class list
                new_indices.append(new_class_number)
                new_class_number += 1
                new_classes.append(classes[i])

    # get the distribution of classes
    new_class_distribution = np.zeros(len(new_classes))
    for ind in new_indices:
        new_class_distribution[ind] += 1

    return new_classes, new_class_distribution


# plots the class distribution, most of the params are for the figure title and save locations
def plot_classes(num_windows, num_disagreements, hive, aos, k, classes, class_distribution, window_size, perc_option):
    plt.xlabel('Class Label')
    plt.ylabel('Number per Class')
    save_loc = ''
    if perc_option:
        perc = (num_windows - num_disagreements) / num_windows * 100
        plt.title("Hive: %s for %s with K of %s\nTotal Classes: %s\n%s Percent Agreement" % (hive, aos, str(k), str(len(classes)), "%.2f" % perc))
        save_loc = "%s/test_figures/winsize%s_hive%s_aos%s.png" % (image_directory, str(window_size),  str(hive), str(aos))
    else:
        plt.title("Hive: %s for %s with K of %s\nTotal Classes: %s\nRefined Classification" % (hive, aos, str(k), str(len(classes))))
        save_loc = "%s/test_figures/winsize%s_hive%s_aos%s_r.png" % (image_directory, str(window_size), str(hive), str(aos))
    plt.yticks(np.arange(0, max(class_distribution), 5))
    x = [i for i in range(0, len(class_distribution))]
    plt.bar(x, class_distribution)
    # plt.savefig(save_loc)
    plt.show()
    plt.clf()


def main():
    # this is the testing and training image location for the images that were used for the window clustering
    testing = image_directory + 'base/'
    base_filenames, base_images = directory_to_images(testing)

    # if you would like to check the color distribution of your images
    # potentially useful for color analysis, such as correlation among colors or determining if a channel isn't needed
    # rbg_percentages = rbg_percentage(base_images)

    # stripe representations of the honey bee stripe patterns
    stripes, max_size, min_size = stripes_dir(testing)

    # normalize all our stripes to the same length so that they can be compared
    # this 84 is known that the longest stripe of the data we want to analyze is 84, this is to go ahead and normalize
    # our testing bees.  With a new data set, all images need to be have their stripes extracted and lengths analysed
    # to find what size they need to be normalized to beforehand
    max_size = 84
    processed_stripes = stripe_processor(stripes, max_size)

    # from here it is suggest that the processed stripes be saved and loaded for future analysis
    # pickle.dump(processed_stripes, open("pickles1/base_stripes.p", "wb"))
    processed_stripes = pickle.load(open("pickles1/base_stripes.p", "rb"))

    # area of study, what month, day, time, hive
    hive = '11b'

    month = 'oct'
    aos = month
    dir1 = '%s/%s/%s/10/' % (image_directory, hive, month)
    dir2 = '%s/%s/%s/12/' % (image_directory, hive, month)
    dir3 = '%s/%s/%s/2/' % (image_directory, hive, month)
    dir4 = '%s/%s/%s/4/' % (image_directory, hive, month)
    # time = '4'
    # aos = time
    # dir1 = '%s/%s/apr/%s/' % (image_directory, hive, time)
    # dir2 = '%s/%s/jun/%s/' % (image_directory, hive, time)
    # dir3 = '%s/%s/aug/%s/' % (image_directory, hive, time)
    # dir4 = '%s/%s/oct/%s/' % (image_directory, hive, time)

    # images of area of study
    fn1, im1 = directory_to_images(dir1)
    fn2, im2 = directory_to_images(dir2)
    fn3, im3 = directory_to_images(dir3)
    fn4, im4 = directory_to_images(dir4)
    aos_fns = fn1 + fn2 + fn3 + fn4
    aos_imgs = im1 + im2 + im3 + im4

    stripes1, max_size1, min_size1 = stripes_dir(dir1)
    stripes2, max_size2, min_size2 = stripes_dir(dir2)
    stripes3, max_size3, min_size3 = stripes_dir(dir3)
    stripes4, max_size4, min_size4 = stripes_dir(dir4)
    aos_stripes = stripes1 + stripes2 + stripes3 + stripes4
    aos_max_size = max(max_size1, max_size2, max_size3, max_size4)
    # previously known
    aos_max_size = 84
    aos_processed_stripes = stripe_processor(aos_stripes, aos_max_size)

    # pickle.dump(aos_processed_stripes, open("pickles1/%s/%s/stripes.p" % (hive, aos), "wb"))
    # aos_processed_stripes = pickle.load(open("pickles1/%s/%s/stripes.p" % (hive, aos), "rb"))

    # gets all the windows for our testing data
    window_hues = []
    for stripe in processed_stripes:
        window_hues.append(rgb_hls(stripe))

    plt.plot(window_hues[5])
    plt.show()
    plt.clf()
    #
    # # shuffles the data
    data = window_hues
    # ran = list(zip(window_hues, base_filenames))
    # random.seed(0)
    # random.shuffle(ran)
    # data, filenames = zip(*ran)
    window_size = 24

    windows = window_creation(data, window_size)
    # the data can be pickled here for easier retrieval to save on computation time for later analysis
    # pickle.dump(windows, open("pickles1/knn_1002_%s_windows.p" % (window_size), "wb"))
    windows = pickle.load(open("pickles1/knn_1002_%s_windows.p" % (window_size), "rb"))

    # analyze the windows, determine a good k size
    # window_testing(windows, window_size)

    # get the hue for our area of study
    aos_hues = []
    for stripe in aos_processed_stripes:
        aos_hues.append(rgb_hls(stripe))

    # creates the windows for our area of study, we pickle them so
    aos_windows = window_creation(aos_hues, window_size)
    # pickle.dump(aos_windows, open("pickles1/%s/%s/%s.p" % (hive, aos, str(window_size)), "wb"))
    # aos_windows = pickle.load(open("pickles1/%s/%s/%s.p" % (hive, aos, str(window_size)), "rb"))

    data = aos_windows

    # this is the chosen k after analysis
    k = 17
    # using our chosen k, we fit our base windows the the KMeans classifier and have the option to view plot the centroids
    num_iter = 300
    km = KMeans(n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=num_iter,
                random_state=0)
    km.fit(windows)
    # centroids = km.cluster_centers_
    # plt.title('Window Centroids')
    # for c in centroids:
    #     plt.plot(c)
    # plt.show()
    # plt.clf()

    # create some information on the windows, how large a window must be slid, and how many total windows are made.
    ws = window_size
    overlap = 0.5
    slide = int(ws * overlap)
    num_windows = int((max_size - ws) / slide + 1)

    # the number of disagreements allowed for our windows.  Do we want 9 outta 10 windows to be the same, num_disagreements = 1
    num_disagreements = 1

    # will keep the class predicted for each window for all of the stripes
    prediction_windows = []
    # for each stripe
    for i in range(len(aos_processed_stripes)):
        # holds the class predicted for each window for a stripe
        prd = []
        # for each window in a stripe
        for j in range(num_windows):
            # get the correct index in our data set
            index = i * num_windows + j
            # get the class prediction from our classifier
            prediction = km.predict([data[index]])
            prd.append(prediction[0])
        # add the prediction series to our overall list
        prediction_windows.append(prd)

    # the classes and class ditribution for the given stripes
    classes, class_distribution = diversity_prediction(prediction_windows, num_disagreements)
    # plots the distribution
    plot_classes(num_windows, num_disagreements, hive, aos, k, classes, class_distribution, window_size, perc_option=True)

    # if a refined seach is chosen, increase the number of diaagreements allowed
    new_num_disagreements = num_disagreements + 1
    new_classes, new_class_distribution, = refined_diversity_prediction(classes, class_distribution, new_num_disagreements)
    plot_classes(num_windows, new_num_disagreements, hive, aos, k, new_classes, new_class_distribution, window_size, perc_option=False)


if __name__ == '__main__':
    image_directory = 'C:/Users/parkerat2/Desktop/diversity_code/images/'
    main()
