import cv2
import numpy as np
import math
from statistics import median
def main():

    image = cv2.imread("lenaa.png",0)
    size = int(input("Please enter filter size: "))
    filter_type = input("Please enter filter type: ")
    if filter_type == "mean":

        new_image = mean_filter(image,size)

    elif filter_type == "median":

        new_image = median_filter(image,size)

    cv2.imshow("original", image)
    cv2.imshow("new", new_image)
    cv2.waitKey()

def mean_filter(image,size): #Smoothing image

    filter = [[1]*size] * size
    print("Filter",filter)
    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)
    new_image = np.zeros((rows,cols),np.uint8)
    sa = int(np.floor(size/2))
    for i in range(sa,rows-sa):
        for j in range(sa,cols-sa):
            for k in range(size):
                for l in range(size):
                    new_image[i][j] += np.round((1/(size*size))*(image[i+k-sa][j+l-sa] * filter[k][l]))

    #Copy original image's density values to empty pixels
    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-1][j] = image[rows-i-1][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-1] = image[i][cols - j-1]

    return new_image

def median_filter(image,size):

    filter = []
    rows, cols = image.shape
    new_image = np.zeros((rows,cols),np.uint8)
    value_size = int(np.floor(size / 2))
    for i in range(value_size,rows-value_size):
        for j in range(value_size,cols-value_size):
            for k in range(size):
                for l in range(size):

                    filter.append(image[i+k-value_size][j+l-value_size])

            new_image[i][j] = np.round(median(filter))
            filter = []

    #Copy original image's density values to empty pixels
    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-1][j] = image[rows-i-1][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-1] = image[i][cols - j-1]

    return new_image

if __name__ == "__main__":
    main()