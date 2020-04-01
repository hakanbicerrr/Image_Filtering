import cv2
import numpy as np
import math
from statistics import median
def main():
    image = cv2.imread("uniform.tif",0)
    size = int(input("Please enter filter size: "))

    new_image = midpoint_filter(image,size)

    cv2.imshow("original", image)
    cv2.imshow("new", new_image)
    cv2.waitKey()

def midpoint_filter(image,size):

    filter = []
    rows, cols = image.shape
    new_image = np.zeros((rows,cols))
    value_size = int(np.floor(size / 2))
    for i in range(value_size,rows-value_size):
        for j in range(value_size,cols-value_size):
            for k in range(size):
                for l in range(size):

                    filter.append(image[i+k-value_size][j+l-value_size])

            max_val = float(np.amax(filter))
            min_val = float(np.amin(filter))
            new_image[i][j] = float((max_val+min_val)/2)
            filter = []

    #Copy original image's density values to empty pixels
    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-value_size][j] = image[rows-i-value_size][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-value_size] = image[i][cols - j-value_size]

    new_image = np.uint8(np.round(new_image))
    return new_image


if __name__ =="__main__":

    main()