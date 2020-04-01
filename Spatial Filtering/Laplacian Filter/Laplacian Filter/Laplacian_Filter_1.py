import cv2
import numpy as np
import math
import scipy.ndimage.filters as filters
#This is done by own convolution code. It is a bit slower than the other.
def main():
    image = cv2.imread("Blurred_moon.tif",0)
    size = int(input("Please enter filter size(3 or 5): "))

    new_image = laplacian_filter(image,size)

    cv2.imshow("original", image)
    cv2.imshow("new", new_image)
    cv2.waitKey()

def laplacian_filter(image,size): #Smoothing image

    rows, cols = image.shape
    #image = (image - np.amin(image)) * 255.0 / (np.amax(image) - np.amin(image))
    print("Original Image Dimensions: ", rows, cols)
    new_image = np.zeros((rows, cols))
    if size == 3:
        filter = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
    if size == 5:
        filter = [[-1, -1, -1, -1, -1],
                  [-1, -1, -1, -1, -1],
                  [-1, -1, 24, -1, -1],
                  [-1, -1, -1, -1, -1],
                  [-1, -1, -1, -1, -1]]

    sa = int(np.floor(size / 2))
    print("filter",filter)

    #Kernel Convolution Algorithm
    for i in range(sa, rows - sa):
        for j in range(sa, cols - sa):
            for k in range(size):
                for l in range(size):
                    new_image[i][j] += image[i + k - sa][j + l - sa] * filter[k][l]
    #Copy original image's density values to empty pixels
    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-sa][j] = image[rows-i-sa][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-sa] = image[i][cols - j-sa]

    laplacian = new_image
    print(np.amin(laplacian), np.amax(laplacian))
    #laplacian = filters.convolve(image,filter)
    #new_image = np.subtract(new_image,image)
    new_image = image - laplacian
    print(np.amin(new_image), np.amax(new_image))
    new_image = new_image - np.amin(new_image)
    new_image = new_image * (255.0/np.amax(new_image)) #Map minus pixel values to between 0 and 255
    new_image = np.uint8(new_image)
    print(np.amin(new_image),np.amax(new_image))
    new_image = stretch(new_image,60,200) #increase contrast
    return new_image

#Histogram Stretching(increase contrast)
def stretch(a, lower_thresh, upper_thresh):
    r = 255.0/(upper_thresh-lower_thresh+2) # unit of stretching
    out = np.round(r*np.where(a>=lower_thresh,a-lower_thresh+1,0)).clip(max=255)
    return out.astype(a.dtype)

if __name__ == "__main__":

    main()