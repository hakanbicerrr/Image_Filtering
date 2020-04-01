import cv2
import numpy as np
import math

def main():
    image = cv2.imread("cameraman.tif",0)
    size = int(input("Please enter filter size: "))
    new_image = harmonic_mean_filter(image,size)
    cv2.imshow("original", image)
    cv2.imshow("new", new_image)
    cv2.waitKey()

def harmonic_mean_filter(image,size): #Smoothing image

    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)
    new_image = np.zeros((rows,cols))
    sa = int(np.floor(size / 2))
    for i in range(sa,rows-sa):
        for j in range(sa,cols-sa):
            for k in range(size):
                for l in range(size):
                    if image[i+k-sa][j+l-sa] != 0:
                        new_image[i][j] += np.true_divide(1,image[i+k-sa][j+l-sa])
            if new_image[i][j] != 0:
                new_image[i][j] = np.true_divide((size*size),new_image[i][j])

    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-1][j] = image[rows-i-1][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-1] = image[i][cols - j-1]
    #new_image = new_image - np.amin(new_image)
    #new_image = new_image * (255.0 / np.amax(new_image))
    new_image = np.uint8(np.round(new_image))
    print(np.amin(new_image),np.amax(new_image))
    return new_image


if __name__ == "__main__":
    main()