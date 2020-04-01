import cv2
import numpy as np
import math

def main():
    image = cv2.imread("circuit-board-pepper.tif",0)
    size = int(input("Please enter filter size: "))
    new_image = mean_filter(image,size)
    cv2.imshow("original", image)
    cv2.imshow("new", new_image)
    cv2.waitKey()

def contraharmonic_mean_filter(image,size): #Smoothing image

    q = float(input("Please enter parameter Q: "))
    rows, cols = image.shape
    image = image.astype("float")
    print("Original Image Dimensions: ",rows,cols)
    new_image = np.zeros((rows,cols))
    sa = int(np.floor(size / 2))
    nom = 0
    denom = 0
    for i in range(sa,rows-sa):
        for j in range(sa,cols-sa):
            for k in range(size):
                for l in range(size):

                    nom += ((image[i+k-sa][j+l-sa])**(q+1))
                    denom += ((image[i+k-sa][j+l-sa])**(q))
                    #print(j,"-",image[i+k-sa][j+l-sa],"-",nom,"-",denom)
            if denom != 0:
                new_image[i][j] = nom/denom
                #print(new_image[i][j])
            nom=0
            denom=0
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