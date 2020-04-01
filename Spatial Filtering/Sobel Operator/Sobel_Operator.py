import cv2
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt
def main():

    image = cv2.imread("Resim1.tif",0)
    new_image = sobel_operator(image)

    #plt.imshow(new_image,cmap="gray")
    #plt.show()
    cv2.imshow("original",image)
    cv2.imshow("new",new_image)
    cv2.waitKey()

def sobel_operator(image):
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    rows, cols = image.shape
    print("Original Image Dimensions: ", rows, cols)
    new_image = np.zeros((rows,cols),  np.uint8)
    convolved_x = np.zeros((rows, cols))
    convolved_y = np.zeros((rows, cols))
    sa = 1
    size = 3
    gx = np.array([[-1,0,1],#x direction kernel
          [-2,0,2],
          [-1,0,1]])

    gy = np.array([[-1,-2,-1],#y direction kernel
          [0, 0, 0],
          [1, 2, 1]])
    # Kernel Convolution Algorithm
    for i in range(sa, rows - sa):
        for j in range(sa, cols - sa):
            for k in range(size):
                for l in range(size):
                    convolved_x[i][j] += image[i + k - sa][j + l - sa] * gx[k][l]
                    convolved_y[i][j] += image[i + k - sa][j + l - sa] * gy[k][l]
    #Sobel Operator
    convolved_x = np.square(convolved_x)
    convolved_y = np.square(convolved_y)
    gradient_magnitude = np.sqrt(convolved_x+convolved_y)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()#scale values to 0-255
    new_image = np.uint8(gradient_magnitude)

    print(np.amin(new_image),np.amax(new_image))
    return new_image


if __name__ == "__main__":

    main()
