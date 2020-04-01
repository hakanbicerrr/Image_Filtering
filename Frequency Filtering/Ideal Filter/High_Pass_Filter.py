import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    size = int(input("Please enter filter size: "))

    new_image = high_pass_filter(image,size)

    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image, cmap='gray')
    plt.title('New Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def high_pass_filter(image,size):

    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    center_row, center_col = rows / 2, cols / 2

    dft_shift[int(center_row) - size:int(center_row) + size, int(center_col) - size:int(center_col) + size] = 0

    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back

if __name__ == "__main__":

    main()