import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)


    new_image = butterworth_low_pass_filter(image)

    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image, cmap='gray')
    plt.title('New Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def butterworth_low_pass_filter(image):

    size = int(input("Please enter filter cutoff: "))
    order = int(input("Please enter the filter order: "))

    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    center_row, center_col = rows / 2, cols / 2

    filter = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            r2 = float(np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2))
            if r2 == 0:
                r2 = 1.0
            filter[i][j] = 1/(1+(r2/size)**order)

    dft_shift = dft_shift * filter
    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back

if __name__ == "__main__":

    main()