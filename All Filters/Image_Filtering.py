import cv2
import numpy as np
import math
from statistics import median
from scipy import ndimage
import matplotlib.pyplot as plt

def main():

    filter_type = int(input("""Please enter a number for the filter type given below: "
                        "1: Arithmetic Mean Filter"
                        "2: Geometric Mean Filter"
                        "3: Harmonic Mean Filter"
                        "4: Contraharmonic Mean Filter"
                        "5: Median Filter"
                        "6: Max Filter"
                        "7: Min Filter"
                        "8: Mid-point Filter"
                        "9: Laplacian Filter"
                        "10: Sobel Operator"
                        "11: Ideal Low-Pass Filter"
                        "12: Ideal High-Pass Filter"
                        "13: Butterworth Low-Pass Filter"
                        "14: Butterworth High-Pass"
                        "15: Gaussian Low-Pass Filter"
                        "16: Gaussian High-Pass Filter: """))
    new_image = choose(filter_type)
    plt.subplot(122), plt.imshow(new_image, cmap='gray')
    plt.title('New Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def choose(filter_type):
    if filter_type == 1:
        new_image = mean_filter()
    elif filter_type == 2:
        new_image = geometric_mean_filter()
    elif filter_type == 3:
        new_image = harmonic_mean_filter()
    elif filter_type == 4:
        new_image = contraharmonic_mean_filter()
    elif filter_type == 5:
        new_image = median_filter()
    elif filter_type == 6:
        new_image = max_filter()
    elif filter_type == 7:
        new_image = min_filter()
    elif filter_type == 8:
        new_image = midpoint_filter()
    elif filter_type == 9:
        new_image = laplacian_filter()
    elif filter_type == 10:
        new_image = sobel_operator()
    elif filter_type == 11:
        new_image = low_pass_filter()
    elif filter_type == 12:
        new_image = high_pass_filter()
    elif filter_type == 13:
        new_image = butterworth_low_pass_filter()
    elif filter_type == 14:
        new_image = butterworth_high_pass_filter()
    elif filter_type == 15:
        new_image = gaussian_low_pass_filter()
    elif filter_type == 16:
        new_image = gaussian_high_pass_filter()

    return new_image

def mean_filter(): #Smoothing image

    image = cv2.imread("lenaa.png",0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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
    for i in range(size-2):
        for j in range(0,cols):
            new_image[i][j] = image[i][j]
            new_image[rows-i-1][j] = image[rows-i-1][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-1] = image[i][cols - j-1]

    return new_image

def median_filter():
    image = cv2.imread("lenaa.png", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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
            new_image[rows-i-value_size][j] = image[rows-i-value_size][j]
    for j in range(size-2):
        for i in range(0,rows):
            new_image[i][j] = image[i][j]
            new_image[i][cols - j-value_size] = image[i][cols - j-value_size]

    return new_image

def geometric_mean_filter(): #Smoothing image

    image = cv2.imread("Fig0507(a)(ckt-board-orig).tif", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)
    new_image = np.ones((rows,cols))
    sa = int(np.floor(size / 2))
    for i in range(sa,rows-sa):
        for j in range(sa,cols-sa):
            for k in range(size):
                for l in range(size):
                    new_image[i][j] *= np.float((image[i+k-sa][j+l-sa])**(1/(size*size)))

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
    new_image = np.uint8(new_image)
    print(np.amin(new_image),np.amax(new_image))
    return new_image

def harmonic_mean_filter(): #Smoothing image

    image = cv2.imread("Fig0507(a)(ckt-board-orig).tif", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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

def contraharmonic_mean_filter(): #Smoothing image

    image = cv2.imread("circuit-board-pepper.tif",0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    q = float(input("Please enter parameter Q(1.5 for pepper noise): "))
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

def max_filter():

    image = cv2.imread("circuit-board-pepper.tif", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    filter = []
    rows, cols = image.shape
    new_image = np.zeros((rows,cols),np.uint8)
    value_size = int(np.floor(size / 2))
    for i in range(value_size,rows-value_size):
        for j in range(value_size,cols-value_size):
            for k in range(size):
                for l in range(size):

                    filter.append(image[i+k-value_size][j+l-value_size])

            new_image[i][j] = np.round(np.amax(filter))
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

    return new_image

def min_filter():

    image = cv2.imread("circuit-board-salt-prob.tif",0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    filter = []
    rows, cols = image.shape
    new_image = np.zeros((rows,cols),np.uint8)
    value_size = int(np.floor(size / 2))
    for i in range(value_size,rows-value_size):
        for j in range(value_size,cols-value_size):
            for k in range(size):
                for l in range(size):

                    filter.append(image[i+k-value_size][j+l-value_size])

            new_image[i][j] = np.round(np.amin(filter))
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

    return new_image


def midpoint_filter():

    image = cv2.imread("uniform.tif",0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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

def laplacian_filter(): #Smoothing image

    image = cv2.imread("Blurred_moon.tif",0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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

def sobel_operator():

    image = cv2.imread("Resim1.tif",0)
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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

def low_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    center_row, center_col = rows / 2, cols / 2
    filter = np.zeros((rows,cols),np.uint8)
    filter[int(center_row) - size:int(center_row) + size, int(center_col) - size:int(center_col) + size] = 1
    dft_shift = dft_shift * filter
    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back

def high_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    size = int(input("Please enter filter size: "))
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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

def butterworth_low_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

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

def butterworth_high_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
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
            filter[i][j] = 1/(1+(size/r2)**order)

    dft_shift = dft_shift * filter
    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back

def gaussian_low_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    size = int(input("Please enter filter cutoff: "))

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
            filter[i][j] = math.exp((-(r2**2))/(2*(size**2)))

    dft_shift = dft_shift * filter
    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back

def gaussian_high_pass_filter():

    image = cv2.imread("Fig0441(a)(characters_test_pattern).tif", 0)
    plt.subplot(121), plt.imshow(image, cmap="gray")
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    size = int(input("Please enter filter cutoff: "))


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
            filter[i][j] = 1-math.exp((-(r2**2))/(2*(size**2)))

    dft_shift = dft_shift * filter
    f_ishift = np.fft.ifftshift(dft_shift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    return image_back





if __name__ == "__main__":

    main()