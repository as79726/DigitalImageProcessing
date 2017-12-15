import cv2 as cv
import numpy as np
import time


# convert value in the specified range
def convert_in_range(value,min_value,max_value):
    if value > max_value:
        return max_value
    if value < min_value:
        return min_value
    return value


# get the neighbors of a pixel position
def get_neighbors(img, y, x):
    lt = img[y - 1, x - 1]      # left top
    ct = img[y - 1, x]          # top
    rt = img[y - 1, x + 1]      # right top
    l = img[y, x - 1]           # left
    c = img[y, x]               # center
    r = img[y, x + 1]           # right
    lb = img[y + 1, x - 1]      # left bottom
    cb = img[y + 1, x]          # bottom
    rb = img[y + 1, x + 1]      # right bottom
    return lt, ct, rt, l, c, r, lb, cb, rb

# convolution function
def convolution(img, y, x, filter):
    lt, ct, rt, l, c, r, lb, cb, rb = get_neighbors(img, y, x)
    bgr = []
    for idx in range(0, 3):
        temp = lt[idx] * filter[0][0] + ct[idx] * filter[0][1] + rt[idx] * filter[0][2] \
                + l[idx] * filter[1][0] + c[idx] * filter[1][1] + r[idx] * filter[1][2] \
                + lb[idx] * filter[2][0] + cb[idx] * filter[2][1] + rb[idx] * filter[2][2]
        temp = convert_in_range(round(temp, 0),0,255)
        bgr.append(temp)
    result = cv.merge([bgr[0], bgr[1], bgr[2]])[0]
    return result


def sobel_filter(img, y, x):
    sobel_x = [
        [-1, -2, -1]
        , [0, 0, 0]
        , [1, 2, 1]
    ]
    sobel_y = [
        [-1, 0, 1]
        , [-2, 0, 2]
        , [-1, 0, 1]
    ]
    lt, ct, rt, l, c, r, lb, cb, rb = get_neighbors(img, y, x)
    bgr = []
    for idx in range(0, 3):
        temp = abs(sobel_y[0][2] * rt[idx] + sobel_y[1][2] * r[idx] + sobel_y[2][2] * rb[idx] \
            + sobel_y[0][0] * lt[idx] + sobel_y[1][0] * l[idx] + sobel_y[2][0] * lb[idx]) \
            + abs(sobel_x[2][0] * lb[idx] + sobel_x[2][1] * cb[idx] + sobel_x[2][2] * rb[idx] \
            + sobel_x[0][0] * lt[idx] + sobel_x[0][1] * ct[idx] + sobel_x[0][2] * rt[idx])
        temp = convert_in_range(temp, 0, 255)
        bgr.append(temp)

    result = cv.merge([bgr[0], bgr[1], bgr[2]])[0]
    return result


def main():
    start = time.clock()
    # read image
    enhance_0 = cv.imread('source/enhance_0.jpg', cv.IMREAD_COLOR)
    height = enhance_0.shape[0]
    width = enhance_0.shape[1]
    print height, width

    # create an array for new image
    enhance_sobel = np.zeros((height, width, 3), enhance_0.dtype)   # sobel
    enhance_blurry = np.zeros((height, width, 3), enhance_0.dtype)   # blurry
    enhance_normalize = np.zeros((height, width, 3), dtype=np.float16)
    enhance_laplacian = np.zeros((height, width, 3), enhance_0.dtype)   # laplacian
    enhance_multiply = np.zeros((height, width, 3), dtype=np.float16)  # edge's detail
    enhance_laplacian_result = np.zeros((height, width, 3), enhance_0.dtype)
    enhance_result = np.zeros((height, width, 3), enhance_0.dtype)

    # define filters
    mean_filter = [
        [1.0/9, 1.0/9, 1.0/9]
        , [1.0/9, 1.0/9, 1.0/9]
        , [1.0/9,1.0/9, 1.0/9]
    ]
    laplacian_filter = [
        [-1, -1, -1]
        , [-1, 8, -1]
        , [-1, -1, -1]
    ]

    # laplacian filtering
    # sobel filtering
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            enhance_laplacian[i, j] = convolution(enhance_0, i, j, laplacian_filter)
            enhance_sobel[i, j] = sobel_filter(enhance_0, i, j)

    # mean filter & normalized
    # multiply laplacian image and normalized image
    # produce the result images
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            enhance_blurry[i, j] = convolution(enhance_sobel, i, j, mean_filter)
            for k in range(0,3):
                enhance_laplacian_result[i, j, k] = convert_in_range(
                    int(enhance_0[i, j, k]) + int(enhance_laplacian[i, j, k]), 0, 255)
                enhance_normalize[i, j, k] = float(enhance_blurry[i, j, k]) / 255
                enhance_multiply[i, j, k] = int(enhance_laplacian[i, j, k]) * float(enhance_normalize[i, j, k])
                enhance_result[i, j, k] = convert_in_range(
                    int(enhance_0[i, j, k]) + float(enhance_multiply[i, j, k]),0, 255)


    cv.imwrite('output/enhance_sobel.jpg', enhance_sobel)
    cv.imwrite('output/enhance_blurry.jpg', enhance_blurry)
    cv.imwrite('output/enhance_laplacian.jpg', enhance_laplacian)
    cv.imwrite('output/enhance_laplacian_result.jpg', enhance_laplacian_result)
    cv.imwrite('output/enhance_result.jpg', enhance_result)
    print "time spent: ", time.clock() - start


if __name__ == '__main__':
    main()
