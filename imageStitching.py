import cv2 as cv
import numpy as np
import time
from scipy import linalg, ceil, floor


def bilinear(img, px, py):
    # find the four neighbors's position
    x0 = int(floor(px))
    y0 = int(floor(py))
    x1 = int(ceil(px))
    y1 = int(ceil(py))

    # return black if the position is out of image's range
    if y1 not in range(0, img.shape[0]) or x1 not in range(0, img.shape[1]):
        return 0

    # calculate the new image's pixel BGR value
    b_x0y0, g_x0y0, r_x0y0 = cv.split(img[y0, x0])[0]
    b_x0y1, g_x0y1, r_x0y1 = cv.split(img[y1, x0])[0]
    b_x1y0, g_x1y0, r_x1y0 = cv.split(img[y0, x1])[0]
    b_x1y1, g_x1y1, r_x1y1 = cv.split(img[y1, x1])[0]
    wx0y0 = (x1 - px) * (y1 - py)
    wx0y1 = (x1 - px) * (py - y0)
    wx1y0 = (px - x0) * (y1 - py)
    wx1y1 = (px - x0) * (py - y0)
    b_result = b_x0y0 * wx0y0 + b_x0y1 * wx0y1 + b_x1y0 * wx1y0 + b_x1y1 * wx1y1
    g_result = g_x0y0 * wx0y0 + g_x0y1 * wx0y1 + g_x1y0 * wx1y0 + g_x1y1 * wx1y1
    r_result = r_x0y0 * wx0y0 + r_x0y1 * wx0y1 + r_x1y0 * wx1y0 + r_x1y1 * wx1y1

    result = cv.merge([b_result, g_result, r_result])

    return result


def main():
    start = time.clock()
    # read image
    img1 = cv.imread('source/img1.jpg', cv.IMREAD_COLOR)
    img2 = cv.imread('source/img2.jpg', cv.IMREAD_COLOR)
    # get the image's width and height
    h_img1 = img1.shape[0]
    w_img1 = img1.shape[1]
    h_img2 = img2.shape[0]
    w_img2 = img2.shape[1]

    # calculate the transformation fomula
    matrix_img1 = np.array([[1424, 960, 1], [1249, 394, 1], [1169, 456, 1]])
    matrix_img2 = np.array([[492, 929], [327, 347], [241, 409]])
    [[a, d], [b, e], [c, f]] = linalg.solve(matrix_img1, matrix_img2)

    # calculate the width and height of new image using the above fomula
    matrix_a = np.array([[a, d], [b, e]])
    matrix_b = np.array([w_img2 - c, h_img2 - f])
    w, h = linalg.solve(matrix_a, matrix_b)
    w_newImg = int(w)
    h_newImg = int(h)

    # create an array for new image
    new_img = np.zeros((h_newImg, w_newImg, 3), img1.dtype)
    for i in range(0, h_newImg):
        for j in range(0, w_newImg):
            if i < h_img1 and j < w_img1:
                # get image1's pixel directly when pixel positon within image1
                new_img[i, j] = img1[i, j]
            else:
                # calculate the pixel positon which we need to get BGR at image 2 from the positon of new image
                newI = d * j + e * i + f
                newJ = a * j + b * i + c
                new_img[i, j] = bilinear(img2, newJ, newI)

    cv.imwrite('output/merge.jpg', new_img)
    print "time spent: ", time.clock() - start


if __name__ == '__main__':
    main()
