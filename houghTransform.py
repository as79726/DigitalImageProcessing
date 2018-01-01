import cv2 as cv
import numpy as np
import time


def hough_transform(x, y, angle):
    return x * np.cos(angle) + y * np.sin(angle)


def main():
    start = time.clock()
    # read image
    original_img = cv.imread('source/hough_original.jpg', cv.IMREAD_GRAYSCALE)
    # for result
    result = cv.imread('source/hough_original.jpg', cv.IMREAD_COLOR)

    height = original_img.shape[0]
    width = original_img.shape[1]
    print height, width

    canny_result = cv.Canny(original_img, 50, 150)
    cv.imwrite('output/canny_result.jpg', canny_result)

    # generate an array
    # convert degree to radian
    thetas = np.deg2rad(np.arange(-180.0, 180.0, 0.12))

    min_rho = 0
    max_rho = 0

    # mark a 3-dimension matrix to store every theta's result in axis-x and axis-y
    pos_rho_list = []
    for i in range(height):  # axis-y
        pos_rho_list.append([])
        for j in range(width):  # axis-x
            rhos = []
            if canny_result[i][j] != 0:
                for theta in thetas:  # thetas
                    rho = hough_transform(j, i, theta)
                    if rho < min_rho:
                        min_rho = rho
                    if rho > max_rho:
                        max_rho = rho
                    rhos.append(rho)
                pos_rho_list[i].append(rhos)
            else:
                pos_rho_list[i].append([])

    hough_img_height = int(max_rho - min_rho) + 1
    hough_img_width = len(thetas)
    hough_img = np.zeros((hough_img_height, hough_img_width), np.uint8)
    co_occurs_max = 0
    for i in range(len(pos_rho_list)):
        rows = pos_rho_list[i]
        for j in range(len(rows)):
            datas = rows[j]
            for tid in range(len(datas)):
                p = int(datas[tid] - min_rho)
                hough_img[p][tid] += 1
                if hough_img[p][tid] >= co_occurs_max:
                    co_occurs_max = hough_img[p][tid]
                    co_occurs_pos = (p, tid)

    for i in range(hough_img_height):
        for j in range(hough_img_width):
            hough_img[i][j] = int((float(hough_img[i][j]) / co_occurs_max) * 255)

    cv.imwrite('output/hough.jpg', hough_img)

    pos = []
    for i in range(len(pos_rho_list)):
        rows = pos_rho_list[i]
        for j in range(len(rows)):
            datas = rows[j]
            if len(datas) > 0:
                if int(datas[co_occurs_pos[1]] - min_rho) == co_occurs_pos[0]:
                    pos.append((j, i))

    cv.line(result, pos[0], pos[len(pos) - 1], (0, 0, 255), 2)

    cv.imwrite('output/hough_result.jpg', result)
    print "time spent: ", time.clock() - start


if __name__ == '__main__':
    main()
