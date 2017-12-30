import cv2 as cv
import numpy as np
import time

#random value to decide pixel value
def generate_impulse():
    value = np.random.uniform(0, 1)
    if value <= 0.25:
        return 0
    elif value > 0.25 and value <= 0.5:
        return 255
    else:
        return None

#find all pixel within the specific window size
def get_neighbors(layer, img, i, j):
    arr = []
    for a in range(i-layer, i+layer+1):
        for b in range(j-layer,j+layer+1):
            arr.append(img[a][b])
    arr.sort()
    return arr


#adaptive median filter
def adaptive_median_filter(threshold, img, i, j):
    for layer in range(1,threshold+1):
        arr = get_neighbors(layer, img, i, j)
        zxy = img[i][j]
        zmed = arr[len(arr)/2]
        zmin = arr[0]
        zmax = arr[len(arr)-1]
        if zmed > zmin and zmed < zmax:
            if zxy > zmin and zxy < zmax:
                return zxy
            else:
                return zmed

    # if the pixel value of Zxy is extreme value, return Zmed may help getting better performance
    # if (zxy == zmin or zxy == zmax) and (zxy == 0 or zxy == 255):
    #     return zmed

    return zxy


def median_filter(win_size, img, i, j):
    arr = []
    layer = (win_size-1)/2
    for a in range(i-layer,i+layer+1):
        for b in range(j-layer, j+layer+1):
            arr.append(img[a][b])
    arr.sort()
    return arr[len(arr)/2]


def main():
    start = time.clock()
    # read image
    original_img = cv.imread('source/original_color.jpg', cv.IMREAD_GRAYSCALE)
    height = original_img.shape[0]
    width = original_img.shape[1]
    print height, width
    impulse_img = np.zeros((height, width), original_img.dtype)
    mf_img = np.zeros((height,width), original_img.dtype)
    amf_img = np.zeros((height, width), original_img.dtype)

    #generate impulse noise
    for i in range(0, height):
        for j in range(0, width):
            impulse = generate_impulse()
            if impulse == None:
                impulse_img[i][j] = original_img[i][j]
            else:
                impulse_img[i][j] = impulse

    #deconvolution by median filter
    for i in range(1,height-1):
        for j in range(1, width-1):
            mf_img[i][j] = median_filter(3, impulse_img, i, j)

    #deconvolution by adaptive median filter
    for i in range(3, height-3):
        for j in range(3, width-3):
            amf_img[i][j] = adaptive_median_filter(3,impulse_img, i, j)

    cv.imwrite('output/original_gray.jpg', original_img)
    cv.imwrite('output/impulse_img.jpg', impulse_img)
    cv.imwrite('output/result_median_filter.jpg', mf_img)
    cv.imwrite('output/result_adaptive_median_filter.jpg', amf_img)
    print "time spent: ", time.clock() - start

if __name__ == '__main__':
    main()
