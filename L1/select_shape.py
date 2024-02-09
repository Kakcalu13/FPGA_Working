import time

import cv2
import numpy as np
from scipy import ndimage
import math

ccx = np.zeros(50)  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ccy = np.zeros(50)  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
size = (256, 256)
iterv = 0
cttr = 0


def reduce(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def calcdist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def selectByColor(frame, Xpt, Ypt):
    '''hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)'''

    gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = gray.astype(int)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    cc = (ndimage.convolve(gray, k, mode='nearest')) / 8
    diff = gray - cc
    mask = cv2.inRange(diff, -5, 5)
    mask = cv2.bitwise_not(mask)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # print(contours)
    distances = []
    dist_dict = {}
    center_dict = {}
    # min_dist=10000
    # print(contours[0][0][0][0])
    if Xpt and Ypt:
        for i in contours:
            M = cv2.moments(i)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                dd = calcdist(Xpt, Ypt, center_x, center_y)
                distances.append(dd)
                dist_dict[dd] = i
                center_dict[dd] = [center_x, center_y]
        big_contour = dist_dict[min(distances)]
        Xpt = center_dict[min(distances)][0]
        Ypt = center_dict[min(distances)][1]

    ##        if Xpt and Ypt:
    ##            xt=(i[0]-Xpt[0])*(i[0]-Xpt[0])
    ##            yt=(i[1]-Ypt[0])*(i[1]-Ypt[0])
    ##            dd=np.sqrt(xt+yt)
    ##            distances.append(dd)
    ##            if dd<min_dist:
    ##                big_contour=0#[[i[0],i[1]]]

    # big_contour = max(contours, key=cv2.contourArea)
    # print(big_contour)
    global iterv
    M = cv2.moments(big_contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    # if (center_x not in ccx) and (center_y not in ccy) :
    # ccx.append(int(M["m10"]/M["m00"]))
    # ccx.append(0)
    # ccy.append(0)
    ccx[iterv] = int(M["m10"] / M["m00"])
    ccy[iterv] = int(M["m01"] / M["m00"])
    iterv = iterv + 1
    # ccy.append(int(M["m01"]/M["m00"]))
    x, y, w, h = cv2.boundingRect(big_contour)
    mask_1 = np.zeros_like(mask)
    cv2.drawContours(mask_1, [big_contour], 0, (255, 255, 255), -1)
    # print(ccx)

    # put mask into alpha channel of input
    # new_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    new_image = np.zeros_like(frame)
    new_image[:, :, 0] = mask_1
    new_image[:, :, 1] = mask_1
    new_image[:, :, 2] = mask_1

    if iterv > 48:
        iterv = 0

    # for a, b in zip(ccx, ccy):
    #     # mask_1[a,b] = 127
    #     a = int(a)
    #     b = int(b)
    #     cv2.circle(new_image, (a, b), 5, (0, 255, 0), -1)

    res = reduce(mask_1, size)

    return res, new_image


if __name__ == '__main__':
    for i in range(1, 1000):
        fname = f"C:/Users/BEH1123_Yuyang/OneDrive - University of Pittsburgh/Research/Circuits/DVS/matlab/shapes_rotation/images/frame_{i:08d}.png"
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        cv2.imshow("image", image)
        Xpt = 214
        Ypt = 134
        [res, new_image] = selectByColor(image, Xpt, Ypt)
        cv2.imshow("new_image", new_image)
        # time.sleep(0.5)
        cv2.waitKey(50)
