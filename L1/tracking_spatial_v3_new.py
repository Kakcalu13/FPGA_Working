import cv2
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import ndimage
import math
import time
import ok

import vanilla_dnf as dnf

# TODO : Pour installer opencv:
# sudo apt-get install opencv* python3-opencv

# Si vous avez des problèmes de performances
#
# self.kernel = np.zeros([width * 2, height * 2], dtype=float)
# for i in range(width * 2):
#     for j in range(height * 2):
#         d = np.sqrt(((i / (width * 2) - 0.5) ** 2 + ((j / (height * 2) - 0.5) ** 2))) / np.sqrt(0.5)
#         self.kernel[i, j] = self.difference_of_gaussian(d)
#
#
# Le tableau de poids latéreaux est calculé de la façon suivante :
# self.lateral = signal.fftconvolve(self.potentials, self.kernel, mode='same')

size = (256, 256)
# size=(480,640)
# size=(120,160)
# size=(3,4)
ccx = np.zeros(50)  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ccy = np.zeros(50)  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Xpt = []
Ypt = []
iterv = 0
cttr = 0


def calcdist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def click_and_extract_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        Xpt.append(x)
        Ypt.append(y)
        print(Xpt, Ypt)


# size=(96,128)


def reduce(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def selectByColor(frame):
    '''hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)'''

    # gray =(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = frame
    gray = gray.astype(int)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    cc = (ndimage.convolve(gray, k, mode='nearest')) / 8
    diff = gray - cc
    cv2.imshow("diff", diff)
    # diff=gray
    mask = cv2.inRange(diff, -5, 5)
    mask = cv2.bitwise_not(mask)
    # mask = diff
    cv2.imshow("Spatial Contrast", mask)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # big_contour = max(contours, key=cv2.contourArea)
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
                dd = calcdist(Xpt[0], Ypt[0], center_x, center_y)
                distances.append(dd)
                dist_dict[dd] = i
                center_dict[dd] = [center_x, center_y]
        big_contour = dist_dict[min(distances)]
        Xpt[0] = center_dict[min(distances)][0]
        Ypt[0] = center_dict[min(distances)][1]

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
    new_image = np.zeros((256, 256, 3))
    new_image[:, :, 0] = mask_1
    new_image[:, :, 1] = mask_1
    new_image[:, :, 2] = mask_1

    if iterv > 48:
        iterv = 0

    for a, b in zip(ccx, ccy):
        # mask_1[a,b] = 127
        a = int(a)
        b = int(b)
        cv2.circle(new_image, (a, b), 5, (0, 255, 0), -1)

    res = reduce(mask_1, size)
    cv2.imshow("res", res)
    cv2.imshow("nimg", new_image)
    return res, new_image


def findCenter(modele):
    # on fait évoluer le modele en espérant qu'il se focalisera sur la tasse
    # modele.update_map()

    # on extrait le centre de la boule ainsi générée
    """index = np.argmax(modele.potentials)
    y = index // modele.potentials.shape[0]
    x = index % modele.potentials.shape[0]"""
    x, y = ndimage.center_of_mass(modele.potentials)
    if np.isnan(x):
        x = 0
    else:
        x = int(x)
    if np.isnan(y):
        y = 0
    else:
        y = int(y)
    # ccx.append(x)
    # ccy.append(y)
    # print(ccx)

    return x, y


def track(frame, modele, i):
    # input, nimg = selectByColor(frame)
    # t_sel_color = time.perf_counter()
    # print(f"time for select by color: {t_sel_color - t_vc:0.4f}")
    # tester selectByColor
    input = frame
    modele.input = input
    modele.update_map(i)
    # t_update_map = time.perf_counter()
    # print(f"time for update map: {t_update_map - t_sel_color:0.4f}")
    # cv2.imshow("Input", modele.input)
    cv2.imshow("Potentials", modele.potentials)
    global cttr
    ff = "Hexagon\\zz1zztrack_" + str(cttr) + ".bmp"
    # cv2.imwrite(ff,nimg)
    cttr += 1
    # cv2.imshow("Visual Tracking",nimg)
    center = findCenter(modele)
    # modele.update_kernel(center)
    centered_frame = np.zeros((size[0], size[1], 3))
    centered_frame[:, :, 0] = modele.potentials
    centered_frame[:, :, 1] = modele.potentials
    centered_frame[:, :, 2] = modele.potentials
    cv2.circle(centered_frame, (center[1], center[0]), 5, (0, 0, 255), -1)

    cv2.imshow("Potentials_Center", centered_frame)
    # t_find_center = time.perf_counter()
    # print(f"time for find center: {t_find_center - t_update_map:0.4f}")
    # motorControl(center)


def buffered_capture(u8image, ulLen):
    i = 0
    length = 0
    dev.ActivateTriggerIn(0x40, 0)  # Readout start trigger

    dev.UpdateWireOuts()

    # for i in range(201):
    #     if(dev.GetWireOutValue(0x23) & 0x0100):
    #         # print("get wire out value", dev.GetWireOutValue(0x23))
    #         break
    #
    #     time.sleep(0.0001)
    #     dev.UpdateWireOuts()
    # # print(i)
    # if(200 == i):
    #     print("timeout")
    i = 0
    while not (dev.GetWireOutValue(0x23) & 0x0100):
        dev.UpdateWireOuts()
        i = i + 1

    # print(i)
    # Frame ready (buffer A)
    # dev.SetWireInValue(0x04, 0x0000)
    # dev.SetWireInValue(0x05, 0x0000)
    # dev.UpdateWireIns()

    length = dev.ReadFromPipeOut(0xA0, u8image)
    # print(length)

    # if (length < 0):
    #   print("Image Read Out Error")
    #    return
    # if (length != ulLen):
    #    print("Image Read Out Short")
    #    return
    # dev.ActivateTriggerIn(0x40, 1);  # Readout done (buffer A)
    return u8image


def I2CWrite8(addr, data):
    dev.ActivateTriggerIn(0x42, 1)
    # Num of Data Words
    dev.SetWireInValue(0x01, 0x0030, 0x00ff)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x42, 2)
    # Device Address
    dev.SetWireInValue(0x01, 0xBA, 0x00ff)  # 0xBA for MT9P031
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x42, 2)
    # Register Address
    dev.SetWireInValue(0x01, addr, 0x00ff)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x42, 2)
    # Data 0 MSB
    dev.SetWireInValue(0x01, data >> 8, 0x00ff)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x42, 2)
    # Data 1 LSB
    dev.SetWireInValue(0x01, data, 0x00ff)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x42, 2)
    # Start I2C Transaction
    dev.ActivateTriggerIn(0x42, 0);
    # Wait for Transaction to Finish
    for i in range(50):
        dev.UpdateTriggerOuts()
        if 0 == dev.IsTriggered(0x61, 0x0001):
            break


def SetShutterWidth(shutter):
    I2CWrite8(0x08, (shutter & 0xffff0000) >> 16)
    I2CWrite8(0x09, shutter & 0xffff)


def SetResolution(horizontal, vertical):
    I2CWrite8(0x04, horizontal)
    I2CWrite8(0x03, vertical)


def SetSkip(col, row):
    I2CWrite8(0x22, col)
    I2CWrite8(0x23, row)


if __name__ == '__main__':
    # cv2.namedWindow("Camera")
    # vc = cv2.VideoCapture(0) # 2 pour la caméra sur moteur, 0 pour tester sur la votre.
    dev = ok.okCFrontPanel()

    # Load FPGA program
    dev.OpenBySerial("")
    error = dev.ConfigureFPGA(
        "./evb1005.bit")  # C:/Users/Rajkumar/PycharmProjects/qDVS_export/qDVS.bit"   C:/Users/BEH1123_Yuyang/Desktop/EVB1005_DVS_FPGA/EVB1005_DVS/EVB1005_DVS.runs/impl_1
    print(error)

    dev.SetWireInValue(0x04, 0xfff)

    dev.SetWireInValue(0x06, 0x01)
    dev.UpdateWireIns()

    dev.ActivateTriggerIn(0x40, 0)  # Readout start trigger
    while not (dev.GetWireOutValue(0x23) & 0x0100):
        dev.UpdateWireOuts()

    dev.UpdateWireOuts()

    m_nHDLVersion = dev.GetWireOutValue(0x3f);
    print("Version is ", m_nHDLVersion)
    m_nHDLCapability = dev.GetWireOutValue(0x3e);

    dev.SetWireInValue(0x06, 0x01)
    dev.UpdateWireIns()
    SetResolution(2399, 1799)
    SetSkip(2, 2)
    SetShutterWidth(2000)

    u8image = bytearray(256 * 256)
    final_image = bytearray(256 * 256)
    ulLen = 65536
    final_image = buffered_capture(u8image, ulLen)
    final_image = np.array(list(final_image))
    final_image = final_image.astype(np.uint8)
    final_image = np.reshape(final_image, (256, 256))
    res = reduce(final_image, size)
    res = res.astype(np.uint8)

    cv2.namedWindow('Select Focus')
    cv2.setMouseCallback('Select Focus', click_and_extract_points)
    cv2.imshow('Select Focus', res)

    clone = final_image.copy()

    dev.SetWireInValue(0x06, 0x00)
    dev.UpdateWireIns()
    while 1:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):  # if mouse clicks done incorrectly, press 'r'. Resets arrays and map image.
            Xpt = []
            Ypt = []
            image = clone.copy()

        elif key == ord('c'):
            break
    #     # initialisez votre modele ici
    modele = dnf.DNF(size[0], size[1], Xpt[0], Ypt[0])
    #     # with open('kernel.txt','w') as f:
    #     #     for i in modele.kernel:
    #     #         np.savetxt(f, i, fmt='%.8e')
    #     # print(modele.kernel)
    #
    i = 0
    while 1:
        cv2.imshow("Camera", final_image)
        final_image = buffered_capture(u8image, ulLen)
        final_image = np.array(list(final_image))
        final_image = final_image.astype(np.uint8)
        final_image = np.reshape(final_image, (256, 256))
        res = reduce(final_image, size)
        res = np.subtract(np.round(np.divide(res, 128)), 1)
        res = np.abs(res)
        cv2.imshow("absolute_camera", res)

        track(res, modele, i)

        i = i + 1
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
