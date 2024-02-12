# SNN Testbench
# Author: Hitesh Ahuja and Rajkumar Kubendran
import math
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from numpy import double
# from scipy.optimize import curve_fit
# import struct
import time
import ok
# import os
# import csv
# import random as rn
# from scipy.sparse.csc import csc_matrix
# # import more_itertools as mit
# from scipy.ndimage import convolve


# if ((m_nHDLVersion & 0xFF00) >= 0x0200) {
# 		SetImageBufferDepth(IMAGE_BUFFER_DEPTH_AUTO);
# 	} setting image buffer depth
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

    print(i)
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


# Initialize Opal Kelly
dev = ok.okCFrontPanel()

# Load FPGA program
dev.OpenBySerial("")
error = dev.ConfigureFPGA(
    "C:/Users/ereij/OneDrive/Documents/PITT/FPGA/L1/evb1005.bit")   # "C:/Users/ereij/OneDrive/Documents/PITT/FPGA/L1/evb1005.bit"
print(error)

dev.SetWireInValue(0x04, 0xfff)

dev.UpdateWireOuts()

m_nHDLVersion = dev.GetWireOutValue(0x3f)
print("Version is ", m_nHDLVersion)
m_nHDLCapability = dev.GetWireOutValue(0x3e)

dev.SetWireInValue(0x06, 0x01)
dev.UpdateWireIns()
SetResolution(2399, 1799)
SetSkip(2, 2)
# SetShutterWidth(2000)

while True:
    t_0 = time.perf_counter()
    u8image = bytearray(256 * 256)
    final_image = bytearray(256 * 256)
    ulLen = 65536
    final_image = buffered_capture(u8image, ulLen)
    # final_image = np.array(list(final_image))
    # # print(np.unique(final_image))
    # final_image = final_image.astype(np.uint8)
    # print(np.shape(np.reshape(final_image,(256,256))))

    # print(final_image)
    cv2.imshow("frame", np.reshape(final_image, (256, 256)))
    # with open('image_test.txt', 'w') as fp:
    #     for i in np.reshape(final_image, (65536, 1)):
    #         np.savetxt(fp, i, fmt='%.8e')
    # while 1:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('c'):
    #         break
    # while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t_1 = time.perf_counter()
    print(f"single frame time: {t_1 - t_0:0.4f}")
