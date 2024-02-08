# SNN Testbench
# Author: Hitesh Ahuja and Rajkumar Kubendran
import math
import os

import numpy as np
import cv2
import time
import ok
import struct


def buffered_capture(u8image, ulLen):
    i = 0
    length = 0
    dev.ActivateTriggerIn(0x40, 0)  # Readout start trigger

    dev.UpdateWireOuts()

    i = 0
    while not (dev.GetWireOutValue(0x23) & 0x0100):
        dev.UpdateWireOuts()
        i = i + 1

    print(i)

    length = dev.ReadFromPipeOut(0xA0, u8image)
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
    dev.ActivateTriggerIn(0x42, 0)
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
    # Initialize Opal Kelly
    dev = ok.okCFrontPanel()

    # Load FPGA program
    dev.OpenBySerial("")
    error = dev.ConfigureFPGA(
        "L1/evb1005.bit")  # C:/Users/Rajkumar/PycharmProjects/qDVS_export/qDVS.bit"   C:/Users/BEH1123_Yuyang/Desktop/EVB1005_DVS_FPGA/EVB1005_DVS/EVB1005_DVS.runs/impl_1
    if error:
        print(f"Error with code {error}")
        exit(1)
    else:
        print(" FPGA configured")

    dev.SetWireInValue(0x06, 0x1)
    dev.UpdateWireIns()
    time.sleep(0.1)
    dev.SetWireInValue(0x06, 0x0)
    dev.UpdateWireIns()

    dev.SetWireInValue(0x08, 0x0A)
    dev.UpdateWireIns()
    if os.path.isfile("./center_log.txt"):
        os.remove("./center_log.txt")

    for i in range(52, 70):
        print(f"#{i}")
        write_image = bytearray(256 * 256)
        write_image_origin = np.loadtxt(f"./saved_new_large/img_new{i}.txt")
        write_image_origin = write_image_origin * 32
        # write_image_origin = np.transpose(write_image_origin)
        # if i % 2 == 0:
        #     write_image_origin = cv2.rotate(write_image_origin, cv2.ROTATE_180)
        # write_image_origin = cv2.flip(write_image_origin, 0)
        write_image = np.reshape(write_image_origin, 65536).astype(np.uint8)
        # cv2.imshow("write_image_before_write", write_image_origin)
        # time.sleep(0.5)

        dev.SetWireInValue(0x06, 0x1)
        dev.UpdateWireIns()
        time.sleep(0.1)
        dev.SetWireInValue(0x06, 0x0)
        dev.UpdateWireIns()

        dev.WriteToPipeIn(0x80, write_image)

        dev.UpdateWireOuts()

        # dev.SetWireInValue(0x07, 0x1)
        # dev.UpdateWireIns()
        #
        # time.sleep(0.1)
        # dev.SetWireInValue(0x07, 0x0)
        # dev.UpdateWireIns()

        dev.ActivateTriggerIn(0x43, 0)

        dev.UpdateWireOuts()

        clk_counter = 0
        wire_out_old = 0
        wire_out = 0
        t_0 = time.perf_counter()
        max_ltr = 0
        while not (dev.GetWireOutValue(0x23) & 0x0100):
            dev.UpdateWireOuts()
            wire_out_old = wire_out
            wire_out = dev.GetWireOutValue(0x23)
            if wire_out != wire_out_old:
                t_1 = time.perf_counter()
                print(f"time = {t_1 - t_0:02.4f};    wire out 0x23 = {wire_out:012b}")

        dev.UpdateWireOuts()
        time.sleep(0.001)
        center = dev.GetWireOutValue(0x22)
        # max_ltr = dev.GetWireOutValue(0x3D)
        # rd_data_cnt = dev.GetWireOutValue(0x2E)
        # print(f"max_ltr={max_ltr:04x}, rd_data_cnt = {rd_data_cnt:08x}")
        # print(f"after processing: {w_count:x}")
        read_ptl0 = bytearray(4 * 8192)
        read_ptl1 = bytearray(4 * 8192)
        length = dev.ReadFromPipeOut(0xA0, read_ptl0)
        length = dev.ReadFromPipeOut(0xA1, read_ptl1)

        # cv2.imshow("write_image", np.reshape(np.multiply(write_image, 8), (256, 256)))
        # cv2.imshow("read_image", np.reshape(np.multiply(read_image, 8), (256, 256)))

        # cv2.imshow("write_image", np.reshape(write_image, (256, 256)))
        # time.sleep(0.5)
        # cv2.imshow("read_image", np.reshape(read_image, (256, 256)))
        # time.sleep(0.5)

        with open(f'ptl_0_{i}.txt', 'w') as f:
            np.savetxt(f, read_ptl0, fmt='%2x')
        with open(f'ptl_1_{i}.txt', 'w') as f:
            np.savetxt(f, read_ptl1, fmt='%2x')
        with open(f'center_log.txt', 'a') as f:
            np.savetxt(f, np.reshape(center, (1, 1)), fmt='%08x')
        # print(read_image)

    # while 1:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
