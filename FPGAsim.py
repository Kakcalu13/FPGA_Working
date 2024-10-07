# forever test -- ideal data

import traceback
from time import sleep
import time
import threading
import serial
import numpy as np
from datetime import datetime
from feagi_connector import feagi_interface as feagi
from feagi_connector import sensors
from feagi_connector import pns_gateway as pns
from feagi_connector.version import __version__
from feagi_connector import actuators
import ok

steps = [[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]]

def feagi_to_petoi_id(device_id):
    mapping = {
        0: 8,
        1: 12,
        2: 9,
        3: 13,
        4: 11,
        5: 15,
        6: 10,
        7: 14
    }
    return mapping.get(device_id, None)

def action(obtained_data):
    try:
        # fpga here section:
        servo_for_feagi = 'i '
        print("full raw data: ", obtained_data)
        for servo_id in range(0, len(obtained_data), 2):
            mapped_id = feagi_to_petoi_id(servo_id // 2)
            value1, value2 = obtained_data[servo_id], obtained_data[servo_id + 1]
            turn = 0
            if mapped_id in [8, 9, 10, 11]:
                turn = 15
            if mapped_id in [12, 13, 14, 15]:
                turn = 30
            if value1 == 1:
                # print(servo_id, value1)
                servo_status[servo_id // 2] -= turn
            if value2 == 1:
                # print(servo_id+1, value2)
                servo_status[servo_id // 2] += turn
            servo_status[servo_id // 2] = actuators.servo_keep_boundaries(servo_status[servo_id // 2], 90, -90) # block from exceeded 90
            # Append the mapped ID and the adjusted status to the result string
            servo_for_feagi += str(mapped_id) + " " + str(servo_status[servo_id // 2]) + " "
        # print("final: ", servo_for_feagi.encode())
        print("final: ", servo_status)
        ser.write(servo_for_feagi.encode())

    except KeyboardInterrupt:
        ser.write('kbalance'.encode())
        ser.close()
        print("keyboard interrupt".capitalize())
        exit()

servo_status = [30,	30,	30,	30,	30,	30,	30,	30]
gyro = {}
count = 0x02000000
clk_f = 200E6
tc = int(count) / clk_f
print("Time Constant: ", tc)

try:
    print("Connecting to Bot...")
    ser = serial.Serial('COM5', 115200)
    time.sleep(5)
    ser.write('i 0 4 1 -1 2 3 3 0 4 0 5 0 6 0 7 0 8 29 9 31 10 31 11 29 12 25 13 23 14 26 15 25'.encode())
    idx = 0
    while True:
        action(steps[idx])
        idx += 1
        if idx > 15:
            idx = 0
        time.sleep(tc)  # to emulate the clock cycle of FPGA
        # print(ser.reset_input_buffer())
except Exception as error:
    print(error)
    ser.close()
    exit()