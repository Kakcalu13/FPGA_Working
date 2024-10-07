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
# i 0 4 1 -1 2 3 3 0 4 0 5 0 6 0 7 0 8 29 9 31 10 31 11 29 12 25 13 23 14 26 15 25
servo_status = [30,	30,	30,	30,	30,	30,	30,	30]
gyro = {}

# set timing
global count
count = 0x02000000
clk_f = 200E6
tc = int(count) / clk_f
# print(int(count), tc)

# Program FPGA
try:
    print("Programming FPGA...")
    print("\tTime Constant: ", str(tc))
    dev = ok.okCFrontPanel()
    dev.OpenBySerial("")
    error = dev.ConfigureFPGA(r"C:\Users\ereij\OneDrive\Documents\PITT\FPGA\qDVS\CPG_OK.bit")

    print("Connecting to Bot...")
    ser = serial.Serial('COM5', 115200)
    time.sleep(5)
    ser.write('i 0 4 1 -1 2 3 3 0 4 0 5 0 6 0 7 0 8 29 9 31 10 31 11 29 12 25 13 23 14 26 15 25'.encode())

    print("Starting now!")
except Exception as Error_case:
    print(Error_case)
    ser.write('kbalance'.encode())
    dev.Close()
    ser.close()
    exit()

def simulation_from_fpga():
    return [np.random.choice([0, 1]) for _ in range(16)]


def bitfield(n, length = 16):
    bf = [int(digit) for digit in bin(n)[2:]]
    for i in range(length - len(bf)):
        bf = [0] + bf
    return bf
# Function to handle receiving data
def read_from_port(ser='', dev = ''):
    global received_data, gyro, count
    # received_data = 0
    full_data = ''

    try:
        # Reset chip
        dev.SetWireInValue(0x01, count)  # sets count
        dev.SetWireInValue(0x00, 0b0010)  # pull both resets and spike in down
        dev.UpdateWireIns()

        time.sleep(tc)
        dev.SetWireInValue(0x00, 0b1000)  # pull clk_rst_n high

        time.sleep(tc * 10)
        dev.SetWireInValue(0x00, 0b1101)  # pull rst_n high and start high
        dev.UpdateWireIns()

        time.sleep(tc * 1.6)
        dev.SetWireInValue(0x00, 0b1100)  # pull start low
        dev.UpdateWireIns()

        old_data = 0
        t_0 = time.time()
        step_i = 0
        print("-------------------------------------------------------------")
        print("count: ", hex(count))
        while count >= 0x00400000:
            dev.SetWireInValue(0x01, count)  # sets count
            dev.UpdateWireIns()

            dev.UpdateWireOuts()
            data = dev.GetWireOutValue(0x20)
            if data != old_data:
                # print(bitfield(data), len(bitfield(data)), time.time() - t_0)
                t_0 = time.time()
                # steps.append(bitfield(data))
                reading = bitfield(data)
                received_data = reading
                full_data = received_data
                action(full_data)
                step_i += 1
            old_data = data
            if step_i >= 16:
                count -= 0x00200000
                step_i = 0
                print("-------------------------------------------------------------")
                print("count: ", hex(count))
    except Exception as Error_case:
        pass
        print(Error_case)
        dev.Close()
        exit()
        # counter += 1
    except KeyboardInterrupt:
        dev.Close()
        ser.write('kbalance'.encode())
        print("keyboard interrupt".capitalize())
        exit()

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
                turn = 20
            if mapped_id in [12, 13, 14, 15]:
                turn = 45
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


if __name__ == "__main__":
    # To give ardiuno some time to open port. It's required
    try:
        # for x in range(8):
        #     servo_status.append(0)
        read_from_port(dev=dev)
    except Exception as Error_case:
        print(Error_case, "FPGA: ",error)
        ser.write('kbalance'.encode())
        ser.close()
        exit()
    except KeyboardInterrupt:
        ser.write('kbalance'.encode())
        ser.close()
        print("keyboard interrupt".capitalize())
        exit()
