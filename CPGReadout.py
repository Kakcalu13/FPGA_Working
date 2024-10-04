import time
import ok

def bitfield(n, length = 16):
    bf = [int(digit) for digit in bin(n)[2:]]
    for i in range(length - len(bf)):
        bf = [0] + bf
    return bf

dev = ok.okCFrontPanel()
dev.OpenBySerial("")
error = dev.ConfigureFPGA(r"C:\Users\ereij\OneDrive\Documents\PITT\FPGA\qDVS\CPG_OK.bit")
print(error)

# set timing
count = 0x01000000
clk_f = 200E6
tc = int(count)/clk_f
print(int(count), tc)

# Reset chip
dev.SetWireInValue(0x01, count)    # sets count
dev.SetWireInValue(0x00, 0b0010)        # pull both resets and spike in down
dev.UpdateWireIns()

time.sleep(tc)
dev.SetWireInValue(0x00, 0b1000)        # pull clk_rst_n high

time.sleep(tc*10)
dev.SetWireInValue(0x00, 0b1101)        # pull rst_n high and start high
dev.UpdateWireIns()

time.sleep(tc*1.6)
dev.SetWireInValue(0x00, 0b1100)        # pull start low
dev.UpdateWireIns()


old_data = 0
t_0 = time.time()
steps = []
while True:
    data = dev.UpdateWireOuts()
    data = dev.GetWireOutValue(0x20)
    if data != old_data:
        print(bitfield(data), len(bitfield(data)), time.time()-t_0)
        t_0 = time.time()
        steps.append(bitfield(data))
    old_data = data