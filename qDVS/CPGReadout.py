import ok

dev = ok.okCFrontPanel()
dev.OpenBySerial("")    # selects default value
error = dev.ConfigureFPGA(r"C:\Users\ereij\OneDrive\Documents\PITT\FPGA\qDVS\CPG_OK_ms.bit")    # change file address as needed
print(error)

# Reset chip
dev.SetWireInValue(0x00, 0b1100)
dev.UpdateWireIns()

# Set both rst_n low (active low reset)
dev.SetWireInValue(0x00, 0b0000)
dev.UpdateWireIns()

# Set clock_rst_n high
dev.SetWireInValue(0x00, 0b1000)
dev.UpdateWireIns()

# Set start signal high
dev.SetWireInValue(0x00, 0b1101)
dev.UpdateWireIns()

# polling FPGA WireOut values and prints if value changes
old_data = 0
while True:
    data = dev.UpdateWireOuts()
    data = dev.GetWireOutValue(0x20)
    if data != old_data:
        print(bin(data))
    old_data = data