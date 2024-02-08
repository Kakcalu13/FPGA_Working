import cv2
import numpy as np
import math
from PIL import Image
outfile = open("read_out.txt", "r")
data = outfile.readlines()
outfile.close()
size = 0
for line in data:
    #print(line)
    size = size+1
j = 0



def binaryToDecimal(binary):
    decimal, i = 0, 0
    while (binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal
    #print(decimal)
frame = 127*np.ones((1024, 1024), dtype=np.uint8)

for line in data:
    addresses = line.split()
    res = []
    for i in addresses:
        if i.isnumeric():
            res.append(int(i))
    print(res)
    row_tile_address = binaryToDecimal(int(str(res[0])[-6:]))
    print(row_tile_address)
    try:
        col_tile_address = binaryToDecimal(int(str(res[1])[-6:]))
    except:
        print("file over")
    row_address = binaryToDecimal(math.floor(res[0]/1000000))
    try:
        col_address = binaryToDecimal(math.floor(res[1] / 1000000))
    except:
        print("file over")
    #row_address = binaryToDecimal(int(str(res[0])[6:]))
    #col_tile_address = 1 * int(str(res[1])[-1:]) + 2 * int(str(res[1])[-2:-1])

    #row_address = 1 * int(str(res[0])[-2:-1])+ 2 * int(str(res[0])[-4:-3]) + 4 * int(str(res[0])[-5:-4]) + 8 * int(str(res[0])[-6:-5]) + 16 * int(str(res[0])[-7:-6])
    #col_address = 1 * int(str(res[1])[-3:-2]) + 2 * int(str(res[1])[-4:-3]) + 4 * int(str(res[1])[-5:-4]) + 8 * int(str(res[1])[-6:-5]) + 16 * int(str(res[1])[-7:-6])


    frame[row_tile_address*16 + row_address][col_tile_address*16 + col_address] = 255
    j=j+1
    #print(x)

    #print(addresses)

print(frame.shape)
print(frame)
Image.fromarray(frame.astype('uint8')).save('16x16.jpeg')
img = Image.fromarray(frame)
img.show()
#img.save("64x64", format='jpg')