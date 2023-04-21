import serial
from struct import pack

se = serial.Serial("COM3", 115200, timeout=1)

def send_data(SDI_Point0, SDI_Point1):
    # 设置校验位并发送包头'AC'
    sumA = 0
    sumB = 0
    data = bytearray([0x41, 0x43])
    se.write(data)

    # 发送消息类型和消息长度
    data = bytearray([0x02, 8])  # 0x02为消息类型，此处值为2。若为1就是0x01。4是消息长度，只发1个数据就是4,2个就是8
    for b in data:
        sumB = sumB + b
        sumA = sumA + sumB
    se.write(data)

    # 发送数据
    ###################################################
    float_bytes = pack('f', float(SDI_Point0))
    for b in float_bytes:
        sumB = sumB + b
        sumA = sumA + sumB
    se.write(float_bytes)

    var_bytes = pack('f', float(SDI_Point1))  # -0.16 * SDI_Point1
    for b in var_bytes:
        sumB = sumB + b
        sumA = sumA + sumB
    se.write(var_bytes)
    ###################################################

    # 发送校验位
    while sumA > 255:
        sumA = sumA - 255
    while sumB > 255:
        sumB = sumB - 255
    data = bytearray([sumA, sumB])
    se.write(data)


while (True):
    a = -5.90
    b = -5.45
    send_data(a, b)
