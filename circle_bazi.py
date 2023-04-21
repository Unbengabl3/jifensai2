# https://blog.csdn.net/libaineu2004/article/details/122807673
import cv2
import serial
import numpy as np
from struct import pack

# se = serial.Serial("COM3", 115200, timeout=1)

# def send_data(SDI_Point0, SDI_Point1):
#     # 设置校验位并发送包头'AC'
#     sumA = 0
#     sumB = 0
#     data = bytearray([0x41, 0x43])
#     se.write(data)
#
#     # 发送消息类型和消息长度
#     data = bytearray([0x02, 8])  # 0x02为消息类型，此处值为2。若为1就是0x01。4是消息长度，只发1个数据就是4,2个就是8
#     for b in data:
#         sumB = sumB + b
#         sumA = sumA + sumB
#     se.write(data)
#
#     # 发送数据
#     ###################################################
#     float_bytes = pack('f', float(SDI_Point0))
#     for b in float_bytes:
#         sumB = sumB + b
#         sumA = sumA + sumB
#     se.write(float_bytes)
#
#     var_bytes = pack('f', float(SDI_Point1))  # -0.16 * SDI_Point1
#     for b in var_bytes:
#         sumB = sumB + b
#         sumA = sumA + sumB
#     se.write(var_bytes)
#     ###################################################
#
#     # 发送校验位
#     while sumA > 255:
#         sumA = sumA - 255
#     while sumB > 255:
#         sumB = sumB - 255
#     data = bytearray([sumA, sumB])
#     se.write(data)

def findcircle2(img3):
    h, w = img3.shape[:2]
    # print(h, w)

    # 进行高斯模糊
    img_blur = cv2.GaussianBlur(img3, (5, 5), 0)
    # cv2.imshow('blur', img_blur)

    # 利用Sobel算子计算x和y方向上的梯度
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的幅值和方向
    mag, ang = cv2.cartToPolar(sobelx, sobely)

    # 设定阈值，提取出强梯度
    threshold = 50
    mag_thresh = np.zeros_like(mag)
    mag_thresh[mag > threshold] = mag[mag > threshold]

    # 对提取出的梯度进行二值化处理
    _, binary = cv2.threshold(mag_thresh, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary)
    binary = cv2.convertScaleAbs(binary)

    img_gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gyay', img_gray)

    # 设定圆形检测的参数
    min_radius = 20
    max_radius = 200
    dp = 1
    param1 = 120
    param2 = 120
    # 此参数用于识别靶子

    # 进行圆形检测
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist=20, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    # 绘制检测到的圆形
    if circles is not None:
        circles = sorted(np.round(circles[0, :]).astype("int"), key=lambda circle: circle[1])

        # 去除同心圆
        i = 0
        while i < len(circles) - 1:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[i + 1]
            if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2 * r1:
                circles.pop(i + 1)
            else:
                i += 1

        # 绘制检测到的圆形
        for (x, y, r) in circles:
            cv2.circle(img3, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img3, (x, y), 2, (0, 0, 255), 3)

            # 输出圆形信息
            for i, (x, y, r) in enumerate(circles):
                # print("Circle {}: x = {}, y = {}, r = {}".format(i + 1, x, y, r))
                souce = [0, 0]
                souce[0] = (x - w / 2 + 40) * 200 / w
                souce[1] = (y - h / 2 + 40) * 200 / h
                print(souce[0], souce[1])
                # send_data(souce[0], souce[1])

    else:
        # send_data(0, 0)
        print(0, 0)

    # 显示结果
    cv2.imshow('result', img3)


# 读取图像
cap2 = cv2.VideoCapture(0)
while(True):
    ret2, frame2 = cap2.read()
    findcircle2(frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()