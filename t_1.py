import cv2
import numpy as np
kernel = np.ones((4, 4), np.uint8)
cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
    # 读取图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 创建二值掩模，将白色部分设置为0
    mask_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # 将红色掩模与白色掩模相乘，去除白色部分
    mask_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(mask_red, lower_red, upper_red)
    mask_red2 = cv2.inRange(mask_red, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    # cv2.imshow("mask", mask_red)
    mask_red = cv2.bitwise_and(mask_red, mask_white)

    mask_red = cv2.dilate(mask_red, kernel, iterations=3)

    cv2.imshow("dilate", mask_red)

    # 应用掩模，分离红色通道
    # red_channel = cv2.bitwise_and(frame, frame, mask=mask_red)
    # cv2.imshow("red", red_channel)

    # gray = cv2.cvtColor(red_channel, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # grad = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)

    # 应用自适应阈值处理
    # th = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -100)
    ret, thresh2 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=4)
    # 显示结果
    cv2.imshow('image', thresh2)
    cv2.imshow('pic', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

