# https://blog.csdn.net/libaineu2004/article/details/122807673
import cv2
import numpy as np

circles_list = []
counter = 1
# 读取图像
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    # print(h, w)
    # 进行中值滤波
    dst_img = cv2.medianBlur(frame, 7)
    # cv2.imshow('dst', dst_img)

    img_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # cv2.imshow('blur', img_blur)

    # 设定圆形检测的参数
    min_radius = 20
    max_radius = 200
    dp = 1.5
    param1 = 300
    param2 = 0.93

    # 进行圆形检测
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT_ALT, dp, minDist=20, param1=param1, param2=param2,
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
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            # 输出圆形信息
            for i, (x, y, r) in enumerate(circles):
                print("Circle {}: x = {}, y = {}, r = {}".format(i + 1, x, y, r))
                if(w / 2 - 10 < x < w / 2 + 10 and h / 2 - 10 < y < h / 2 + 10):
                    print(0)
                else:
                    if(x < w / 2): print(1)
                    else: print(2)
                    if (y < h / 2): print(1)
                    else: print(2)

    # 显示结果
    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()