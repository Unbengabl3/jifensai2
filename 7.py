import numpy as np
import cv2

cap = cv2.VideoCapture(0)
setItem = set()
if cap.isOpened() is True:  # 检查摄像头是否正常启动
    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰色通道
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 定义结构元素
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # 形态学开运算
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # 形态学闭运算
        edges = cv2.Canny(opening, 75, 145)  # 边缘识别

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print("Number of contours found:", len(contours))

        # 遍历所有轮廓，找到圆度最高的两个轮廓
        h, w = gray.shape
        max_contours = []
        roundness_threshold = 0.75
        aspect_ratio_min = 0.5
        aspect_ratio_max = 2
        for contour in contours:
            # 计算轮廓的最小包围圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # 如果最小包围圆的半径小于指定半径，则跳过该轮廓
            if radius < 20 or radius > 400:
                continue
            # 对轮廓进行逼近
            approx = cv2.approxPolyDP(contour, epsilon=0.005 * cv2.arcLength(contour, True), closed=True)
            # 计算逼近后多边形的边数和周长
            sides = len(approx)
            perimeter = cv2.arcLength(approx, True)
            # 计算多边形的圆度
            if perimeter != 0:
                roundness = 4 * np.pi * cv2.contourArea(contour) / perimeter ** 2
            else:
                roundness = 0
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            # print("Contour roundness:", roundness)

            # 判断圆度是否大于阈值
            if roundness >= roundness_threshold:
                # 判断宽高比是否在合理的范围内
                if aspect_ratio >= aspect_ratio_min and aspect_ratio <= aspect_ratio_max:
                    # 将当前轮廓添加到最圆的轮廓列表中
                    if len(max_contours) < 2:
                        max_contours.append((contour, approx, roundness))
                    else:
                        if roundness > max_contours[0][2]:
                            max_contours[1] = max_contours[0]
                            max_contours[0] = (contour, approx, roundness)
                        elif roundness > max_contours[1][2]:
                            max_contours[1] = (contour, approx, roundness)

        # 输出圆的数量和圆度值
        # print("Number of circles found:", len(max_contours))
        # for i, (contour, approx, roundness) in enumerate(max_contours):
        #     print("Circle", i + 1, "roundness:", roundness)
        target_circle = []
        # 绘制最圆的两个轮廓并显示图像
        for i, (contour, approx, roundness) in enumerate(max_contours):
            color = (0, 0, 255) if i == 0 else (0, 255, 0)
            cv2.drawContours(frame, [approx], -1, color, 2)
            # 计算最小包围圆
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            # 输出圆心坐标
            # target_circle[i][0] = center_x
            # target_circle[i][1] = center_y
            print(f"圆{i + 1}的圆心坐标为: ({center_x}, {center_y})")

        # if len(target_circle) >= 2:
        #     if target_circle[0][1] > target_circle[1][1]:
        #         print(1, target_circle[0][0], target_circle[0][1], 2, target_circle[1][0], target_circle[1][1])
        #     else:
        #         print(1, target_circle[1], 2, target_circle[0])

        cv2.imshow('edges', edges)
        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print('cap is not opened!')