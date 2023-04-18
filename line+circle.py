import cv2
import numpy as np

def print_circle(i, h, w, x, y, r):
    # print("Circle {}: x = {}, y = {}, r = {}".format(i + 1, x, y, r), end='\t')
    if (w / 2 - 10 < x < w / 2 + 10 and h / 2 - 10 < y < h / 2 + 10):
        print(0, end='')
    else:
        if (x < w / 2):
            print(1, end='')
        else:
            print(2, end='')
        if (y < h / 2):
            print(1, end=' ')
        else:
            print(2, end=' ')


def circle(img2):
    h, w = img2.shape[:2]
    # print(h, w)

    # 进行中值滤波
    dst_img = cv2.medianBlur(img2, 7)

    img_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

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
            cv2.circle(img2, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img2, (x, y), 2, (0, 0, 255), 3)

        # 输出圆形信息
        for i, (x, y, r) in enumerate(circles):
            if(i == 0):
                # print("c1")
                print_circle(i, h, w, x, y, r)
            if (i == 1):
                # print("c2")
                print_circle(i, h, w, x, y, r)
                print()
            if (i >= 2):
                continue


def line(img):# img 必须为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(a)
    img = frame  # 切片后图像  这是为了能够更好的检测，选取图像中的某一位置的图像经行检测
    # 直方滤波
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    img = cv2.merge((bH, gH, rH))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度图

    img = cv2.GaussianBlur(img, (5,5),1)

    img = cv2.blur(img, (3,3))# 5,5

    img = cv2.medianBlur(img, 5)
    # 双边滤波

    edges = cv2.Canny(img, 10, 200, apertureSize=3)  # 边缘检测 #10
    cv2.imshow('Canny',edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20) # 直线检测
    # print(lines,'\n')
    if lines is None:
        return ((0,1),(1,0))
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        # print(line,rho,theta,np.pi / 4.,3. * np.pi / 4.0,'----------------------------------')
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # 该直线与最后一行的焦点
            pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
        else:  # 水平直线
            pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # 该直线与最后一列的交点
            pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
        return (pt1 ,pt2)
    pass

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    circle(frame)

    pt1, pt2 = line(frame)
    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
    # print((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

