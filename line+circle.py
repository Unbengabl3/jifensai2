import cv2
import numpy as np

#输出内容先x再y，左右方向x，上下方向y，靠左上为1+++++++++++++++++++++++++++++++++++++++
def print_circle(i, h, w, x, y, r):
    # print("Circle {}: x = {}, y = {}, r = {}".format(i + 1, x, y, r), end='\t')
    # if (w / 2 - 50 < x < w / 2 + 50 and h / 2 - 50 < y < h / 2 + 50):
    #     print(00, end='')
    # else:
    #     if (x < w / 2):
    #         print(1, end='')
    #     else:
    #         print(2, end='')
    #     if (y < h / 2):
    #         print(1, end=' ')
    #     else:
    #         print(2, end=' ')
    souce = [0, 0]
    souce[0] = (x - w / 2) * 100 / w
    souce[1] = (y - h / 2) * 100 / h

    print(souce[0], souce[1])


def is_green_color(img, x1, y1, x2, y2):
    # 裁剪图像，提取给定区域
    region = img[y1:y2, x1:x2]
    if np.all(region == 255):
        # Handle the case where region is all white
        return False
    else:
        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 定义绿色的HSV范围
        lower_green = (36, 25, 25)
        upper_green = (70, 255, 255)

        # 利用inRange函数得到绿色像素的掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 计算绿色像素所占比例
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = cv2.countNonZero(mask)
        green_ratio = green_pixels / total_pixels

        # 判断绿色像素所占比例是否大于0.3
        if green_ratio > 0.3:
            return True
        else:
            return False


def circle(img2):
    h, w = img2.shape[:2]
    # print(h, w)

    # 进行中值滤波
    # dst_img = cv2.medianBlur(img2, 7)
    # cv2.imshow("dst", dst_img)

    #img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊
    img_blur = cv2.GaussianBlur(img2, (5, 5), 0)
    cv2.imshow("blur", img_blur)

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
    cv2.imshow('binary', binary)
    binary = cv2.convertScaleAbs(binary)
    img_gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gyay', img_gray)

    # 设定圆形检测的参数
    min_radius = 40
    max_radius = 400
    dp = 1
    param1 = 100
    param2 = 70

    # 进行圆形检测
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist=20, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    # min_radius = 40
    # max_radius = 400
    # dp = 1.5
    # param1 = 300
    # param2 = 0.95
    #
    # # 进行圆形检测
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT_ALT, dp, minDist=20, param1=param1, param2=param2,
    #                            minRadius=min_radius, maxRadius=max_radius)

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

        for i, (x, y, r) in enumerate(circles):
            if (i == 0):
                print_circle(i, h, w, x, y, r)
                print()
            if ( i >= 1):
                continue

        # 输出圆形信息
        # circle_green = [5,5,5]
        # circle_white = [5,5,5]
        # for i, (x, y, r) in enumerate(circles):
        #     if (len(circles) == 1):
        #         print_circle(i, h, w, x, y, r)
        #         print_circle(i, h, w, x, y, r)
        #         print()
        #
        #     else:
        #         if (i == 0):
        #             # print("c1")
        #             if (is_green_color(img2, round(x - r / 4), round(y - r / 4), round(x + r / 4),
        #                                round(y + r / 4)) == True):
        #                 #print("绿", end='')
        #                 circle_green[0] = x
        #                 circle_green[1] = y
        #                 circle_green[2] = r
        #             else:
        #                 #print("白", end='')
        #                 circle_white[0] = x
        #                 circle_white[1] = y
        #                 circle_white[2] = r
        #             # print_circle(i, h, w, x, y, r)
        #         if (i == 1):
        #             # print("c2")
        #             if (is_green_color(img2, round(x - r / 4), round(y - r / 4), round(x + r / 4),
        #                                round(y + r / 4)) == True):
        #                 #print("绿", end='')
        #                 circle_green[0] = x
        #                 circle_green[1] = y
        #                 circle_green[2] = r
        #             else:
        #                 #print("白", end='')
        #                 circle_white[0] = x
        #                 circle_white[1] = y
        #                 circle_white[2] = r
        #             # print_circle(i, h, w, x, y, r)
        #             # print()
        #         if (i >= 2):
        #             continue
        #         if(len(circle_green) == 3 and len(circle_white) == 3 ):
        #             #print("绿", end='')
        #             print_circle(i, h, w, circle_green[0], circle_green[1], circle_green[2])
        #             #print("白", end='')
        #             print_circle(i, h, w, circle_white[0], circle_white[1], circle_white[2])
        #             print()

def line(img):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(a)
    # img = frame  # 切片后图像  这是为了能够更好的检测，选取图像中的某一位置的图像经行检测
    # 直方滤波
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    img = cv2.merge((bH, gH, rH))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

    img = cv2.GaussianBlur(img, (5, 5), 1)

    img = cv2.blur(img, (3, 3))  # 5,5

    img = cv2.medianBlur(img, 5)
    # 双边滤波

    edges = cv2.Canny(img, 10, 200, apertureSize=3)  # 边缘检测 #10
    cv2.imshow('Canny', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)  # 直线检测
    # print(lines,'\n')
    if lines is None:
        return ((0, 1), (1, 0))
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
        return (pt1, pt2)
    pass



cap = cv2.VideoCapture(1)
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
