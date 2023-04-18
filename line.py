import cv2
import numpy as np

def opencv_img(img):# img 必须为灰度图
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('ded',img2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度图
    cv2.imshow('huidutu',img)
    # 高斯滤波
    img = cv2.GaussianBlur(img, (5,5),1)
    # cv2.imshow('gaosi', img)
    # # 均值滤波
    img = cv2.blur(img, (3,3))# 5,5
    # cv2.imshow('junzhi', img)
    # 中值滤波
    img = cv2.medianBlur(img, 5)
    # 双边滤波
    """
    src：输入图像
    d：过滤时周围每个像素领域的直径
    sigmaColor：Sigma_color较大，则在邻域中的像素值相差较大的像素点也会用来平均。
    sigmaSpace：Sigma_space较大，则虽然离得较远，但是，只要值相近，就会互相影响。
    将sigma_space设置较大，sigma_color设置较小，可获得较好的效果（椒盐噪声）。
    """
    # img = cv2.bilateralFilter(img, 10, 75,75)#10,75,75

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
        return (pt1,pt2)
    pass


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(a)
    img = frame# 切片后图像  这是为了能够更好的检测，选取图像中的某一位置的图像经行检测
    #直方滤波
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    img = cv2.merge((bH, gH, rH))
    cv2.imshow('a',img)
    # Display the resulting frame
    pt1, pt2 = opencv_img(img)
    cv2.line(frame, pt1, pt2,(0,0,255) , 2)  # 绘制一条蓝线
    print((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()