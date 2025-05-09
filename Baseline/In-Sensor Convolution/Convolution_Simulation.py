import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取图像
img = cv2.imread('image.jpg')
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', grayImage)

#卷积核
dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)


cv2.imwrite('Simulation_Sharpened.jpg', Laplacian)