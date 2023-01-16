import cv2
import numpy as np
filename = 'data/in.jpg'
img = cv2.imread(filename)
cv2.imshow('input', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 输入图像必须是float32，最后一个参数在0.04 到0.06 之间
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# 返回角点检测结果
dst = cv2.dilate(dst, None)
# 设置标注角点颜色为红色
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('output', img)
cv2.waitKey()
