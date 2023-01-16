import cv2
import numpy as np
filename = 'data/in.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 输入图像必须是float32，最后一个参数在0.04 到0.06 之间
dst = cv2.cornerHarris(gray, 2, 3, 0.06)
# 返回角点图片
dst = cv2.dilate(dst, None)
# 将角点标注为红色
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.06)
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#Python: cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)
# 返回值由角点坐标组成的一个数组（而非图像）
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
res = np.hstack((centroids, corners))
#np.int0 可以用来省略小数点后面的数字（四舍五入）。
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]
cv2.imshow('img', img)
cv2.waitKey()
