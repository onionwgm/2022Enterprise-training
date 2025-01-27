import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("data/nba.jpg", 0)
f = np.fft.fft2(img)    #得到结果为复数矩阵
fshift = np.fft.fftshift(f) #直接取中心
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0    #蒙板大小60×60
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)       #使用FFT逆变换，此时结果仍然是复数
img_back = np.abs(img_back)	    # 取绝对值
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()
