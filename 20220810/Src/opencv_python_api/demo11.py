import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("data/nba.jpg", 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 构建振幅图
# 先取绝对值，表示取模。取对数，将数据范围变小
magnitude_spectrum = 20*np.log(np.abs(fshift))
print(magnitude_spectrum)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
