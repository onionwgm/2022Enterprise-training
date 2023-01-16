# 导入 OpenCV、numpy、matplotlib 模块
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 设置命名窗口的名称
WINDOW_TITLE = "Histgram"
# 指定要加载的图片
IMAGE_PATH = "images/jf1.jpg"

# 读入原图片对象
source_image = cv.imread(IMAGE_PATH)
# 转换为灰度图片
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

# 计算灰度图的直方数据
channel_index = 0           # 0 代表三个颜色通道中的第一个，因为是灰度图片，就只有这一个通道
hist_size = 256
range = [0, 256]
# 计算灰度图-gray_image中，每个像素的第0个颜色通道的取值在0~255的每个值的存在数量
hist = cv.calcHist([gray_image], [channel_index], None, [hist_size], range)
# 逐一输出 0~255 的每个数值在灰度图gray_image中每个像素第0个颜色通道的取值的个数
for i in np.arange(hist_size):
    print("灰度值【%d】的像素个数：%d" % (i, hist[i]))

# 手动绘制直方图
PLOT_WIDTH = 512        # 指定绘图坐标的X轴和Y轴长度
PLOT_HEIGHT = 400
# 归一化，hist中每个元素值将介于0~PLOT_HEIGHT之间。实际上求出了每个元素所占的百分比
normalized_hist = hist / np.max(hist)
hist_image = np.zeros((PLOT_HEIGHT, PLOT_WIDTH, 3))
for i in np.arange(hist_size):
    w = 2
    h = PLOT_HEIGHT * (1 - normalized_hist[i])
    x = i * w
    y = PLOT_HEIGHT
    cv.rectangle(hist_image, (int(x), int(y)), (int(x) + int(w), int(h)), (0, 0, 255))
    # 参数必须是整型，使用强制类型转换强调|转换了
    cv.rectangle(hist_image, (int(x), int(y)), (int(x) + int(w), int(h)), (0, 0, 255))

# 在命名窗口中显示直方图
cv.imshow(WINDOW_TITLE, hist_image)
cv.waitKey()
cv.destroyAllWindows()

# 在matplot中直接生成直方图
plt.hist(gray_image.ravel(), 256, range)
plt.show()


