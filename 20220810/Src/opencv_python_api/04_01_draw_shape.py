# 导入 OpenCV 和 numpy 模块
import cv2 as cv
import numpy as np

# 设置命名窗口的名称
WINDOW_TITLE = "Draw Shape"

# 创建480行x640x列x3通道的数组 - numpy 数组
# 实际上表示 高度480，宽度640，像素颜色值初始化为全0(黑色-三个RGB通道均为0)
image = np.zeros((480, 640, 3))

# 在命名窗口中显示上述 numpy 数组表示的图像
cv.imshow(WINDOW_TITLE, image)
cv.waitKey()

# 绘制矩形，指定画布图片，左上、右下坐标，颜色（BGR,此为红色），线宽
cv.rectangle(image, (20, 20), (120, 220), (0, 0, 255), 3)       # 绘制红色矩形， 线宽度为3
# cv.rectangle(image, (20, 20), (120, 220), (0, 0, 255), -1)       # 绘制红色矩形， 线宽度为3
# 绘制圆形，指定画布图片，圆心坐标、半径，颜色（BGR,此为绿色），线宽
cv.circle(image, (320, 240), 100, (0, 255, 0), 2)               # 绘制绿色圆，半径100，线宽度为2
# 以上线宽粗细设为-1，将填充对象
# 绘制直线，指定画布图片，起点、终点坐标，颜色（BGR,此为蓝色），线宽
cv.line(image, (70, 120), (320, 240), (255, 0, 0), 1)           # 绘制蓝色直线，线宽度为1
# 绘制文字，指定画布图片，文字左上坐标，字体，线宽，颜色（白色）
cv.putText(image, "Hello World", (320, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))     # 输出白色文本

# 绘制多边形
# 设置多边形顶点的坐标数组为pts（点的简称）
# 使用cv2.polylines来画线，绘制多边形。
# 参数如下：绘制的对象，坐标（逐一连接终止的和起始点），颜色和粗细。
# 5x2 矩阵
pts = np.array([[10, 5], [200, 300], [200, 400], [100, 400], [50, 10]], np.int32)
# reshape 为 (?) x 1 x 2 矩阵 =>> 变为 5x1x2 的矩阵，为 5 个坐标
pts = pts.reshape((-1,1,2))
# 绘制黄色五边形
cv.polylines(image, [pts], True, (0, 255, 255), 5)

# 此时的 image 为含有绘制好的形状的图像
cv.imshow(WINDOW_TITLE, image)
cv.waitKey()
cv.destroyAllWindows()

