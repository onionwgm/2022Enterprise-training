# 导入 OpenCV 模块
import cv2 as cv

# 设置命名窗口的名称
WINDOW_TITLE = "Contours"
# 指定要加载的图片
IMAGE_PATH = "images/contours.jpg"

# 读入原图片对象
source_image = cv.imread(IMAGE_PATH)
# 1. 转换为灰度图片
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
# 2. 使用 THRESH_OTSU 模式进行二值化
ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

# 3. 计算等值线
# 计算二值图中的等值线，将结果存放在一个集合中。
# 集合中的每个元素都记录了该条等值线上主要像素点的信息
contours, _ = cv.findContours(binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# 4. 绘制等值线图，查看该等值线所包围的区域
cv.drawContours(source_image, contours, -1, (0, 0, 255))        # 以红色绘制等值线

cv.imshow(WINDOW_TITLE, source_image)
cv.waitKey()
cv.destroyAllWindows()
