# 导入 OpenCV 模块
import cv2 as cv

# 指明命名窗口名称
WINDOW_TITLE = "Morphology"
# 指定要加载的图片
IMAGE_PATH = "images/morphology.jpg"
# IMAGE_PATH = "images/contours.jpg"

# 读入原图片对象
source_image = cv.imread(IMAGE_PATH)
# 转换为灰度图片
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
# 使用 THRESH_OTSU 模式进行二值化
ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

# 执行闭操作
kernel = cv.getStructuringElement(cv.MORPH_RECT, (32, 5))        #尝试调整Size中的核大小，可以得到不同的连接结果
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 3))
# 闭操作之后，车牌字符区域连接成一片，一般有利于使用等值线轮廓识别该区域矩形
morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

cv.imshow(WINDOW_TITLE, source_image)
cv.waitKey()
cv.imshow(WINDOW_TITLE, morphology_image)
cv.waitKey()
cv.destroyAllWindows()