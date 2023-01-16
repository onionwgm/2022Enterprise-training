# 导入 OpenCV 模块
import cv2 as cv

# 设置命名窗口的名称
WINDOW_TITLE = "Image Binarize"
# 指定要加载的图片
IMAGE_PATH = "images/chess.jpg"

# 直接以灰度图的形式读取原图片
gray_image = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

# 指定阈值的二值化处理
threshold_value = 127.0
max_value = 255

# THRESH_BINARY 模式
ret, binary_image1 = cv.threshold(gray_image, threshold_value, max_value, cv.THRESH_BINARY)
# THRESH_BINARY_INV 模式
ret, binary_image2 = cv.threshold(gray_image, threshold_value, max_value, cv.THRESH_BINARY_INV)

# 采用最大类间方差/大津法（OTSU）自动优化阈值。
ret, binary_image3 = cv.threshold(gray_image, 0, max_value, cv.THRESH_OTSU)

# 逐一显示三种二值化后的灰度图片对象
cv.imshow(WINDOW_TITLE, binary_image1)
cv.waitKey()
cv.imshow(WINDOW_TITLE, binary_image2)
cv.waitKey()
cv.imshow(WINDOW_TITLE, binary_image3)
cv.waitKey()

cv.destroyAllWindows()
