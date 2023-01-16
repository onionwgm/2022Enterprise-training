# 导入 OpenCV 模块
import cv2 as cv

# 设置命名窗口的名称
WINDOW_TITLE = "Gray Image"
# 指定要加载的图片
IMAGE_PATH = "images/chess.jpg"

# 直接以灰度图的形式读取原图片
gray_image1 = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

# 读取原图片，然后转成灰度图
source_image = cv.imread(IMAGE_PATH)
gray_image2 = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

# 在命名窗口中显示两种灰度图片，二者等价。
cv.imshow(WINDOW_TITLE, gray_image1)
cv.waitKey()
cv.imshow(WINDOW_TITLE, gray_image2)
cv.waitKey()

cv.destroyAllWindows()
