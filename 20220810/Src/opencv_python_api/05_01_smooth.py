# 导入 OpenCV 模块
import cv2 as cv

# 设置命名窗口的名称
WINDOW_TITLE = "Image Smooth"
# 指定要加载的图片
IMAGE_PATH = "images/chess.jpg"
# 设置核的大小
KERNEL_SIZE_X = 11
KERNEL_SIZE_Y = 11

# 加载图片对象
image_src = cv.imread(IMAGE_PATH)

# 均值平滑，求核（像素块）的平均值
image_blur = cv.blur(image_src, (KERNEL_SIZE_X, KERNEL_SIZE_Y))     # 均值滤波

# 中值平滑，图像平滑里中值滤波的效果最好
image_MB = cv.medianBlur(image_src, KERNEL_SIZE_X)      #可以更改核的大小

# 高斯平滑, 核尺寸必须为奇数
image_GB = cv.GaussianBlur(image_src, (KERNEL_SIZE_X, KERNEL_SIZE_Y), 0)    # 高斯滤波

# 双边滤波
image_BF = cv.bilateralFilter(image_src, 25, 100, 100)  # 双边滤波

# 逐次显示 原始图片、均匀平滑后的图片、高斯平滑后的图片
cv.imshow(WINDOW_TITLE, image_src)
cv.waitKey()
cv.imshow(WINDOW_TITLE, image_blur)
cv.waitKey()
cv.imshow(WINDOW_TITLE, image_MB)
cv.waitKey()
cv.imshow(WINDOW_TITLE, image_GB)
cv.waitKey()
cv.imshow(WINDOW_TITLE, image_BF)
cv.waitKey()
cv.destroyAllWindows()
