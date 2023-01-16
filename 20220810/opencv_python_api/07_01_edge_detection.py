# 导入 OpenCV 模块
import cv2 as cv

# 设置命名窗口的名称
WINDOW_TITLE = "Edge Detection"
# 指定要加载的图片
IMAGE_PATH = "images/JF1.jpg"

# 直接以灰度图的形式读取原图片
gray_image = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

# 使用SOBEL一阶导数方法
sobel_grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3)       # 对x方向求一阶导
sobel_grad_y = cv.Sobel(gray_image, cv.CV_16S, 0, 1, ksize=3)       # 对y方向求一阶导
cv8u_grad_x = cv.convertScaleAbs(sobel_grad_x)            # 将计算结果转换成CV_8U像素类型
cv8u_grad_y = cv.convertScaleAbs(sobel_grad_y)
sobel_result = cv.addWeighted(cv8u_grad_x, 0.5, cv8u_grad_y, 0.5, 0)       # 合并x和y方向的结果合成最终输出

# 使用Canny边缘检测
blur_image = cv.blur(gray_image, (3, 3))        # 先作blur平滑处理
low_threshold = 48                              # 设定阈值
ratio = 3
kernel_size = 3
canny_result = cv.Canny(blur_image, low_threshold, low_threshold * ratio, kernel_size)

# 逐一显示灰度图片、两种边缘检测的图片
cv.imshow(WINDOW_TITLE, gray_image)
cv.waitKey()
cv.imshow(WINDOW_TITLE, sobel_result)
cv.waitKey()
cv.imshow(WINDOW_TITLE, canny_result)
cv.waitKey()
cv.destroyAllWindows()