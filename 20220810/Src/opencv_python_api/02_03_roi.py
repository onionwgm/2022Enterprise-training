# 导入 OpenCV 模块
import cv2 as cv

# 指定待装载的图片路径
image_path = "images/jf1.jpg"

# 从指定的文件中，读入图片的图像的所有数据，并返回图片对象，为：numpy.ndarray 类型的数组
image = cv.imread(image_path)

# 创建名为"tutorials"的窗口以显示图片
window_name = "tutorials"
# 装载并显示图片1
cv.namedWindow(window_name)

# 获取图像中的一个矩形子区域，实际上做了一个浅层复制
'''
    请注意，多维数组各个维度顺序为高x宽x通道(HxWxC)
    通过下标索引取数组中的元素时：
        第1个下标代表图片的高度方向，同时也代表数组的行标号
        第2个下标代表图片的宽度方向，同时也代表数组的列标号
        第3个下标代表图片的颜色通道
'''
roi = image[:400, 0:200, :]     # 获取原始图像对象像素左上角(0.0)开始 400x200 大小的图像，颜色通道保持不变
# 显示 roi 图像对象
cv.imshow(window_name, roi)
cv.waitKey()

# 如果将roi中的颜色值都修改为0，则image图片也会被修改
roi[:, :, :] = 0  # 每个（颜色）值均为 0
# 显示含有被修改后的roi的原始图像对象
# 因为 roi 是浅拷贝，所以是对原始图像做了修改
cv.imshow(window_name, image)

cv.waitKey()
cv.destroyAllWindows()

