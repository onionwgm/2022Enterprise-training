# 0. 导入 OpenCV 模块
import cv2 as cv

# 指定待装载的图片路径
image_path = "images/jf2.jpg"

# 从指定的文件中，读入图片的图像的所有数据，并返回图片对象，为：numpy.ndarray 类型的数组
image_origin = cv.imread(image_path)

# 调整图像大小为480x640，请注意，新尺寸是按照宽x高的顺序来表示
# 返回缩放后的图片对象
image_resized = cv.resize(image_origin, (480, 640))

# 在指定名称的命名窗口中显示原始图片对象和缩放后的图片对象
cv.imshow("tutorials_origin", image_origin)
cv.imshow("tutorials_resized", image_resized)

# 等待用户按任意键后继续运行后续代码
cv.waitKey()

# 销毁所有的窗口
cv.destroyAllWindows()

