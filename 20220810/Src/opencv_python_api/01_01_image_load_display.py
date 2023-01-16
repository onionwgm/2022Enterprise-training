# 0. 导入 OpenCV 模块
import cv2 as cv

# 指定待装载的图片路径
image1_path = "images/jf1.jpg"
image2_path = "images/jf2.jpg"

# 1. 创建命名窗口，后续可以通过窗口名称来访问创建的命名窗口
# 创建名为"tutorials"的窗口以显示图片
window_name = "tutorials"
cv.namedWindow(window_name)

# 2. 从指定的文件中，读入图片的图像的所有数据，并返回图片对象，为：numpy.ndarray 类型的数组
# 装载并显示图片1
image1 = cv.imread(image1_path)
#print(type(image1))

# 3. 在指定名称的命名窗口中显示图片
cv.imshow(window_name, image1)

# 4. 等待用户按任意键后继续运行后续代码
cv.waitKey()

# 装载并显示图片2
image2 = cv.imread(image2_path)
cv.imshow(window_name, image2)

# 等待用户按任意键后继续运行后续代码
cv.waitKey()

# 5. 销毁所有窗口
cv.destroyAllWindows()
