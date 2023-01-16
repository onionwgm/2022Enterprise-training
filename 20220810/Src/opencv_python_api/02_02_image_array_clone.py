import numpy as np
# 0. 导入 OpenCV 模块
import cv2 as cv

# 指定待装载的图片路径
image_path = "images/jf1.jpg"

# 从指定的文件中，读入图片的图像的所有数据，并返回图片对象，为：numpy.ndarray 类型的数组
image1 = cv.imread(image_path)

# 创建名为"tutorials"的窗口以显示图片
window_name = "tutorials"
# 装载并显示图片1
cv.namedWindow(window_name)

# 浅层复制，修改image2中的数据，相当于也修改了image1中的数据
# 共享|指向同一段内存空间，相当于硬链接
image2 = image1
# 此时二者维度、元素值完全相同，且共享同一块图像元素内存
print("image1的维度：", image1.shape, "；image2的维度：", image2.shape)
# 将image2中所有数据全设为0
image2[:, :, :] = 0
# image1此时也全都变为黑色(颜色值为0)，即：显示全黑的图像对象
cv.imshow(window_name, image1)
cv.waitKey()

# 重新装载image1以进行后续操作
image1 = cv.imread(image_path)

# 深层复制，修改image3中的数据，对image1没有影响
# 相当于在新开辟的内存空间中，放置 image1 图像的所有的数据
image3 = np.copy(image1)
# 此时二者维度、元素值完全相同，但分别具有各自的图像元素内存
print("image1的维度：", image1.shape, "；image3的维度：", image3.shape)
# 将image3中所有数据全设为0
image3[:, :, :] = 0
# image1完全不受影响
cv.imshow(window_name, image1)

# 等待用户按任意键后继续运行后续代码
cv.waitKey()

# 销毁所有窗口
cv.destroyAllWindows()