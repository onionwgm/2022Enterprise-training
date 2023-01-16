# 导入 OpenCV 模块
import cv2 as cv

# 指定2张图像，并加载
image1_path = "images/jf1.jpg"
image2_path = "images/jf2.jpg"
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)

# 指定images中要复制的 ROI （感兴趣区域）
roi = image1[100:, 100:, :]     # 图片左上角(100,100)到图片右下角，颜色保持不变
# roi数组中的第一个维度代表高度(数组行数-图片像素的行数)
roi_height = roi.shape[0]
# roi数组中的第二个维度代表宽度(数组列数-图片像素的列数)
roi_width = roi.shape[1]
# roi数组中的第三个维度代表每个像素的颜色通道数(数组元素含有的描述颜色信息的个数)
roi_pixel = roi.shape[2]    # 本例中为用到此参数，因为此处 ROI 时保持了颜色不变

# 将roi数据复制到image2的目标区域中
image2[0:roi_height, 0:roi_width, :] = roi

# 显示对原始图片赋值了 ROI 后的图像对象（本质：roi覆盖了原始图像）
cv.imshow("image2 with roi copied", image2)
cv.waitKey()
cv.destroyAllWindows()
