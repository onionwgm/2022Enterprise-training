# 导入 OpenCV 模块
import cv2 as cv

# 1. 获取摄像头设备对象
cameraCapture = cv.VideoCapture()

# 2. 按照指定的摄像头媒体类型，打开指定序号的摄像头
isCameraOpen = cameraCapture.open(0, cv.CAP_DSHOW)

# 判断摄像头是否已经正确打开
if isCameraOpen:
    # 获取摄像头的所有属性
    # https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    cap_properties = []
    # 摄像头参数是 字典 类型
    # 获取 OpenCV 中所有的字典类型的常数量的 key
    for key in dir(cv):
        # 摄像头参数在 OpenCV 中的 字典类型 的 key 的开头是 CAP_PROP_...
        # 判断 是否以 CAP_PROP 开头
        if key.startswith('CAP_PROP'):
            # 获取对应 key 的 value
            value = getattr(cv, key)
            # 将获取到的摄像头参数(改次序为： value,key)追加到 cap_properties 列表中
            cap_properties.append((value, key))
    # cap_properties 列表的初始顺序是按 key 加入的次序排列的
    # cap_properties 列表按照 value 进行排序
    properties = sorted(cap_properties)

    # 遍历 cap_properties 列表，按照 key 排序后的次序输出 value,key
    for value, key in properties:
        print(f' {value:5} | cv2.{key}')
else:
    print('摄像头打开失败...')

'''
    常用摄像头参数：
    CV_CAP_PROP_FPS, 30             //帧率 帧/秒
    CV_CAP_PROP_BRIGHTNESS, 1       //亮度 
    CV_CAP_PROP_CONTRAST,40         //对比度 40
    CV_CAP_PROP_SATURATION, 50      //饱和度 50
    CV_CAP_PROP_HUE, 50             //色调 50
    CV_CAP_PROP_EXPOSURE, 50        //曝光 50 获取摄像头参数
'''