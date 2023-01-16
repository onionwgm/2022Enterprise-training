# 导入 OpenCV 模块
import cv2 as cv

# 设置窗口名称
WINDOW_TITLE = "Camera Capture"

# 设置帧速率
FPS = 25          # 每秒25帧
# 设置捕获的图像的宽、高
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# 1. 获取摄像头设备对象
cameraCapture = cv.VideoCapture()

def init():
    #cameraCapture.open(0)
    # 2. 打开|开启（指定序号的，默认是 0）摄像头，返回值为 bool 值
    # 需要指出 要打开|开启的摄像头的序号
    # 需要指出 摄像机的类型，即：摄像机域（camera domain）的媒体类型，这个域值可以是一预定义常量。
    # VideoCapture打开摄像头默认媒体类型是CAP_MSMF，需要指定为cv::CAP_DSHOW 类型才可以打开笔记本摄像头
    # 所有的媒体类型：https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d
    cameraCapture.open(0, cv.CAP_DSHOW)             # 对于有多个摄像头计算机，0、1是摄像头的编号，0代表笔记本自带的摄像头，如果有外接摄像头则可以选填1、2等。
    cameraCapture.get(cv.VideoCaptureProperties)

    # 判断是否开启了摄像头
    if not cameraCapture.isOpened():
        print("无法打开摄像头！")
        return False

    # 略去摄像头初始化过程中的捕获的图片
    for i in range(5):               # 过滤掉前若干帧画面。这些画面因为设备初始化尚未完全完成等原因，拍摄出来的照片不正常
        cameraCapture.read()

    # 3. 获取/设置摄像头属性参数
    # 所有的摄像头属性见：https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    # 获取所有的摄像头参数，名称，对应序号，参看：03_02_camera_properties.py

    # 在open之后才能获得摄像头属性：拍照的宽、高和帧频率
    width = cameraCapture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cameraCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cameraCapture.get(cv.CAP_PROP_FPS)
    # 如果当前驱动不支持获取某个属性值，则该属性值返回0
    print("摄像头捕获宽度：%d, 高度：%d, FPS:%d" % (width, height, fps))

    # 设置摄像头参数：拍照的宽、高
    cameraCapture.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cameraCapture.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 摄像头初始化结束
    return True


# 处理：显示摄像头获取的图像
def process():
    # 4. read方法用于读取1帧数据。
    # read方法返回两个值，第一个值用于标记读取数据是否成功，第二个值存放读取到的数据
    success, frame = cameraCapture.read()
    # 将采集到的数据（图像）显示在指定的命名窗口中
    cv.imshow(WINDOW_TITLE, frame)


if __name__ == "__main__":
    # 初始化摄像头
    init()

    # 指定名称的命名窗口
    cv.namedWindow(WINDOW_TITLE)

    # 设置获取键盘输入的指定时间间隔
    interval = 1000 // FPS      # 计算每张照片采集的时间间隔(毫秒)

    # 开始采集并显示视频
    while True:
        # 获取键盘输入，放入key中，如果没有按键，key返回值为 -1
        # 如果在指定的时间间隔内有按键按下，则执行相应操作
        key = cv.waitKey(interval)
        if key > 0:             # 如果按下任意键，则退出循环
            break
        # 没有按键，则执行处理
        process()

    cv.destroyAllWindows()