# 导入Flask类
from flask import Flask
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['product'])

# 当前Falsk框架的所有业务处理

'''
    创建一个处理函数，绑定客户端的访问地址（根目录访问）
    规范标准：使用装饰器@app.route(url地址)修饰绑定处理函数
'''
@app.route('/')
def index():
    # 读取全局配置参数
    param1 = app.config['GLOBAL_PARAM']
    param2 = app.config['DEBUG']
    # 响应客户端
    return '当前的全局配置参数：<br/>GLOBAL_PARAM:{0}<br/>DEBUG:{1}'.format(param1, param2)

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()