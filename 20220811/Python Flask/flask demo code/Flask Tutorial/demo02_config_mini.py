# 导入Flask类
from flask import Flask
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    在脚本中直接写入应用的配置参数
    方式1：app.config['key参数名称']=value
'''
# 开启Flask框架的调试模式
app.config['DEBUG'] = True
app.config['MY_PARAMS'] = 10

'''
    在脚本中直接写入应用的配置参数
    方式2：app.config.update()函数快速实现
'''
# 开启Flask框架的调试模式
app.config.update(
    PARAM1 = '测试数据',
    PARAM2 = 20
)

# 当前Falsk框架的所有业务处理

'''
    创建一个处理函数，绑定客户端的访问地址（根目录访问）
    规范标准：使用装饰器@app.route(url地址)修饰绑定处理函数
'''
@app.route('/')
def index():
    # 读取应用配置参数
    my_params = app.config['MY_PARAMS']
    param1 = app.config['PARAM1']
    param2 = app.config['PARAM2']
    # 响应客户端
    return '你好，Flask.<br/>当前的应用配置参数[MY_PARMAS]:{0}<br/>param1:{1}<br/>param2:{2}'.format(my_params, param1, param2)

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()