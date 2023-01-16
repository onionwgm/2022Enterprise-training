# 导入Flask类
from flask import Flask
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['development'])

# 当前Falsk框架的所有业务处理

'''
    创建一个处理函数，绑定客户端的访问地址（根目录访问）
    规范标准：使用装饰器@app.route(url地址)修饰绑定处理函数
'''
@app.route('/user/<username>/<int:age>')
def handlerInfo(username, age):
    # 响应客户端
    return '用户名称:{0}<br/>用户年龄：{1}'.format(username, age)

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()