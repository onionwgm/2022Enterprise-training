# 导入Flask类
from flask import Flask
from flask import request, render_template, Response
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['development'])

# 设置一个cookie
@app.route('/cookie/set')
def set_cookie():
    # 创建一个Response对象
    resp = Response('设置Cookie')
    # 使用resp对象函数设置Cookie
    resp.set_cookie('username', 'alvin')
    # 响应客户端
    return resp

# 获取Cookie
@app.route('/cookie/get')
def get_cookie():
    # 使用request对象函数获取cookie值
    username = request.cookies.get('username', None)
    # 响应客户端
    return 'Cookie:{0}'.format(username)

# 删除Cookie
@app.route('/cookie/delete')
def delete_cookie():
    # 创建一个Response对象
    resp = Response('删除Cookie中的值')
    # 使用resp对象函数删除Cookie
    resp.delete_cookie('username')
    # 响应客户端
    return resp

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()