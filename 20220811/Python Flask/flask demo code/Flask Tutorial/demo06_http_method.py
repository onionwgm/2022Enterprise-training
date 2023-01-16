# 导入Flask类
from flask import Flask
from flask import request
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['development'])

# 当前Falsk框架的所有业务处理

@app.route('/method', methods=['GET', 'POST'])
def handler_method():
    # 判断请求的方式
    if request.method == 'POST':
        # POST请求处理
        return '接收POST请求……'
    else:
        # GET请求处理
        return '接收GET请求……'


# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()