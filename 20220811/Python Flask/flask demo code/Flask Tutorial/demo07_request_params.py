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

@app.route('/method/', methods=['GET', 'POST'])
def handler_method():
   # 接收客户端请求参数
   userid = int(request.args.get('userid', None))
   print(type(userid))
   # 响应客户端
   return '参数userid:{0}'.format(userid)


# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()