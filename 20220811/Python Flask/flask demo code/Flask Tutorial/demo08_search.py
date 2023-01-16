# 导入Flask类
from flask import Flask
from flask import request, render_template
# 使用类创建一个Flask实例对象app
app = Flask(__name__)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['development'])

@app.route('/search/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # 接收客户端请求数据（表单提交的数据处理）
        keywords = request.values.get('keywords', None)
        # 处理请求数据
        msg = '搜索关键字：{0}'.format(keywords)
        # 响应客户端
        return render_template('search.html', msg=msg)
    else:
        # 页面跳转
        return render_template('search.html')

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()