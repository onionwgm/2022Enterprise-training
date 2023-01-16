# 导入Flask类
from flask import Flask
from flask import request, render_template, Response, session
# 使用类创建一个Flask实例对象app
app = Flask(__name__)
# 生成secret_key
import os
app.secret_key = os.urandom(24)

'''
    导入配置脚本中的配置类
    方式：app.config.from_object(confg['配置类Key值'])
'''
from settings import config
app.config.from_object(config['development'])

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 接收客户端请求数据（表单提交的数据处理）
        account = request.values.get('account', None)
        password = request.values.get('password', None)
        # 处理请求数据
        if account == 'zhangsan' and password == '123':
            # 将正确的账号存入到session会话中
            session['account'] = account
            # 获取复选框中的值
            chkCookie = request.values.get('chkCookie', None)
            # 判断是否选中
            if chkCookie:
                # 创建一个响应对象，实现页面跳转传值的操作
                resp = Response(render_template('login.html', msg='你好，{0}'.format(account)))
                # 通过响应对象创建一个Cookie
                resp.set_cookie('account', account)
                # 响应客户端
                return resp
            else:
                # 响应客户端
                # return render_template('login.html', msg='你好，{0}'.format(account))
                # 从session中获取登录账号
                accont = session.get('account', None)
                return render_template('home.html', account=account)
        else:
            # 响应客户端
            return render_template('login.html', msg='账号或密码错误')
    else:
        # 获取Cookie中的登录账号
        account = request.cookies.get('account', None)
        # 判断是否存在
        if account:
            # 页面跳转
            return render_template('login.html', account=account)
        else:
            # 页面跳转
            return render_template('login.html')

@app.route('/logout/')
def logout():
    # 将登录账号从session中删除
    session.pop('account', None)
    return render_template('login.html')

# 创建该脚本的程序入口
if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()