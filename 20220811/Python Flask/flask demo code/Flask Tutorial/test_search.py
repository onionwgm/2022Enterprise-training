from flask import Flask, request, render_template

app = Flask(__name__)

# 以 post 的方式访问 /search/ url
@app.route('/search/', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':  # 地址栏中GET方法的访问
        return render_template('search.html')
    else:   # form表单中POST方法的访问
        # 获取表单中输入的数据
        keywords = request.values.get('keywords', None)
        msg = '搜索内容：{0}'.format(keywords)
        return render_template('search.html', msg=msg)

# # 访问 url ，返回|在浏览器中显示 search.html
# @app.route('/search/')
# def search():
#     # 对指定的 html 页面进行渲染后返回
#     return render_template('search.html')


if __name__ == '__main__':
    app.run()