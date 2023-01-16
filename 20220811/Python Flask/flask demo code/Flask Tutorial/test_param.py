from flask import Flask, request
app = Flask(__name__)


# 获取path url 路径中的参数 == Restful 形式
# http://127.0.0.1:5000/user/11
@app.route('/user/<userid>')
def restful_param(userid):
    return userid

# 获取?开头的请求参数
# http://127.0.0.1:5000/user2/?userid=11
@app.route('/user2/')
def request_param():
    userid = int(request.args.get('userid'))
    return 'userid : %d' % userid

if __name__ == '__main__':
    app.run()