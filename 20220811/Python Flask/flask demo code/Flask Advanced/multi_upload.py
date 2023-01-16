#-*- coding:utf-8 -*-
from flask import Flask
from flask import render_template, request, send_from_directory
import os
# from werkzeug import secure_filename
from werkzeug.utils import secure_filename

# 创建Flask实例
app = Flask(__name__)

# 导入Flask核心配置文件
from settings import config
app.config.from_object(config['development'])

# 业务处理的实现

@app.route('/file/multiupload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 获取客户端上传的文件对象
        uploadFiles = request.files.getlist('uploadFile[]')
        # 创建一个列表用于存放所有上上传文件的名称
        fileNames = []
        # 循环上传文件的对象
        for uploadFile in uploadFiles:
            # 获取上传文件的名称
            fileName = uploadFile.filename
            # 验证上传文件的类型
            extName = os.path.splitext(fileName)[1]
            # 判断
            if extName in app.config['ALLOWED_EXTENSIONS']:
                # 获取上传文件夹的路径
                uploadPath = os.path.dirname(os.path.realpath(__file__))
                # 上传文件
                uploadFile.save(uploadPath + app.config['UPLOAD_FOLDER'] + secure_filename(fileName))
                # 将当前的上传文件名称存放到类表中
                fileNames.append(secure_filename(fileName))
            else:
                # 响应客户端
                return render_template('multi_upload.html', msg='文件类型错误.')
        # 响应客户端
        return render_template('multi_upload.html', msg='文件上传成功.', fileNames = fileNames)
    else:
        # 页面跳转
        return render_template('multi_upload.html')

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory(os.getcwd() + '/upload/' , filename, as_attachment=True)

# 启动框架
if __name__ == '__main__':
    app.run()