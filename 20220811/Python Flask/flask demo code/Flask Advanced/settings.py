#-*- coding:utf-8 -*-
'''
   settings.py
   ~~~~~~~~~~~~~~~~~~~~
   工程独立的配置文件
'''

# 创建配置类（实际开发标准）
# 导入os
import os
# 获取工程绝对路径
basedir = os.path.abspath(os.path.dirname(__name__))

# 创建工程的基础配置类
class BasicConfig:
    GLOBAL_PARAM = '基础配置类中的参数数据'
    UPLOAD_FOLDER = '/upload/'
    ALLOWED_EXTENSIONS = set ( ['.png', '.jpg', '.jpeg', '.gif'] ) 
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    pass

# 创建场景1：开发模式的配置类
class DevelopmentConfig(BasicConfig):
    DEBUG = True
    pass

# 创建场景2：生产环境下的配置类
class ProductConfig(BasicConfig):
    DEBUG = False
    pass

# 创建一个字典列表，对照所有的配置场景类
config = {
    'default':DevelopmentConfig,
    'development': DevelopmentConfig,
    'product': ProductConfig
}