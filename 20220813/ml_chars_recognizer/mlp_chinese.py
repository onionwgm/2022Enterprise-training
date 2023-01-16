import cv2 as cv
import numpy as np
import os
# 导入 多次感知机的分类模型 MLPClassifier
from sklearn.neural_network import MLPClassifier
# 导入 完成模型持久化的 joblib
import joblib
# 导入 ml 工具类
import ml_train_utility

# 初始配置
# 训练集、测试集位置
TRAIN_DIR = '../data/mlanddl/chs_train'
TEST_DIR = '../data/mlanddl/chs_test'
# 汉字图片宽、高
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48
# 给出类别数、类别-数值 字典
CLASSIFICATION_COUNT = 31
LABEL_DICT = {
	'chuan':0, 'e':1, 'gan':2, 'gan1':3, 'gui':4, 'gui1':5, 'hei':6, 'hu':7, 'ji':8, 'jin':9,
	'jing':10, 'jl':11, 'liao':12, 'lu':13, 'meng':14, 'min':15, 'ning':16, 'qing':17,	'qiong':18, 'shan':19,
	'su':20, 'sx':21, 'wan':22, 'xiang':23, 'xin':24, 'yu':25, 'yu1':26, 'yue':27, 'yun':28, 'zang':29,
	'zhe':30
}
# 模型持久化的位置
MLP_ENU_MODEL_PATH = '../model/mlp/mlp_chs.m'

# 3. 训练 + 保存
def train():
	# 加载训练数据
	train_data, train_labels = ml_train_utility.load_data(TRAIN_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, LABEL_DICT)
	# 数据的预处理
	normalized_data = ml_train_utility.normalize_data(train_data)

	# 模型创建
	model = MLPClassifier(hidden_layer_sizes=(48, 24), solver='lbfgs', alpha=1e-5, random_state=42)

	# 模型训练
	model.fit(normalized_data, train_labels)

	# 模型保存
	joblib.dump(model, MLP_ENU_MODEL_PATH)


# 4. 测试 + 评估
def test():
	# test_data, test_labels = load_data(TEST_DIR)
	test_data, test_labels = ml_train_utility.load_data(TEST_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, LABEL_DICT)
	# normalized_data = normalize_data(test_data)
	normalized_data = ml_train_utility.normalize_data(test_data)

	model = joblib.load(MLP_ENU_MODEL_PATH)

	predicts = model.predict(normalized_data)

	errors = np.count_nonzero(predicts-test_labels)
	print(errors)
	print((len(predicts)-errors) / len(predicts))


if __name__ == '__main__':
	train()
	test()
