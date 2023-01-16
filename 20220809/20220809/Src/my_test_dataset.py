#1. 获取指定目录(子目录)下的所有的文件的名称、该文件所对应的标签名称（该文件的父目录的名称）==>> 编码 == list
import os
import pathlib

root_dir = os.getcwd() + os.sep + 'datasets/dogs-vs-cats/minidata'
root_dir_path = pathlib.Path(root_dir)
all_image_filename = [str(jpg_path) for jpg_path in root_dir_path.glob('**/*.jpg')]

all_image_label = [pathlib.Path(image_file).parent.name for image_file in all_image_filename]

all_image_unique_labelname = list(set(all_image_label))
name_index = dict((name, index) for index, name in enumerate(all_image_unique_labelname))

all_image_lable_code = [name_index[pathlib.Path(path).parent.name] for path in all_image_filename]

print('ss')


#2. 生成 dataset
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 借助文件名获取执行的图片，还有同时给出对应的标签信息
def get_image_by_filename(filename, label):
    print('aa')
    # 借助 tf 读入文件
    image_data = tf.io.read_file(filename)
    image_jpg = tf.image.decode_jpeg(image_data)
    image_resized = tf.image.resize(image_jpg, [256, 256])
    image_scale = image_resized / 255.0
    print('bb')
    return image_scale, label


tf_train_feature_filenames = tf.constant(all_image_filename)
tf_train_labels = tf.constant(all_image_lable_code)
train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_feature_filenames, tf_train_labels))

# for f, l in train_dataset:
#     print(f.numpy(), l.numpy())
#
# map
train_dataset = train_dataset.map(map_func=get_image_by_filename, num_parallel_calls=AUTOTUNE)
#
# for f, l in train_dataset:
#     print(f.numpy(), l.numpy())

#3. 生成cnn网络，完成训练
num_epochs = 10
batch_size = 32
learning_rate = 0.001

#3-0: 为了做训练，对 datset 执行一些预处置
train_dataset = train_dataset.shuffle(buffer_size=200000)
train_dataset = train_dataset.batch(batch_size=batch_size)
train_dataset = train_dataset.prefetch(AUTOTUNE)

#3-1： 使用 tf.keras 的序贯模型完成cnn网络结构的声明
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
)

model.fit(train_dataset, epochs=num_epochs)
