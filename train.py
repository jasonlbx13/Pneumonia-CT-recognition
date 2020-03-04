from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils import multi_gpu_model
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pynvml import *
import time

nvmlInit()
GPUNUM = nvmlDeviceGetCount()
INPUT_SIZE = 224
if GPUNUM>1:
    BATCHSIZE = 64
else:
    BATCHSIZE = 16

EPOCH = 12

# 配置GPU
if GPUNUM == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为0号的GPU
elif GPUNUM == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用编号为0,1号的GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 每个GPU现存上届控制在80%以内
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

base_model = InceptionV3(weights='imagenet', include_top=False)  # 接上全连接头部
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[:249]:  #冻结骨架
    layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

if GPUNUM>=2:
    parallel_model = multi_gpu_model(model, gpus=GPUNUM) # 设置使用4个gpu，该句放在模型compile之前
    parallel_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])
else:
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

def read_img(img_id, data_dir, train_or_test, size):

    img = image.load_img(os.path.join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

def train(model_num):
    train_csv = pd.read_csv('./data/train_{}.csv'.format(model_num))
    val_csv = pd.read_csv('./data/val_{}.csv'.format(model_num))
    x_train = np.zeros((len(train_csv), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    x_val = np.zeros((len(val_csv), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    print ('making train file.................')
    for i, img_id in tqdm(enumerate(train_csv['id'])):
        img = read_img(img_id, './', 'train', (INPUT_SIZE, INPUT_SIZE))
        x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_train[i] = x
    print ('train file made completed!trainfile_shape:{}'.format(len(x_train)))
    print ('making val file.................')
    for i, img_id in tqdm(enumerate(val_csv['id'])):
        img = read_img(img_id, './', 'train', (INPUT_SIZE, INPUT_SIZE))
        x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_val[i] = x
    print ('val file made completed!valfile_shape:{}'.format(len(x_val)))


    y_train = to_categorical(np.asarray(list(train_csv['label'])))
    y_val = to_categorical(np.asarray(list(val_csv['label'])))

    train_datagen = image.ImageDataGenerator(rotation_range=45,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.25,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    test_datagen = image.ImageDataGenerator()

    print ('start training......')
    if GPUNUM>1:
        hist = parallel_model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCHSIZE),
                                   steps_per_epoch=int(len(x_train)) // BATCHSIZE,
                                   epochs=EPOCH,
                                   validation_data=test_datagen.flow(x_val, y_val, batch_size=BATCHSIZE),
                                   validation_steps=(len(x_val)) // BATCHSIZE,
                                   verbose=1)
    else:
        hist = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCHSIZE),
                                   steps_per_epoch=int(len(x_train)) // BATCHSIZE,
                                   epochs=EPOCH,
                                   validation_data=test_datagen.flow(x_val, y_val, batch_size=BATCHSIZE),
                                   validation_steps=(len(x_val)) // BATCHSIZE,
                                   verbose=1)
    print ('training complete!')
    print ('saving model......')
    model.save('inceptionv3_model_{}'.format(model_num))
    print ('inceptionv3_model_{} saved.'.format(model_num))

if __name__ == '__main__':

    for i in range(1,6):
        train(i)