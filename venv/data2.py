import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.io
import scipy.misc
from matplotlib.pyplot import imshow


cwd = '/anaconda3/category/fruit/'
classes = ['Apple', 'Avocado', 'Banana','Cantaloupe','Cherry 1', 'Lemon','Mulberry','Pear','Strawberry','Tomato 3'] # 人为设定2类
#writer = tf.python_io.TFRecordWriter("category_train.tfrecords")  # 要生成的文件

#train_data = np.array([64,64,3])
train_data=[]
# train_label=[]
# test_data=[]
# test_label=[]

for name in classes:
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个图片的地址
        #img = Image.open(img_path)
        #img = img.resize([64, 64])
        img = scipy.misc.imread(img_path)
        img = scipy.misc.imresize(img, size=(64, 64))
        img = np.reshape(img, (1,64,64,3))


        if train_data==[]:
            train_data = img
        else:
            train_data=np.concatenate((train_data,img),0)
        #imshow(img)
        #plt.show()
        #print(train_data.shape)
print(train_data[2,20,24:40,2])
train_data = np.reshape(train_data, (1420,12288))
print(train_data.shape)
np.savetxt('train_data2.csv',train_data,fmt='%d',delimiter=',')
b = np.loadtxt('train_data2.csv',delimiter=',')
print(b.shape)

b=np.reshape(b,(1420,64,64,3))
print(b[2,20,24:40,2])



label=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
label=np.repeat(label,142,axis=1)
#print(label)
label = np.reshape(label,[1,1420])
#print(label)
np.savetxt('train_label2.csv',label,fmt='%d',delimiter=',')