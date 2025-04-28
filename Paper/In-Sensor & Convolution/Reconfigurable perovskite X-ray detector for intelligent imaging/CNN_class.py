<<<<<<< HEAD
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
import tensorflow as tf
tf.device('/gpu:2')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, Add,Flatten,Dense
from tensorflow.keras.utils import to_categorical
import scipy.io
import numpy as np
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
print('[*] run basic configs ... ')
save_path = r'D:\howdy\CNN\train_logs'
save_path = os.path.join(save_path, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
checkpoint_dir = os.path.join(save_path, "checkpoint")
best_checkpoint_dir = os.path.join(save_path, "best_checkpoint")
traindata_path_image = r'D:\howdy\CNN\image_3D'
traindata_path_label = r'D:\howdy\CNN\image_address'

batchsize = 4
EPOCHS = 10
SaveEpoch=10
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE

accuracy=[]
prob=[]
label_address=[]
def data_process2(data_path_image, data_path_label):


    X = scipy.io.loadmat(data_path_image)
    Y = scipy.io.loadmat(data_path_label)
    image = X['image_3D'].astype(np.float32)/255
    image = np.rollaxis(image, 2, 0)
    image = image[..., np.newaxis]

    label = Y['image_address'].astype(np.int32)
    label = np.rollaxis(label, 1, 0)
    return image,label

# Input Pipeline
print('[*] load data ... ')
image,label = data_process2(traindata_path_image,traindata_path_label)
train_dataset = tf.data.Dataset.from_tensor_slices((image[:160], label[:160]))
train_dataset = train_dataset.batch(batchsize)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# test
test_dataset = tf.data.Dataset.from_tensor_slices((image[160:], label[160:]))
test_dataset = test_dataset.batch(1)

# model
network = Sequential([Conv2D(4, 3, activation='softmax',padding='valid')
]
                     )
network.build(input_shape=(None,3,3,1))
network.summary()

optimizer = optimizers.Adam(lr=0.001)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()
for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            out = network(x)
            out1 = tf.squeeze(out)
            y_onehot = to_categorical(y,4)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out1))
            loss_meter.update_state(loss)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
    # evaluate
    total, total_correct = 0., 0
    acc_meter.reset_states()
    for step, (x1, y1) in enumerate(test_dataset):
        out = network(x1)
        y1 = tf.squeeze(y1)
        label_address.append(y1)
        out1 = tf.squeeze(out)
        out2=out1.numpy()
        prob.append(out2)
        pred = tf.argmax(out1, axis=0)
        pred1 = tf.cast(pred, dtype=tf.int32)
        # bool type
        correct = tf.equal(pred1, y1)
        # bool tensor => int tensor => numpy
        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
        total += x1.shape[0]
        acc_meter.update_state(y1, pred1)
    accuracy.append(acc_meter.result().numpy())
    print(epoch, 'Evaluate Acc:',  acc_meter.result().numpy())
np.savetxt(os.path.join("./result", 'accuracy.txt'), np.asarray(accuracy), fmt='%.2f')
np.savetxt(os.path.join("./result", 'label.txt'), np.asarray(label_address), fmt='%d')
=======
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
import tensorflow as tf
tf.device('/gpu:2')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, Add,Flatten,Dense
from tensorflow.keras.utils import to_categorical
import scipy.io
import numpy as np
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
print('[*] run basic configs ... ')
save_path = r'D:\howdy\CNN\train_logs'
save_path = os.path.join(save_path, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
checkpoint_dir = os.path.join(save_path, "checkpoint")
best_checkpoint_dir = os.path.join(save_path, "best_checkpoint")
traindata_path_image = r'D:\howdy\CNN\image_3D'
traindata_path_label = r'D:\howdy\CNN\image_address'

batchsize = 4
EPOCHS = 10
SaveEpoch=10
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE

accuracy=[]
prob=[]
label_address=[]
def data_process2(data_path_image, data_path_label):


    X = scipy.io.loadmat(data_path_image)
    Y = scipy.io.loadmat(data_path_label)
    image = X['image_3D'].astype(np.float32)/255
    image = np.rollaxis(image, 2, 0)
    image = image[..., np.newaxis]

    label = Y['image_address'].astype(np.int32)
    label = np.rollaxis(label, 1, 0)
    return image,label

# Input Pipeline
print('[*] load data ... ')
image,label = data_process2(traindata_path_image,traindata_path_label)
train_dataset = tf.data.Dataset.from_tensor_slices((image[:160], label[:160]))
train_dataset = train_dataset.batch(batchsize)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# test
test_dataset = tf.data.Dataset.from_tensor_slices((image[160:], label[160:]))
test_dataset = test_dataset.batch(1)

# model
network = Sequential([Conv2D(4, 3, activation='softmax',padding='valid')
]
                     )
network.build(input_shape=(None,3,3,1))
network.summary()

optimizer = optimizers.Adam(lr=0.001)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()
for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            out = network(x)
            out1 = tf.squeeze(out)
            y_onehot = to_categorical(y,4)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out1))
            loss_meter.update_state(loss)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
    # evaluate
    total, total_correct = 0., 0
    acc_meter.reset_states()
    for step, (x1, y1) in enumerate(test_dataset):
        out = network(x1)
        y1 = tf.squeeze(y1)
        label_address.append(y1)
        out1 = tf.squeeze(out)
        out2=out1.numpy()
        prob.append(out2)
        pred = tf.argmax(out1, axis=0)
        pred1 = tf.cast(pred, dtype=tf.int32)
        # bool type
        correct = tf.equal(pred1, y1)
        # bool tensor => int tensor => numpy
        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
        total += x1.shape[0]
        acc_meter.update_state(y1, pred1)
    accuracy.append(acc_meter.result().numpy())
    print(epoch, 'Evaluate Acc:',  acc_meter.result().numpy())
np.savetxt(os.path.join("./result", 'accuracy.txt'), np.asarray(accuracy), fmt='%.2f')
np.savetxt(os.path.join("./result", 'label.txt'), np.asarray(label_address), fmt='%d')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
np.savetxt(os.path.join("./result", 'output_prob.txt'), np.asarray(prob),fmt='%.2f')