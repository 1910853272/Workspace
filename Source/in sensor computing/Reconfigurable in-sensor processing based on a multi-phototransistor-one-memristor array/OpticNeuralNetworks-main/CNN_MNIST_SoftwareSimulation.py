import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

class BatchAccuracyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_accuracies = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_accuracies.append(logs.get('accuracy'))

def weight_constraint(w):
    return K.clip(w, 0, 1)

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28))
train_images = np.asarray([np.asarray(Image.fromarray(image).resize((20, 20))) for image in train_images])
train_images = train_images.reshape((60000, 20, 20, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28))
test_images = np.asarray([np.asarray(Image.fromarray(image).resize((20, 20))) for image in test_images])
test_images = test_images.reshape((10000, 20, 20, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Set the model parameters
input_shape = (20, 20, 1)
num_classes = 10

# Create the CNN model
model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_constraint=weight_constraint))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', input_shape=(400,), kernel_constraint=weight_constraint))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_accuracy_callback = BatchAccuracyCallback()
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), callbacks=[batch_accuracy_callback])

# Get the weights of the dense layer
dense_layer_weights = model.layers[-1].get_weights()[0]
plt.figure(1)
plt.hist(dense_layer_weights.flatten(), bins=50, color='blue')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Histogram of Dense Layer Weights')
plt.show()

# Save the weights as a CSV file
np.savetxt('dense_layer_weights.csv', dense_layer_weights, delimiter=',')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Plot the training and validation accuracy per epoch
plt.figure(2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(batch_accuracy_callback.batch_accuracies, label='Training accuracy per batch')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy per Batch')
plt.legend()
plt.show()

# Select the image to plot the heatmap
image_index = 1

# Convert the image to a heatmap
heatmap = train_images[image_index]
plt.figure()
# Plot the heatmap
plt.imshow(heatmap, cmap='hot')
plt.colorbar()
plt.show()



# Apply 4-bit quantization to the model
quantize_model = tfmot.quantization.keras.quantize_annotate_model(model)
quantize_model = tfmot.quantization.keras.quantize_apply(quantize_model)

# Compile the quantized model
quantize_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the quantized model
history_quantized = quantize_model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), callbacks=[batch_accuracy_callback])

# Evaluate the quantized model
test_loss_quantized, test_acc_quantized = quantize_model.evaluate(test_images, test_labels)
print('Quantized test accuracy:', test_acc_quantized)
