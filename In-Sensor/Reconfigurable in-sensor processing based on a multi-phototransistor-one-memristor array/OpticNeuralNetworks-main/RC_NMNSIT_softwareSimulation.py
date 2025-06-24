<<<<<<< HEAD
import os
import numpy as np
import cv2
import easyesn
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

def load_nmnist_data(path):
    """
    Load N-MNIST dataset from the given path.
    Assumes the dataset is stored in folders 'train' and 'test', 
    with subfolders for each class.
    """
    def load_images_from_folder(folder):
        images = []
        labels = []
        for class_idx, class_folder in enumerate(sorted(os.listdir(folder))):
            class_folder_path = os.path.join(folder, class_folder)
            for filename in sorted(os.listdir(class_folder_path)):
                img_path = os.path.join(class_folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        return np.array(images), np.array(labels)
    
    x_train, y_train = load_images_from_folder(os.path.join(path, 'train'))
    x_test, y_test = load_images_from_folder(os.path.join(path, 'test'))
    
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, x_test, target_size=(20, 20), time_steps=10):
    """
    Preprocess the data by resizing and reshaping.
    """
    x_train_resized = np.array([cv2.resize(img, target_size) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, target_size) for img in x_test])
    
    x_train_reshaped = x_train_resized.reshape(-1, time_steps, target_size[0], target_size[1])
    x_test_reshaped = x_test_resized.reshape(-1, time_steps, target_size[0], target_size[1])
    
    return x_train_reshaped, x_test_reshaped

def create_reservoir_layer(input_shape, n_reservoir=20, spectral_radius=0.9, noise_level=0.01):
    """
    Create an Echo State Network (Reservoir Computing Layer) using easyesn.
    """
    esn = easyesn.ESN(n_input=input_shape[1] * input_shape[2],
                      n_output=n_reservoir,
                      n_reservoir=n_reservoir,
                      spectralRadius=spectral_radius,
                      noiseLevel=noise_level)
    return esn

def transform_data_with_esn(esn, data):
    """
    Transform data using the Echo State Network.
    """
    transformed_data = []
    for sequence in data:
        sequence = sequence.reshape(sequence.shape[0], -1)
        esn_output = esn.simulate(sequence)
        transformed_data.append(esn_output[-1])
    return np.array(transformed_data)

def plot_samples(x, y, num_samples=10):
    """
    Plot sample images from the dataset.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x[i].reshape(20, 20), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

def plot_weights(model):
    """
    Plot the weights of the model's layers.
    """
    for layer in model.layers:
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f'{layer.name} weights')
            plt.hist(weights.flatten(), bins=50)
            plt.subplot(1, 2, 2)
            plt.title(f'{layer.name} biases')
            plt.hist(biases.flatten(), bins=50)
            plt.show()

def create_classification_model(input_shape):
    """
    Create a more complex classification model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Load and preprocess data
x_train, y_train, x_test, y_test = load_nmnist_data('./n_mnist')
x_train, x_test = preprocess_data(x_train, x_test)

# Plot sample images from the dataset
plot_samples(x_train[:, 0], y_train, num_samples=10)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Create and train ESN
esn = create_reservoir_layer(x_train.shape, n_reservoir=20, spectral_radius=1.2, noise_level=0.05)

# Transform data with ESN
x_train_transformed = transform_data_with_esn(esn, x_train)
x_test_transformed = transform_data_with_esn(esn, x_test)

# Create and compile the model
model = create_classification_model(x_train_transformed.shape[1])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with validation
history = model.fit(x_train_transformed, y_train, epochs=200, batch_size=64, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy: {accuracy}')

# Classification report
y_pred = model.predict(x_test_transformed)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Plot model weights
plot_weights(model)

# Apply 4-bit quantization
def apply_4bit_quantization(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_model = quantize_model(model, tfmot.experimental.combine.Default8BitQuantizeScheme())
    
    # Recompile the quantized model
    q_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return q_model

# Quantize the model
q_model = apply_4bit_quantization(model)

# Train the quantized model with validation
q_history = q_model.fit(x_train_transformed, y_train, epochs=200, batch_size=64, validation_split=0.2)

# Evaluate the quantized model
q_loss, q_accuracy = q_model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy after quantization: {q_accuracy}')

# Plot training & validation accuracy values for quantized model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(q_history.history['accuracy'])
plt.plot(q_history.history['val_accuracy'])
plt.title('Quantized model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for quantized model
plt.subplot(1, 2, 2)
plt.plot(q_history.history['loss'])
plt.plot(q_history.history['val_loss'])
plt.title('Quantized model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
=======
import os
import numpy as np
import cv2
import easyesn
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

def load_nmnist_data(path):
    """
    Load N-MNIST dataset from the given path.
    Assumes the dataset is stored in folders 'train' and 'test', 
    with subfolders for each class.
    """
    def load_images_from_folder(folder):
        images = []
        labels = []
        for class_idx, class_folder in enumerate(sorted(os.listdir(folder))):
            class_folder_path = os.path.join(folder, class_folder)
            for filename in sorted(os.listdir(class_folder_path)):
                img_path = os.path.join(class_folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        return np.array(images), np.array(labels)
    
    x_train, y_train = load_images_from_folder(os.path.join(path, 'train'))
    x_test, y_test = load_images_from_folder(os.path.join(path, 'test'))
    
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, x_test, target_size=(20, 20), time_steps=10):
    """
    Preprocess the data by resizing and reshaping.
    """
    x_train_resized = np.array([cv2.resize(img, target_size) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, target_size) for img in x_test])
    
    x_train_reshaped = x_train_resized.reshape(-1, time_steps, target_size[0], target_size[1])
    x_test_reshaped = x_test_resized.reshape(-1, time_steps, target_size[0], target_size[1])
    
    return x_train_reshaped, x_test_reshaped

def create_reservoir_layer(input_shape, n_reservoir=20, spectral_radius=0.9, noise_level=0.01):
    """
    Create an Echo State Network (Reservoir Computing Layer) using easyesn.
    """
    esn = easyesn.ESN(n_input=input_shape[1] * input_shape[2],
                      n_output=n_reservoir,
                      n_reservoir=n_reservoir,
                      spectralRadius=spectral_radius,
                      noiseLevel=noise_level)
    return esn

def transform_data_with_esn(esn, data):
    """
    Transform data using the Echo State Network.
    """
    transformed_data = []
    for sequence in data:
        sequence = sequence.reshape(sequence.shape[0], -1)
        esn_output = esn.simulate(sequence)
        transformed_data.append(esn_output[-1])
    return np.array(transformed_data)

def plot_samples(x, y, num_samples=10):
    """
    Plot sample images from the dataset.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x[i].reshape(20, 20), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

def plot_weights(model):
    """
    Plot the weights of the model's layers.
    """
    for layer in model.layers:
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f'{layer.name} weights')
            plt.hist(weights.flatten(), bins=50)
            plt.subplot(1, 2, 2)
            plt.title(f'{layer.name} biases')
            plt.hist(biases.flatten(), bins=50)
            plt.show()

def create_classification_model(input_shape):
    """
    Create a more complex classification model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Load and preprocess data
x_train, y_train, x_test, y_test = load_nmnist_data('./n_mnist')
x_train, x_test = preprocess_data(x_train, x_test)

# Plot sample images from the dataset
plot_samples(x_train[:, 0], y_train, num_samples=10)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Create and train ESN
esn = create_reservoir_layer(x_train.shape, n_reservoir=20, spectral_radius=1.2, noise_level=0.05)

# Transform data with ESN
x_train_transformed = transform_data_with_esn(esn, x_train)
x_test_transformed = transform_data_with_esn(esn, x_test)

# Create and compile the model
model = create_classification_model(x_train_transformed.shape[1])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with validation
history = model.fit(x_train_transformed, y_train, epochs=200, batch_size=64, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy: {accuracy}')

# Classification report
y_pred = model.predict(x_test_transformed)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Plot model weights
plot_weights(model)

# Apply 4-bit quantization
def apply_4bit_quantization(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_model = quantize_model(model, tfmot.experimental.combine.Default8BitQuantizeScheme())
    
    # Recompile the quantized model
    q_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return q_model

# Quantize the model
q_model = apply_4bit_quantization(model)

# Train the quantized model with validation
q_history = q_model.fit(x_train_transformed, y_train, epochs=200, batch_size=64, validation_split=0.2)

# Evaluate the quantized model
q_loss, q_accuracy = q_model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy after quantization: {q_accuracy}')

# Plot training & validation accuracy values for quantized model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(q_history.history['accuracy'])
plt.plot(q_history.history['val_accuracy'])
plt.title('Quantized model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for quantized model
plt.subplot(1, 2, 2)
plt.plot(q_history.history['loss'])
plt.plot(q_history.history['val_loss'])
plt.title('Quantized model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
