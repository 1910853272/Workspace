import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test_original = x_test_original / 255.0

# Reshape images to add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test_original = x_test_original.reshape(-1, 28, 28, 1)

# Device parameters
R_form_LH = 0.962
R_form_RH = 0.0383
S_form_LH = 0.0492
S_form_RH = 0.951
intensity_LH = 0.5
intensity_RH = 1 - intensity_LH


# Function to create mixed dataset based on device parameters
def create_device_mixed_data(x_data, y_data, LH_param, RH_param, label_type='label1'):
    n = x_data.shape[0]
    mixed_images = []
    mixed_labels = []
    for i in range(n - 1):
        img1, label1 = x_data[i], y_data[i]
        img2, label2 = x_data[i + 1], y_data[i + 1]
        mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH
        mixed_img = mixed_img / np.max(mixed_img)  # Normalize to [0,1]
        mixed_images.append(mixed_img)
        mixed_labels.append(label1 if label_type == 'label1' else label2)
    # Handle the last image
    img1, label1 = x_data[-1], y_data[-1]
    img2, label2 = x_data[0], y_data[0]
    mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH
    mixed_img = mixed_img / np.max(mixed_img)
    mixed_images.append(mixed_img)
    mixed_labels.append(label1 if label_type == 'label1' else label2)
    return np.array(mixed_images), np.array(mixed_labels)


# Create R-form and S-form datasets
x_test_R_form, y_test_R_form = create_device_mixed_data(
    x_test_original, y_test_original, R_form_LH, R_form_RH, label_type='label1')
x_test_S_form, y_test_S_form = create_device_mixed_data(
    x_test_original, y_test_original, S_form_LH, S_form_RH, label_type='label2')
# Create mixed dataset without device parameters (for comparison)
x_test_mixed, y_test_mixed = create_device_mixed_data(
    x_test_original, y_test_original, 1.0, 1.0, label_type='label1')

# Prepare labels for categorical crossentropy loss
y_train_cat = to_categorical(y_train, 10)
y_test_original_cat = to_categorical(y_test_original, 10)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation settings
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

batch_size = 64
epochs = 20  # Changed epoch count to 20
train_gen = datagen.flow(x_train, y_train_cat, batch_size=batch_size)
test_gen = datagen.flow(x_test_original, y_test_original_cat, batch_size=batch_size)

# Prepare directory for results (모든 파일은 result 폴더에 저장)
os.makedirs('results', exist_ok=True)

# Initialize array to store accuracies
accuracy_data = np.empty((epochs, 5))

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(train_gen, epochs=1, validation_data=test_gen, verbose=1)

    # Evaluate on original test set
    pred_test = np.argmax(model.predict(x_test_original), axis=-1)
    test_acc = np.mean(pred_test == y_test_original.flatten())

    # Evaluate on mixed dataset
    pred_mixed = np.argmax(model.predict(x_test_mixed), axis=-1)
    test_acc_mixed = np.mean(pred_mixed == y_test_mixed.flatten())

    # Evaluate on R-form dataset
    pred_R_form = np.argmax(model.predict(x_test_R_form), axis=-1)
    test_acc_R_form = np.mean(pred_R_form == y_test_R_form.flatten())

    # Evaluate on S-form dataset
    pred_S_form = np.argmax(model.predict(x_test_S_form), axis=-1)
    test_acc_S_form = np.mean(pred_S_form == y_test_S_form.flatten())

    # Store accuracies: [epoch, original, mixed, R-form, S-form]
    accuracy_data[epoch] = [epoch + 1, test_acc, test_acc_mixed, test_acc_R_form, test_acc_S_form]

    # Print accuracies
    print(f"Test Accuracy (Original): {test_acc:.4f}")
    print(f"Test Accuracy (Mixed): {test_acc_mixed:.4f}")
    print(f"Test Accuracy (R-form): {test_acc_R_form:.4f}")
    print(f"Test Accuracy (S-form): {test_acc_S_form:.4f}")

# Save accuracy data in CSV format
np.savetxt('results/MNIST_accuracy_data.csv', accuracy_data, fmt='%1.4f', delimiter=',')


# Function to save confusion matrix as heatmap and numerical data
def save_confusion_matrix(y_true, y_pred, filename_image, filename_data):
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix as image
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, square=True, annot=False, fmt='d', cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename_image)
    plt.close()  # Explicitly close the figure to free up resources

    # Save confusion matrix as numerical data (CSV)
    np.savetxt(filename_data, cm, fmt='%d', delimiter=',')


# Save confusion matrices using final epoch predictions
save_confusion_matrix(y_test_original.flatten(), pred_test, 'results/MNIST_confusion_matrix_original.png',
                      'results/MNIST_confusion_matrix_original.csv')
save_confusion_matrix(y_test_mixed.flatten(), pred_mixed, 'results/MNIST_confusion_matrix_mixed.png',
                      'resultsMNIST_confusion_matrix_mixed.csv')
save_confusion_matrix(y_test_R_form.flatten(), pred_R_form, 'resultsMNIST_confusion_matrix_R_form.png',
                      'resultsMNIST_confusion_matrix_R_form.csv')
save_confusion_matrix(y_test_S_form.flatten(), pred_S_form, 'resultsMNIST_confusion_matrix_S_form.png',
                      'results/MNIST_confusion_matrix_S_form.csv')

import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os

# Load and preprocess Fashion MNIST dataset
(x_train, y_train), (x_test_original, y_test_original) = fashion_mnist.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test_original = x_test_original / 255.0

# Reshape images to add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test_original = x_test_original.reshape(-1, 28, 28, 1)

# Device parameters
R_form_LH = 0.962
R_form_RH = 0.0383
S_form_LH = 0.0492
S_form_RH = 0.951
intensity_LH = 0.5
intensity_RH = 1 - intensity_LH

# Function to create mixed dataset based on device parameters
def create_device_mixed_data(x_data, y_data, LH_param, RH_param, label_type='label1'):
    n = x_data.shape[0]
    mixed_images = []
    mixed_labels = []
    for i in range(n - 1):
        img1, label1 = x_data[i], y_data[i]
        img2, label2 = x_data[i + 1], y_data[i + 1]
        mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH
        mixed_img = mixed_img / np.max(mixed_img)  # Normalize to [0,1]
        mixed_images.append(mixed_img)
        if label_type == 'label1':
            mixed_labels.append(label1)
        else:
            mixed_labels.append(label2)
    # Handle the last image
    img1, label1 = x_data[-1], y_data[-1]
    img2, label2 = x_data[0], y_data[0]
    mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH
    mixed_img = mixed_img / np.max(mixed_img)
    mixed_images.append(mixed_img)
    if label_type == 'label1':
        mixed_labels.append(label1)
    else:
        mixed_labels.append(label2)
    return np.array(mixed_images), np.array(mixed_labels)

# Create R-form and S-form datasets
x_test_R_form, y_test_R_form = create_device_mixed_data(
    x_test_original, y_test_original, R_form_LH, R_form_RH, label_type='label1')
x_test_S_form, y_test_S_form = create_device_mixed_data(
    x_test_original, y_test_original, S_form_LH, S_form_RH, label_type='label2')
# Create mixed dataset without device parameters (for comparison)
x_test_mixed, y_test_mixed = create_device_mixed_data(
    x_test_original, y_test_original, 1.0, 1.0, label_type='label1')

# Prepare labels for categorical crossentropy loss
y_train_cat = to_categorical(y_train, 10)
y_test_original_cat = to_categorical(y_test_original, 10)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation settings
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

batch_size = 64
epochs = 20
train_gen = datagen.flow(x_train, y_train_cat, batch_size=batch_size)
test_gen = datagen.flow(x_test_original, y_test_original_cat, batch_size=batch_size)

# Prepare directory for results
os.makedirs('results', exist_ok=True)

# Initialize arrays to store accuracies
accuracy_data = np.empty((epochs, 5))

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(train_gen, epochs=1, validation_data=test_gen)

    # Evaluate on original test set
    pred_test = np.argmax(model.predict(x_test_original), axis=-1)
    test_acc = np.mean(pred_test == y_test_original.flatten())

    # Evaluate on mixed dataset
    pred_mixed = np.argmax(model.predict(x_test_mixed), axis=-1)
    test_acc_mixed = np.mean(pred_mixed == y_test_mixed.flatten())

    # Evaluate on R-form dataset
    pred_R_form = np.argmax(model.predict(x_test_R_form), axis=-1)
    test_acc_R_form = np.mean(pred_R_form == y_test_R_form.flatten())

    # Evaluate on S-form dataset
    pred_S_form = np.argmax(model.predict(x_test_S_form), axis=-1)
    test_acc_S_form = np.mean(pred_S_form == y_test_S_form.flatten())

    # Store accuracies
    accuracy_data[epoch] = [epoch + 1, test_acc, test_acc_mixed, test_acc_R_form, test_acc_S_form]

    # Print accuracies
    print(f"Test Accuracy (Original): {test_acc:.4f}")
    print(f"Test Accuracy (Mixed): {test_acc_mixed:.4f}")
    print(f"Test Accuracy (R-form): {test_acc_R_form:.4f}")
    print(f"Test Accuracy (S-form): {test_acc_S_form:.4f}")

# Save accuracy data
np.savetxt('results/Fasion_MNIST_accuracy_data.csv', accuracy_data, fmt='%1.4f', delimiter=',')

# Function to save confusion matrix as heatmap and numerical data
def save_confusion_matrix(y_true, y_pred, filename_image, filename_data):
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix as image
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, square=True, annot=False, fmt='d', cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename_image)
    plt.close()  # Explicitly close the figure to free up resources

    # Save confusion matrix as numerical data (CSV)
    np.savetxt(filename_data, cm, fmt='%d', delimiter=',')


# Save confusion matrices using final epoch predictions
save_confusion_matrix(y_test_original.flatten(), pred_test, 'results/Fasion_MNIST_confusion_matrix_original.png',
                      'results/Fasion_MNIST_confusion_matrix_original.csv')
save_confusion_matrix(y_test_mixed.flatten(), pred_mixed, 'results/Fasion_MNIST_confusion_matrix_mixed.png',
                      'results/Fasion_MNIST_confusion_matrix_mixed.csv')
save_confusion_matrix(y_test_R_form.flatten(), pred_R_form, 'results/Fasion_MNIST_confusion_matrix_R_form.png',
                      'results/Fasion_MNIST_confusion_matrix_R_form.csv')
save_confusion_matrix(y_test_S_form.flatten(), pred_S_form, 'results/Fasion_MNIST_confusion_matrix_S_form.png',
                      'results/Fasion_MNIST_confusion_matrix_S_form.csv')