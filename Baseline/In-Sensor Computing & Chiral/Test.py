import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])
transform_test = transforms.Compose([transforms.ToTensor()])

# 下载与加载MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False,num_workers=0)

x_test_original, y_test_original = next(iter(test_loader))
# x_test_original: [10000, 1, 28, 28], y_test_original: [10000]

x_test_original = x_test_original.numpy()
y_test_original = y_test_original.numpy()

def create_device_mixed_data(x_data, y_data, LH_param, RH_param, shift=7, label_type='label1'):
    n = x_data.shape[0]
    mixed_images = []
    mixed_labels = []
    intensity_LH = 0.5
    intensity_RH = 1 - intensity_LH

    for i in range(n - 1):
        img1, label1 = x_data[i], y_data[i]
        #img2, label2 = x_data[i + 1], y_data[i + 1]
        img2, label2 = img1.copy(), label1

        # 保证形状是 [1, 28, 28]
        if img1.shape != (1, 28, 28):
            img1 = img1.reshape(1, 28, 28)
        if img2.shape != (1, 28, 28):
            img2 = img2.reshape(1, 28, 28)

        shifted_img2 = np.zeros_like(img2)
        shifted_img2[:, :, shift:] = img2[:, :, :-shift]

        mixed_img = img1 * LH_param * intensity_LH + shifted_img2 * RH_param * intensity_RH
        mixed_img = mixed_img / (np.max(mixed_img) + 1e-7)
        mixed_images.append(mixed_img.astype(np.float32))
        mixed_labels.append(label1 if label_type == 'label1' else label2)

    img1, label1 = x_data[-1], y_data[-1]
    img2, label2 = x_data[0], y_data[0]
    if img1.shape != (1, 28, 28):
        img1 = img1.reshape(1, 28, 28)
    if img2.shape != (1, 28, 28):
        img2 = img2.reshape(1, 28, 28)
    shifted_img2 = np.zeros_like(img2)
    #shifted_img2[:, :, shift:] = img2[:, :, :-shift]
    # 左移 shift 个像素
    shifted_img2[:, :, :-shift] = img2[:, :, shift:]

    mixed_img = img1 * LH_param * intensity_LH + shifted_img2 * RH_param * intensity_RH
    mixed_img = mixed_img / (np.max(mixed_img) + 1e-7)
    mixed_images.append(mixed_img.astype(np.float32))
    mixed_labels.append(label1 if label_type == 'label1' else label2)

    mixed_images = np.stack(mixed_images)
    mixed_labels = np.array(mixed_labels)
    return mixed_images, mixed_labels



R_form_LH = 0.7
R_form_RH = 0.3
S_form_LH = 0.4
S_form_RH = 0.6

x_test_mixed, y_test_mixed = create_device_mixed_data(x_test_original, y_test_original, 1.0, 1.0, label_type='label1')
x_test_R_form, y_test_R_form = create_device_mixed_data(x_test_original, y_test_original, R_form_LH, R_form_RH, label_type='label1')
x_test_S_form, y_test_S_form = create_device_mixed_data(x_test_original, y_test_original, S_form_LH, S_form_RH, label_type='label2')

idx = np.random.randint(0, x_test_original.shape[0])

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
plt.suptitle(f'Index {idx}: Original/Mixed/R-form/S-form')

axs[0].imshow(x_test_original[idx][0], cmap='gray')
axs[0].set_title(f'Original\nlabel={y_test_original[idx]}')
axs[0].axis('off')

axs[1].imshow(x_test_mixed[idx][0], cmap='gray')
axs[1].set_title(f'Mixed\nlabel={y_test_mixed[idx]}')
axs[1].axis('off')

axs[2].imshow(x_test_R_form[idx][0], cmap='gray')
axs[2].set_title(f'R-form\nlabel={y_test_R_form[idx]}')
axs[2].axis('off')

axs[3].imshow(x_test_S_form[idx][0], cmap='gray')
axs[3].set_title(f'S-form\nlabel={y_test_S_form[idx]}')
axs[3].axis('off')

plt.tight_layout()
plt.show()
