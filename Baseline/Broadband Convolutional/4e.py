import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore') # 忽略警告信息

# 设置 Matplotlib 显示参数
plt.rcParams['font.family'] = 'DejaVu Sans' #  DejaVu Sans 包含多种字符
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

class ColoredLetterDataset:
    """彩色字母数据集：红绿紫三色组成的字母"""
    
    def __init__(self, num_samples_per_letter=30, noise_std=0.05):
        self.num_samples_per_letter = num_samples_per_letter
        self.noise_std = noise_std
        self.letters = ['h', 'u', 's', 't']
        
        # 字母的3x3空间模板
        self.letter_templates = {
            'h': np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]]),
            'u': np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]),
            's': np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
            't': np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
        }
        
    def generate_colored_letter(self, letter):
        """
        生成彩色字母图像：这里的思路就是NE的思路 
        - 字母主干：每个像素随机选择深红、深绿、深紫中的一种
        - 背景区域：每个像素随机选择浅红、浅绿、浅紫中的一种作为噪声
        Output shape: (H, W, C) -> (3, 3, 3)
        """
        template = self.letter_templates[letter]
        colored_img = np.zeros((3, 3, 3)) # (H, W, C)
        
        for i in range(3):
            for j in range(3):
                if template[i, j] == 1:  # 字母主干位置
                    color_choice = np.random.choice(3)  # 0=红, 1=绿, 2=紫
                    if color_choice == 0:  # 深红色
                        colored_img[i, j, 0] = 0.8 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 1] = 0.1 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 2] = 0.1 + np.random.normal(0, self.noise_std/2)
                    elif color_choice == 1:  # 深绿色
                        colored_img[i, j, 0] = 0.1 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 1] = 0.8 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 2] = 0.1 + np.random.normal(0, self.noise_std/2)
                    else:  # 深紫色
                        colored_img[i, j, 0] = 0.4 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 1] = 0.1 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 2] = 0.8 + np.random.normal(0, self.noise_std)
                else:  # 背景位置
                    noise_choice = np.random.choice(3)
                    if noise_choice == 0:  # 浅红色噪声
                        colored_img[i, j, 0] = 0.2 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 1] = 0.05 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 2] = 0.05 + np.random.normal(0, self.noise_std/2)
                    elif noise_choice == 1:  # 浅绿色噪声
                        colored_img[i, j, 0] = 0.05 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 1] = 0.2 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 2] = 0.05 + np.random.normal(0, self.noise_std/2)
                    else:  # 浅紫色噪声
                        colored_img[i, j, 0] = 0.1 + np.random.normal(0, self.noise_std)
                        colored_img[i, j, 1] = 0.05 + np.random.normal(0, self.noise_std/2)
                        colored_img[i, j, 2] = 0.2 + np.random.normal(0, self.noise_std)
        
        colored_img = np.clip(colored_img, 0, 1)
        return colored_img # (H, W, C)
    
    def generate_dataset(self):
        """生成完整数据集"""
        X_broadband_list = []  # 宽带：完整彩色图像 (C, H, W) 
        X_single_bands_list = []  # 单波段：分离的单色图像 (num_bands, C_single, H, W)
        y_labels_list = []
        
        for label, letter in enumerate(self.letters):
            for _ in range(self.num_samples_per_letter):
                colored_img_hwc = self.generate_colored_letter(letter) # (H, W, C)
                
                # 宽带数据: (C, H, W)
                X_broadband_list.append(colored_img_hwc.transpose(2, 0, 1)) 
                
                # 单波段数据: list of (C_single, H, W)
                red_img_hwc = colored_img_hwc[:, :, 0:1]    # (H, W, 1)
                green_img_hwc = colored_img_hwc[:, :, 1:2]  # (H, W, 1)
                purple_img_hwc = colored_img_hwc[:, :, 2:3] # (H, W, 1)
                
                current_single_bands = [
                    red_img_hwc.transpose(2, 0, 1),    # (1, H, W)
                    green_img_hwc.transpose(2, 0, 1),  # (1, H, W)
                    purple_img_hwc.transpose(2, 0, 1)   # (1, H, W)
                ]
                X_single_bands_list.append(current_single_bands)
                y_labels_list.append(label)
        

        return np.array(X_broadband_list), np.array(X_single_bands_list), np.array(y_labels_list)

class BroadbandCNN_PT(nn.Module):
    """宽带网络：处理完整彩色图像 """
    def __init__(self, input_channels=3, num_classes=4):
        super(BroadbandCNN_PT, self).__init__()
        # PyTorch Conv2d: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(8, 16) # After global avg pool, channels = 8
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):

        x = F.pad(x, (0, 1, 0, 1)) # (pad_left, pad_right, pad_top, pad_bottom)
        x = F.relu(self.conv1(x))  # Output: (N, 16, 3, 3)
        
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv2(x))  # Output: (N, 8, 3, 3)
        
        x = self.pool(x)           # Output: (N, 8, 1, 1)
        x = torch.flatten(x, 1)    # Output: (N, 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1) # Use softmax for probability output
        return x

    def compile_and_train(self, X_train_np, y_train_np, X_val_np, y_val_np, epochs=10, batch_size=4, learning_rate=0.01):
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(X_train_np).float()
        y_train = torch.from_numpy(y_train_np).long()
        X_val = torch.from_numpy(X_val_np).float()
        y_val = torch.from_numpy(y_val_np).long()

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss() # CrossEntropyLoss includes Softmax
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            self.train() # 训练模式
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs) # Forward pass
                loss = criterion(outputs, labels)
                loss.backward() # Backward pass
                optimizer.step() # Optimize
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct_train / total_train
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            self.eval() # 评估模式
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = correct_val / total_val
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_acc)
            
        
        return {'history': history} # 

    def predict_proba(self, X_test_np):
        self.eval()
        X_test = torch.from_numpy(X_test_np).float()
        with torch.no_grad():
            outputs = self(X_test)
        return outputs.cpu().numpy()


class SingleBandCNN_PT_Model(nn.Module):
    """单波段网络中单个模型的定义 (PyTorch)"""
    def __init__(self, input_channels=1, num_classes=4):
        super(SingleBandCNN_PT_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(8, 12)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(12, num_classes)

    def forward(self, x):
        # x shape: (N, C, H, W) -> (N, 1, 3, 3)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv1(x)) # Output: (N, 8, 3, 3)
        
        x = self.pool(x)          # Output: (N, 8, 1, 1)
        x = torch.flatten(x, 1)   # Output: (N, 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

class SingleBandCNN_PT:
    """单波段网络：分别处理红、绿、紫单色图像 (PyTorch)"""
    def __init__(self, input_channels=1, num_classes=4):
        self.num_classes = num_classes
        self.band_names = ['Red', 'Green', 'Purple']
        self.models = {}
        for band in self.band_names:
            self.models[band] = SingleBandCNN_PT_Model(input_channels=input_channels, num_classes=num_classes)

    def compile_and_train(self, X_single_bands_train_np, y_train_np, X_single_bands_val_np, y_val_np, epochs=10, batch_size=4, learning_rate=0.008):
        # X_single_bands_train_np shape: (N, num_bands, C_single, H, W) -> (N, 3, 1, 3, 3)
        histories_dict_list = []
        
        y_train = torch.from_numpy(y_train_np).long()
        y_val = torch.from_numpy(y_val_np).long()

        for i, band_name in enumerate(self.band_names):
            model = self.models[band_name]
            
            X_train_band_np = X_single_bands_train_np[:, i, :, :, :] # (N_train, 1, 3, 3)
            X_val_band_np = X_single_bands_val_np[:, i, :, :, :]     # (N_val, 1, 3, 3)

            X_train_band = torch.from_numpy(X_train_band_np).float()
            X_val_band = torch.from_numpy(X_val_band_np).float()

            train_dataset_band = TensorDataset(X_train_band, y_train)
            val_dataset_band = TensorDataset(X_val_band, y_val)
            train_loader_band = DataLoader(train_dataset_band, batch_size=batch_size, shuffle=True)
            val_loader_band = DataLoader(val_dataset_band, batch_size=batch_size, shuffle=False)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            current_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0
                for inputs, labels in train_loader_band:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                
                epoch_loss = running_loss / len(train_loader_band.dataset)
                epoch_acc = correct_train / total_train
                current_history['loss'].append(epoch_loss)
                current_history['accuracy'].append(epoch_acc)

                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for inputs, labels in val_loader_band:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / len(val_loader_band.dataset)
                epoch_val_acc = correct_val / total_val
                current_history['val_loss'].append(epoch_val_loss)
                current_history['val_accuracy'].append(epoch_val_acc)
            
            histories_dict_list.append({'history': current_history}) # 
        return histories_dict_list

    def predict_ensemble(self, X_single_bands_test_np):
        # X_single_bands_test_np shape: (N, num_bands, C_single, H, W)
        all_preds_proba = []
        
        for i, band_name in enumerate(self.band_names):
            model = self.models[band_name]
            model.eval()
            
            X_test_band_np = X_single_bands_test_np[:, i, :, :, :] # (N_test, 1, 3, 3)
            X_test_band = torch.from_numpy(X_test_band_np).float()
            
            with torch.no_grad():
                outputs = model(X_test_band)
            all_preds_proba.append(outputs.cpu().numpy())
            
        # Average probabilities from each model
        ensemble_pred_proba = np.mean(all_preds_proba, axis=0)
        return ensemble_pred_proba


def visualize_colored_dataset(X_broadband_chw, X_single_bands_n_b_cshw, y_labels, letters):
    """可视化彩色字母数据集：展示分离过程"""
    # X_broadband_chw: (N, C, H, W)
    # X_single_bands_n_b_cshw: (N, num_bands, C_single, H, W)
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle('Dataset: Broadband (Mixed Colors) → Single-band (Separated Colors)', fontsize=16)
    
    for i, letter in enumerate(letters):
        letter_indices = np.where(y_labels == i)[0][:2]
        
        for j, idx in enumerate(letter_indices):
            broadband_img_chw = X_broadband_chw[idx] # (C, H, W)
            broadband_img_hwc = broadband_img_chw.transpose(1, 2, 0) # Convert to (H, W, C) for imshow

            # 第1列：宽带图像（三色混合）
            ax = axes[i, j*4]
            ax.imshow(broadband_img_hwc) # imshow expects (H,W,C) or (H,W)
            ax.set_title(f"'{letter}' Mixed\n(R+G+P)")
            ax.axis('off')
            
            # 第2-4列：分离的单色图像
            colors_cmap = ['Reds', 'Greens', 'Purples']
            band_names_display = ['Red', 'Green', 'Purple']
            
            for k in range(3): # Iterate through R, G, P channels of the broadband image
                ax = axes[i, j*4 + k + 1]
                # Display k-th channel from the broadband image
                channel_data_hw = broadband_img_chw[k, :, :] # (H, W)
                ax.imshow(channel_data_hw, cmap=colors_cmap[k], vmin=0, vmax=1)
                ax.set_title(f"'{letter}' {band_names_display[k]}\nfrom Mixed")
                ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('1.png')
    plt.show()

def visualize_separation_concept(X_broadband_chw, X_single_bands_n_b_cshw, y_labels, letters):
    """可视化分离概念：一个完整图像如何分离为三个部分"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle('Concept: How Broadband Image Separates into Single-band Images', fontsize=16)
    
    for i, letter in enumerate(letters):
        letter_idx = np.where(y_labels == i)[0][0]
        broadband_img_chw = X_broadband_chw[letter_idx] # (C,H,W)
        broadband_img_hwc = broadband_img_chw.transpose(1, 2, 0) # (H,W,C)
        
        # 第1列：原始宽带图像（混合色）
        ax = axes[i, 0]
        ax.imshow(broadband_img_hwc)
        ax.set_title(f"'{letter}'\nComplete")
        ax.axis('off')
        
        # 第2-4列：分离出的三个单色通道
        colors_cmap = ['Reds', 'Greens', 'Purples'] 
        band_names_display = ['Red\nChannel', 'Green\nChannel', 'Purple\nChannel']
        
        for k in range(3):
            ax = axes[i, k + 1]
            channel_data_hw = broadband_img_chw[k, :, :] # (H,W)
            ax.imshow(channel_data_hw, cmap=colors_cmap[k], vmin=0, vmax=1)
            ax.set_title(f"'{letter}'\n{band_names_display[k]}")
            ax.axis('off')
            
        # 第5列：显示信息损失
        ax = axes[i, 4]
        ax.imshow(broadband_img_chw[0, :, :], cmap='Reds', vmin=0, vmax=1) # Red channel (H,W)
        ax.set_title(f"'{letter}'\nPartial Info\n(Red only)")
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('2.png')
    plt.show()

def plot_training_comparison(bb_history_keras_like, sb_histories_keras_like, epochs):
    """绘制训练对比结果"""
    # bb_history_keras_like is {'history': {'val_accuracy': [...], ...}}
    # sb_histories_keras_like is a list of such dicts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    bb_history_data = bb_history_keras_like['history']
    sb_histories_data = [h['history'] for h in sb_histories_keras_like]

    # 准确率对比
    bb_val_acc = bb_history_data['val_accuracy']
    ax1.plot(range(1, epochs+1), bb_val_acc, 'ro-', 
             label='Broadband (Complete colors)', linewidth=3, markersize=8)
    
    sb_val_acc_mean = []
    for epoch_idx in range(epochs):
        epoch_accs = [hist_data['val_accuracy'][epoch_idx] for hist_data in sb_histories_data]
        sb_val_acc_mean.append(np.mean(epoch_accs))
    
    ax1.plot(range(1, epochs+1), sb_val_acc_mean, 'bo-',
             label='Single-band (Partial colors)', linewidth=3, markersize=8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Recognition Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 损失对比
    bb_val_loss = bb_history_data['val_loss']
    ax2.plot(range(1, epochs+1), bb_val_loss, 'r-', 
             label='Broadband', linewidth=3)
    
    sb_val_loss_mean = []
    for epoch_idx in range(epochs):
        epoch_losses = [hist_data['val_loss'][epoch_idx] for hist_data in sb_histories_data]
        sb_val_loss_mean.append(np.mean(epoch_losses))
    
    ax2.plot(range(1, epochs+1), sb_val_loss_mean, 'b-',
             label='Single-band', linewidth=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 各单色性能
    band_names_plot = ['Red', 'Green', 'Purple']
    colors_plot = ['red', 'green', 'purple']
    
    for hist_data, band, color in zip(sb_histories_data, band_names_plot, colors_plot):
        ax3.plot(range(1, epochs+1), hist_data['val_accuracy'], 
                 f'-', color=color, label=f'{band} only', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Individual Color Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 最终对比
    final_scores = [max(bb_val_acc), max(sb_val_acc_mean)] # Based on best val accuracy during training
    ax4.bar(['Complete Colors\n(Broadband)', 'Partial Colors\n(Single-band)'], 
            final_scores, color=['red', 'blue'], alpha=0.7)
    ax4.set_ylabel('Best Validation Accuracy')
    ax4.set_title('Final Performance (Best Validation)')
    ax4.set_ylim([0, 1.05])
    
    for i, v_score in enumerate(final_scores):
        ax4.text(i, v_score + 0.02, f'{v_score:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3.png')
    plt.show()

def main():
    num_epochs = 10 # 定义训练轮数

    # 1. 生成彩色字母数据集
    dataset = ColoredLetterDataset(num_samples_per_letter=35, noise_std=0.05)
    # X_broadband: (N, C, H, W)
    # X_single_bands: (N, num_bands, C_single, H, W)
    X_broadband, X_single_bands, y_labels = dataset.generate_dataset()
    
    print(f"Generated dataset: {len(X_broadband)} samples")
    print(f"Broadband images (complete colors) shape for PyTorch: {X_broadband.shape}")
    print(f"Single-band images (separated colors) shape for PyTorch: {X_single_bands.shape}")
    
    # 2. 可视化数据集对比 (确保可视化函数能处理CHW格式)
    print("\nVisualizing dataset...")
    visualize_colored_dataset(X_broadband, X_single_bands, y_labels, dataset.letters)
    visualize_separation_concept(X_broadband, X_single_bands, y_labels, dataset.letters)
    
    # 3. 分割数据
    X_bb_train, X_bb_test, y_bb_train, y_bb_test = train_test_split(
        X_broadband, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    X_sb_train, X_sb_test, y_sb_train, y_sb_test = train_test_split(
        X_single_bands, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    # 4. 训练宽带网络（完整彩色信息）
    print("\nTraining broadband network (complete color information) with PyTorch...")
    bb_cnn_pt = BroadbandCNN_PT()
    bb_history_pt = bb_cnn_pt.compile_and_train(
        X_bb_train, y_bb_train, X_bb_test, y_bb_test, epochs=num_epochs)
    
    # 5. 训练单波段网络（分离彩色信息）
    print("Training single-band networks (separated color information) with PyTorch...")
    sb_cnn_pt = SingleBandCNN_PT()
    sb_histories_pt = sb_cnn_pt.compile_and_train(
        X_sb_train, y_sb_train, X_sb_test, y_sb_test, epochs=num_epochs)
    
    # 6. 评估性能
    bb_pred_proba = bb_cnn_pt.predict_proba(X_bb_test)
    bb_pred_classes = np.argmax(bb_pred_proba, axis=1)
    bb_accuracy = accuracy_score(y_bb_test, bb_pred_classes)
    
    sb_pred_proba = sb_cnn_pt.predict_ensemble(X_sb_test)
    sb_pred_classes = np.argmax(sb_pred_proba, axis=1)
    sb_accuracy = accuracy_score(y_sb_test, sb_pred_classes)
    
    print(f"\nResults (Test Set Accuracy):")
    print(f"Broadband (complete colors) PyTorch: {bb_accuracy:.1%}")
    print(f"Single-band (partial colors, ensemble) PyTorch: {sb_accuracy:.1%}")
    print(f"Advantage of complete information: {(bb_accuracy - sb_accuracy)*100:+.1f} percentage points")
    
    # 7. 可视化训练结果
    plot_training_comparison(bb_history_pt, sb_histories_pt, epochs=num_epochs)
    
    # 8. 收敛分析 (基于验证集)
    bb_val_acc_history = bb_history_pt['history']['val_accuracy']
    bb_90_epoch = next((i+1 for i, acc in enumerate(bb_val_acc_history) if acc >= 0.9), None)
    
    sb_avg_val_acc = []
    num_sb_epochs = len(sb_histories_pt[0]['history']['val_accuracy']) # Should be num_epochs
    for epoch_idx in range(num_sb_epochs):
        epoch_accs = [hist['history']['val_accuracy'][epoch_idx] for hist in sb_histories_pt]
        sb_avg_val_acc.append(np.mean(epoch_accs))
        
    sb_90_epoch = next((i+1 for i, acc in enumerate(sb_avg_val_acc) if acc >= 0.9), None)
    
    print(f"\nConvergence Analysis (based on Validation Accuracy reaching 90%):")
    if bb_90_epoch:
        print(f"Broadband model (PyTorch) reached 90% val_accuracy at epoch: {bb_90_epoch}")
    else:
        print(f"Broadband model (PyTorch) did not reach 90% val_accuracy. Best val_accuracy: {max(bb_val_acc_history):.1%}")
        
    if sb_90_epoch:
        print(f"Single-band ensemble (avg, PyTorch) reached 90% val_accuracy at epoch: {sb_90_epoch}")  
    else:
        print(f"Single-band ensemble (avg, PyTorch) did not reach 90% val_accuracy. Best avg val_accuracy: {max(sb_avg_val_acc):.1%}")

if __name__ == "__main__":
    main()
