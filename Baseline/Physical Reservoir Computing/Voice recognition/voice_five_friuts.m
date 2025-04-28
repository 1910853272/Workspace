<<<<<<< HEAD
clc
clear
close all;

%% blueberry (蓝莓)
% 读取蓝莓的音频文件并获得采样频率
[blueberry, Fs_blue] = audioread('E:\Desktop\voice\zmz\blueberry.wav');
N = length(blueberry);  % 获取音频数据的长度
time = (0:N-1) / Fs_blue;  % 创建时间向量

% 绘制蓝莓音频的波形
subplot(5,3,1)
plot(time, blueberry)
xlabel('Time/s');  % 横轴为时间
ylabel('Amplitude');  % 纵轴为振幅
title('blueberry');  % 图标题为“blueberry”

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(blueberry .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_blue / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,2)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('frequency');  % 图标题为“frequency”

% 绘制频域信号的幅值
subplot(5,3,3)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = blueberry(dt,:);  % 按照选择的时间点提取数据
sz_t(:,1) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,1) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,1) = [1, 0, 0, 0, 0];  % 创建一个标签矩阵，标记为蓝莓（第1类）

%% lychee (荔枝)
% 读取荔枝的音频文件并获得采样频率
[lychee, Fs_lychee] = audioread('E:\Desktop\zmz\lychee.wav');
N = length(lychee);  % 获取音频数据的长度
time = (0:N-1) / Fs_lychee;  % 创建时间向量

% 绘制荔枝音频的波形
subplot(5,3,4)
plot(time, lychee)
xlabel('Time/s');
ylabel('Amplitude');
title('lychee');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(lychee .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_lychee / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,5)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,6)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = lychee(dt,:);  % 按照选择的时间点提取数据
sz_t(:,2) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,2) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,2) = [0, 1, 0, 0, 0];  % 创建一个标签矩阵，标记为荔枝（第2类）

%% mango (芒果)
% 读取芒果的音频文件并获得采样频率
[mango, Fs_mango] = audioread('E:\Desktop\zmz\mango.wav');
N = length(mango);  % 获取音频数据的长度
time = (0:N-1) / Fs_mango;  % 创建时间向量

% 绘制芒果音频的波形
subplot(5,3,7)
plot(time, mango)
xlabel('Time/s');
ylabel('Amplitude');
title('mango');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(mango .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_mango / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,8)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,9)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = mango(dt,:);  % 按照选择的时间点提取数据
sz_t(:,3) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,3) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,3) = [0, 0, 1, 0, 0];  % 创建一个标签矩阵，标记为芒果（第3类）

%% pomegranate (石榴)
% 读取石榴的音频文件并获得采样频率
[pomegranate, Fs_pomegranate] = audioread('E:\Desktop\zmz\pomegranate.wav');
N = length(pomegranate);  % 获取音频数据的长度
time = (0:N-1) / Fs_pomegranate;  % 创建时间向量

% 绘制石榴音频的波形
subplot(5,3,10)
plot(time, pomegranate)
xlabel('Time/s');
ylabel('Amplitude');
title('pomegranate');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(pomegranate .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_pomegranate / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,11)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,12)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = pomegranate(dt,:);  % 按照选择的时间点提取数据
sz_t(:,4) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,4) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,4) = [0, 0, 0, 1, 0];  % 创建一个标签矩阵，标记为石榴（第4类）

%% shadock (柚子)
% 读取柚子的音频文件并获得采样频率
[shadock, Fs_shadock] = audioread('E:\Desktop\zmz\shadock.wav');
shadock = shadock(:,1);  % 选择单声道通道
N = length(shadock);  % 获取音频数据的长度
time = (0:N-1) / Fs_shadock;  % 创建时间向量

% 绘制柚子音频的波形
subplot(5,3,13)
plot(time, shadock)
xlabel('Time/s');
ylabel('Amplitude');
title('shadock');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(shadock .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_shadock / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,14)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,15)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = shadock(dt,:);  % 按照选择的时间点提取数据
sz_t(:,5) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,5) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,5) = [0, 0, 0, 0, 1];  % 创建一个标签矩阵，标记为柚子（第5类）

%% 数据归一化并准备最终的数据集
% 对时间数据进行归一化处理
t_min = min(min(sz_t));
t_max = max(max(sz_t));
new_sz(1:250,1:5) = (sz_t - t_min) / (t_max - t_min) * 255;  % 归一化到0-255范围

% 对频率数据进行归一化处理
f_min = min(min(sz_f));
f_max = max(max(sz_f));
new_sz(251:500,1:5) = (sz_f - f_min) / (f_max - f_min) * 255;  % 归一化到0-255范围
new_sz = round(new_sz);  % 四舍五入处理

% 转换为uint8类型
sz_uint8 = uint8(new_sz);

% 加载已有的数据集并扩展
load('voice_sz.mat');  % 加载已存在的时间-频率数据
load('voice_bq.mat');  % 加载已存在的标签数据
len_sz = size(sz,2);  % 获取已有数据集的列数
len_bq = size(bq,2);  % 获取已有标签数据的列数

% 将新生成的数据添加到已有数据集中
sz(:, len_sz+1:len_sz+5) = new_sz;  % 扩展时间-频率数据
bq(:, len_bq+1:len_bq+5) = new_bq;  % 扩展标签数据
=======
clc
clear
close all;

%% blueberry (蓝莓)
% 读取蓝莓的音频文件并获得采样频率
[blueberry, Fs_blue] = audioread('E:\Desktop\voice\zmz\blueberry.wav');
N = length(blueberry);  % 获取音频数据的长度
time = (0:N-1) / Fs_blue;  % 创建时间向量

% 绘制蓝莓音频的波形
subplot(5,3,1)
plot(time, blueberry)
xlabel('Time/s');  % 横轴为时间
ylabel('Amplitude');  % 纵轴为振幅
title('blueberry');  % 图标题为“blueberry”

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(blueberry .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_blue / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,2)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('frequency');  % 图标题为“frequency”

% 绘制频域信号的幅值
subplot(5,3,3)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = blueberry(dt,:);  % 按照选择的时间点提取数据
sz_t(:,1) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,1) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,1) = [1, 0, 0, 0, 0];  % 创建一个标签矩阵，标记为蓝莓（第1类）

%% lychee (荔枝)
% 读取荔枝的音频文件并获得采样频率
[lychee, Fs_lychee] = audioread('E:\Desktop\zmz\lychee.wav');
N = length(lychee);  % 获取音频数据的长度
time = (0:N-1) / Fs_lychee;  % 创建时间向量

% 绘制荔枝音频的波形
subplot(5,3,4)
plot(time, lychee)
xlabel('Time/s');
ylabel('Amplitude');
title('lychee');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(lychee .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_lychee / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,5)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,6)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = lychee(dt,:);  % 按照选择的时间点提取数据
sz_t(:,2) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,2) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,2) = [0, 1, 0, 0, 0];  % 创建一个标签矩阵，标记为荔枝（第2类）

%% mango (芒果)
% 读取芒果的音频文件并获得采样频率
[mango, Fs_mango] = audioread('E:\Desktop\zmz\mango.wav');
N = length(mango);  % 获取音频数据的长度
time = (0:N-1) / Fs_mango;  % 创建时间向量

% 绘制芒果音频的波形
subplot(5,3,7)
plot(time, mango)
xlabel('Time/s');
ylabel('Amplitude');
title('mango');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(mango .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_mango / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,8)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,9)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = mango(dt,:);  % 按照选择的时间点提取数据
sz_t(:,3) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,3) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,3) = [0, 0, 1, 0, 0];  % 创建一个标签矩阵，标记为芒果（第3类）

%% pomegranate (石榴)
% 读取石榴的音频文件并获得采样频率
[pomegranate, Fs_pomegranate] = audioread('E:\Desktop\zmz\pomegranate.wav');
N = length(pomegranate);  % 获取音频数据的长度
time = (0:N-1) / Fs_pomegranate;  % 创建时间向量

% 绘制石榴音频的波形
subplot(5,3,10)
plot(time, pomegranate)
xlabel('Time/s');
ylabel('Amplitude');
title('pomegranate');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(pomegranate .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_pomegranate / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,11)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,12)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = pomegranate(dt,:);  % 按照选择的时间点提取数据
sz_t(:,4) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,4) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,4) = [0, 0, 0, 1, 0];  % 创建一个标签矩阵，标记为石榴（第4类）

%% shadock (柚子)
% 读取柚子的音频文件并获得采样频率
[shadock, Fs_shadock] = audioread('E:\Desktop\zmz\shadock.wav');
shadock = shadock(:,1);  % 选择单声道通道
N = length(shadock);  % 获取音频数据的长度
time = (0:N-1) / Fs_shadock;  % 创建时间向量

% 绘制柚子音频的波形
subplot(5,3,13)
plot(time, shadock)
xlabel('Time/s');
ylabel('Amplitude');
title('shadock');

% 创建汉宁窗并计算FFT
win = hanning(N);  % 创建一个汉宁窗
nfft = 30000;  % FFT的点数
Y1 = fft(shadock .* win, nfft);  % 对音频信号加窗后进行FFT变换
f = (0:nfft/2) * Fs_shadock / nfft;  % 计算频率轴

% 绘制频域信号（幅值谱）
subplot(5,3,14)
y = Y1(1:nfft/2+1);  % 取出正频率部分
plot(f, y);
title('wave');  % 图标题为“wave”

% 绘制频域信号的幅值
subplot(5,3,15)
y_abs = abs(Y1(1:nfft/2+1));  % 计算FFT结果的幅度
plot(f, y_abs);
xlabel('Frequency/Hz');  % 横轴为频率（Hz）
ylabel('Amplitude');  % 纵轴为幅度
title('frequency');  % 图标题为“frequency”

% 从频域中采样，选择特定频率和时间点的数据
df = 1:40:10000;  % 选择频率轴的采样点
dt = 1:100:25000;  % 选择时间轴的采样点
t = shadock(dt,:);  % 按照选择的时间点提取数据
sz_t(:,5) = t;  % 将提取的时间数据存储在sz_t中
fr = y(df,:);  % 按照选择的频率点提取频率数据
sz_f(:,5) = abs(fr);  % 存储频率数据的幅度
new_bq(1:5,5) = [0, 0, 0, 0, 1];  % 创建一个标签矩阵，标记为柚子（第5类）

%% 数据归一化并准备最终的数据集
% 对时间数据进行归一化处理
t_min = min(min(sz_t));
t_max = max(max(sz_t));
new_sz(1:250,1:5) = (sz_t - t_min) / (t_max - t_min) * 255;  % 归一化到0-255范围

% 对频率数据进行归一化处理
f_min = min(min(sz_f));
f_max = max(max(sz_f));
new_sz(251:500,1:5) = (sz_f - f_min) / (f_max - f_min) * 255;  % 归一化到0-255范围
new_sz = round(new_sz);  % 四舍五入处理

% 转换为uint8类型
sz_uint8 = uint8(new_sz);

% 加载已有的数据集并扩展
load('voice_sz.mat');  % 加载已存在的时间-频率数据
load('voice_bq.mat');  % 加载已存在的标签数据
len_sz = size(sz,2);  % 获取已有数据集的列数
len_bq = size(bq,2);  % 获取已有标签数据的列数

% 将新生成的数据添加到已有数据集中
sz(:, len_sz+1:len_sz+5) = new_sz;  % 扩展时间-频率数据
bq(:, len_bq+1:len_bq+5) = new_bq;  % 扩展标签数据
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
