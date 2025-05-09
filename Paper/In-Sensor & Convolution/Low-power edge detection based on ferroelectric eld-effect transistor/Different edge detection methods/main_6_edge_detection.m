clc;
clear;

% 选择文件夹
folder_path = 'BSDS500';
% 获取文件夹中的所有图片文件
image_files = dir(fullfile(folder_path, '*.jpg'));



% 创建保存处理后图片的文件夹
output_folder = fullfile('results_6_edge_detection');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
% 初始化计数器
processed_count = 0;
% 遍历每个图片文件
for k = 1:length(image_files)
    % 读取图片
    image_path = fullfile(folder_path, image_files(k).name);
    I = imread(image_path);

    % 转换为灰度图像
    image_size = size(I);
    dimension = numel(image_size);
    if dimension == 3
        I = rgb2gray(I);
    end
    fprintf('The current image name is %s \n', image_files(k).name);
    % 处理并保存图片
    process_and_save_image(I, output_folder, image_files(k).name);

    % 更新并打印计数器
    processed_count = processed_count + 1;
    fprintf('%d images done\n', processed_count);
    

end

function process_and_save_image(I, output_folder, original_filename)
% 拉普拉斯-高斯边缘检测
E = edge(I, 'log');
E=double(E);
E(E==0)=-1;
E(E==1)=0;
E(E==-1)=1;
save_image(E, output_folder, original_filename, 'Laplacian_of_Gaussian');
save_data(E, output_folder, original_filename, 'Laplacian_of_Gaussian');
% Sobel边缘检测
E = edge(I, 'sobel');
E=double(E);
E(E==0)=-1;
E(E==1)=0;
E(E==-1)=1;
save_image(E, output_folder, original_filename, 'Sobel');
save_data(E, output_folder, original_filename, 'Sobel');
% Prewitt边缘检测
E = edge(I, 'Prewitt');
E=double(E);
E(E==0)=-1;
E(E==1)=0;
E(E==-1)=1;
save_image(E, output_folder, original_filename, 'Prewitt');
save_data(E, output_folder, original_filename, 'Prewitt');

% Roberts边缘检测
E = edge(I, 'Roberts');
E=double(E);
E(E==0)=-1;
E(E==1)=0;
E(E==-1)=1;
save_image(E, output_folder, original_filename, 'Roberts');
save_data(E, output_folder, original_filename, 'Roberts');


% Susan边缘检测
E = susan(I, 25);
save_image(E, output_folder, original_filename, 'Susan');
save_data(E, output_folder, original_filename, 'Susan');

% Musan边缘检测
[E, Musan_matchnum] = musan_proposed(I, 22);
save_image(E, output_folder, original_filename, 'Musan');
save_data(E, output_folder, original_filename, 'Musan');
fprintf('The matching times of Musan is %d\n', Musan_matchnum);
end

function save_image(image_data, output_folder, original_filename, method)
% 创建处理后图片的文件名
[~, name, ext] = fileparts(original_filename);
output_filename = sprintf('%s_%s%s', name, method, ext);
output_path = fullfile(output_folder, output_filename);

% 保存图片
imwrite(image_data, output_path);
end

function save_data(image_data, output_folder, original_filename, method)
% 创建处理后图片的文件名
[~, name] = fileparts(original_filename);
output_filename = sprintf('%s_%s.csv', name, method);
output_path = fullfile(output_folder, output_filename);

% 保存图片
writematrix(image_data, output_path);
end
