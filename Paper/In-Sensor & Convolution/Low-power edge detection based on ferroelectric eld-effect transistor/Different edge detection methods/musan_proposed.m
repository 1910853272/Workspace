function [image_out,Musan_matchnum] = musan_proposed(im,threshold)
% 功能：实现运用SUNSAN算子进行边缘检测
% 输入：image_in-输入的待检测的图像
%       threshold-阈值
% 输出：image_out-检测边缘出的二值图像

% 将输入的图像矩阵转换成double型
d = length(size(im));
if d==3
    image=double(rgb2gray(im));
elseif d==2
    image=double(im);
end

% 建立SUSAN模板
mask = ([0 0 1 0 0 ;0 0 1 0 0 ;1 1 1 1 1;0 0 1 0 0;0 0 1 0 0]);

% image_out=ones(size(image));
% 定义USAN 区域
% nmax = 3*37/4;
nmax = 3*10/4;
[a b]=size(image);
new=zeros(a+5,b+5);
[c d]=size(new);
new(3:c-3,3:d-3)=image;
R=ones(c,d);
matchnum=zeros(size(image));
for i=6:c-6
    
    for j=6:d-6
        
        
        
        current_image = new(i-2:i+2,j-2:j+2);
        current_masked_image = mask.*current_image;
        
        %   调用susan_threshold函数进行阈值比较处理
        
        current_thresholded = musan_threshold(current_masked_image,threshold);
        
        input1=[current_thresholded(3,1),current_thresholded(3,2),current_thresholded(3,4),current_thresholded(3,5)];
        input2=[current_thresholded(1,3),current_thresholded(2,3),current_thresholded(4,3),current_thresholded(5,3)];
        input=[input1;input2];
        output=zeros(2,4);
        for ii=1:2
            O1=0;
            O2=0;
            O3=0;
            O4=0;
            if input(ii,1:2)==[0,0]
                O1=1;
            end
            
            if input(ii,3:4)==[0,0]
                O2=1;
            end
            if input(ii,:)==[0,1,1,1]
                O3=1;
            end
            if input(ii,:)==[1,1,1,0]
                O4=1;
            end
            
            output(ii,:)=[O1,O2,O3,O4];
            if O1|O2
                R(i,j)=0;
                break
            end
            
            output(ii,:)=[O1,O2,O3,O4];
            
            
        end
        matchnum(i,j)=sum(sum(output));
        if  (output(1,3)|output(1,4))&(output(2,3)|output(2,4))
            R(i,j)=0;
        end
        
    end
end

image_out=R(3:c-3,3:d-3);
Musan_matchnum=sum(sum(matchnum));


end



