function thresholded = musan_threshold(image,threshold)
% 功能：设定SUSAN算法的阈值
 
[a b]=size(image);
intensity_center = image((a+1)/2,(b+1)/2);
 
% temp1 = (image-intensity_center)/threshold;
% temp2 = temp1.^6;
% thresholded = exp(-1*temp2);
r=threshold-abs(image-intensity_center);
r(r<0)=0;
r(r>0)=1;
thresholded=r;
% if abs(image-intensity_center)<threshold
%     thresholded=1;
% else
%     thresholded=0;
% end