<<<<<<< HEAD
clc
clear
close all
%% plot scene
load('scenepic.mat');
close(figure(91))
figure(91)
imshow(scenepic);
set(gcf,'Units','pixel','Position',[449.8,201,585,518])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
%% plot PSL
load('PSLpic.mat');
close(figure(92))
figure(92)
imshow(PSLpic);
set(gcf,'Units','pixel','Position',[449.8,201,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
%% plot stereo vision
load('stereopic.mat');
close(figure(93))
figure(93)
imshow(stereopic);
set(gcf,'Units','pixel','Position',[449.8,201,640,400])
set(gca,'Position',[0 0 1 1]);
=======
clc
clear
close all
%% plot scene
load('scenepic.mat');
close(figure(91))
figure(91)
imshow(scenepic);
set(gcf,'Units','pixel','Position',[449.8,201,585,518])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
%% plot PSL
load('PSLpic.mat');
close(figure(92))
figure(92)
imshow(PSLpic);
set(gcf,'Units','pixel','Position',[449.8,201,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
%% plot stereo vision
load('stereopic.mat');
close(figure(93))
figure(93)
imshow(stereopic);
set(gcf,'Units','pixel','Position',[449.8,201,640,400])
set(gca,'Position',[0 0 1 1]);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gcf,'Color','k')