<<<<<<< HEAD
clc
clear
close all
%%
sc4 = pcread('PSL\berxelPoint3D_1.ply');
sc5 = pcread('PSL\berxelPoint3D_2.ply');

%% scene
load('PSL\color\Color_1.mat');
close(figure(91))
figure(91)
imshow(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','w')

%% load and convert to depth coordinate
% sc4
xyzdata4 = sc4.Location;
x4 = xyzdata4(:,1);
y4 = xyzdata4(:,2);
z4 = xyzdata4(:,3);
% sc5
xyzdata5 = sc5.Location;
x5 = xyzdata5(:,1);
y5 = xyzdata5(:,2);
z5 = xyzdata5(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 840.270020 / 2;
fy = 840.270020 / 2;
cx = 634.594482 / 2;
cy = 396.714142 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz4 = z4*factor;
ux4 = x4*fx./z4 + cx;
uy4 = y4*fy./z4 + cy;
% sc2 convert to depth map coordinate
uz5 = z5*factor;
ux5 = x5*fx./z5 + cx;
uy5 = y5*fy./z5 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(12)
scatter(ux4,uy4,2,uz4,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(14)
scatter(ux5,uy5,2,uz5,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
=======
clc
clear
close all
%%
sc4 = pcread('PSL\berxelPoint3D_1.ply');
sc5 = pcread('PSL\berxelPoint3D_2.ply');

%% scene
load('PSL\color\Color_1.mat');
close(figure(91))
figure(91)
imshow(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','w')

%% load and convert to depth coordinate
% sc4
xyzdata4 = sc4.Location;
x4 = xyzdata4(:,1);
y4 = xyzdata4(:,2);
z4 = xyzdata4(:,3);
% sc5
xyzdata5 = sc5.Location;
x5 = xyzdata5(:,1);
y5 = xyzdata5(:,2);
z5 = xyzdata5(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 840.270020 / 2;
fy = 840.270020 / 2;
cx = 634.594482 / 2;
cy = 396.714142 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz4 = z4*factor;
ux4 = x4*fx./z4 + cx;
uy4 = y4*fy./z4 + cy;
% sc2 convert to depth map coordinate
uz5 = z5*factor;
ux5 = x5*fx./z5 + cx;
uy5 = y5*fy./z5 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(12)
scatter(ux4,uy4,2,uz4,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(14)
scatter(ux5,uy5,2,uz5,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
