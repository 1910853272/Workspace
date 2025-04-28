<<<<<<< HEAD
clc
clear
close all
%%
sc1 = pcread('berxelPoint3D_1.ply');
sc2 = pcread('berxelPoint3D_3.ply');

%% scene
load('color\Color_1.mat');
close(figure(91))
figure(91)
imshow(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','w')

%% load and convert to depth coordinate
% sc1
xyzdata1 = sc1.Location;
x1 = xyzdata1(:,1);
y1 = xyzdata1(:,2);
z1 = xyzdata1(:,3);
% sc2
xyzdata2 = sc2.Location;
x2 = xyzdata2(:,1);
y2 = xyzdata2(:,2);
z2 = xyzdata2(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 839.977722 / 2;
fy = 839.977722 / 2;
cx = 632.730957 / 2;
cy = 398.668823 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz1 = z1*factor;
ux1 = x1*fx./z1 + cx;
uy1 = y1*fy./z1 + cy;
% sc2 convert to depth map coordinate
uz2 = z2*factor;
ux2 = x2*fx./z2 + cx;
uy2 = y2*fy./z2 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap(CustomColormap);
caxis([900 1900])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(4)
scatter(ux2,uy2,2,uz2,'filled','s');
colormap(CustomColormap);
caxis([900 1900])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
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
sc1 = pcread('berxelPoint3D_1.ply');
sc2 = pcread('berxelPoint3D_3.ply');

%% scene
load('color\Color_1.mat');
close(figure(91))
figure(91)
imshow(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','w')

%% load and convert to depth coordinate
% sc1
xyzdata1 = sc1.Location;
x1 = xyzdata1(:,1);
y1 = xyzdata1(:,2);
z1 = xyzdata1(:,3);
% sc2
xyzdata2 = sc2.Location;
x2 = xyzdata2(:,1);
y2 = xyzdata2(:,2);
z2 = xyzdata2(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 839.977722 / 2;
fy = 839.977722 / 2;
cx = 632.730957 / 2;
cy = 398.668823 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz1 = z1*factor;
ux1 = x1*fx./z1 + cx;
uy1 = y1*fy./z1 + cy;
% sc2 convert to depth map coordinate
uz2 = z2*factor;
ux2 = x2*fx./z2 + cx;
uy2 = y2*fy./z2 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap(CustomColormap);
caxis([900 1900])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(4)
scatter(ux2,uy2,2,uz2,'filled','s');
colormap(CustomColormap);
caxis([900 1900])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
