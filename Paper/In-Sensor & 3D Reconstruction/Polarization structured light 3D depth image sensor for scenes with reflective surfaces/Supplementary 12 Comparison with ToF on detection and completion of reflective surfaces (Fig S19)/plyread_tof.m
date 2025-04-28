<<<<<<< HEAD
clc
clear
close all
%%
sc7 = pcread('ToF\no pol\RGBDPoints_20230602153549.ply');
sc8 = pcread('ToF\Tx pol 0\RGBDPoints_20230602154330.ply');
sc9 = pcread('ToF\Tx 0 + Rx 0\RGBDPoints_20230602154918.ply');
sc10 = pcread('ToF\Tx 0 + Rx 90\RGBDPoints_20230602155250.ply');

%% scene
load('ToF\no pol\ToFColor.mat');
close(figure(91))
figure(91)
image(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
set(gca,'XTick',[])
set(gca,'YTick',[])

%% load and convert to depth coordinate
% sc7
xyzdata7 = sc7.Location;
x7 = xyzdata7(:,1);
y7 = xyzdata7(:,2);
z7 = xyzdata7(:,3);
% sc8
xyzdata8 = sc8.Location;
x8 = xyzdata8(:,1);
y8 = xyzdata8(:,2);
z8 = xyzdata8(:,3);
% sc9
xyzdata9 = sc9.Location;
x9 = xyzdata9(:,1);
y9 = xyzdata9(:,2);
z9 = xyzdata9(:,3);
% sc10
xyzdata10 = sc10.Location;
x10 = xyzdata10(:,1);
y10 = xyzdata10(:,2);
z10 = xyzdata10(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 313.179;
fy = 313.179;
cx = 319.482;
cy = 241.799;
factor = 1000;
% sc1 convert to depth map coordinate
uz7 = z7*factor;
ux7 = x7*fx./z7 + cx;
uy7 = y7*fy./z7 + cy;
% sc2 convert to depth map coordinate
uz8 = z8*factor;
ux8 = x8*fx./z8 + cx;
uy8 = y8*fy./z8 + cy;
% sc3 convert to depth map coordinate
uz9 = z9*factor;
ux9 = x9*fx./z9 + cx;
uy9 = y9*fy./z9 + cy;
% sc3 convert to depth map coordinate
uz10 = z10*factor;
ux10 = x10*fx./z10 + cx;
uy10 = y10*fy./z10 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(22)
scatter(ux7,uy7,10,uz7,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
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
figure(24)
scatter(ux8,uy8,10,uz8,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc3 depth map
figure(26)
scatter(ux9,uy9,10,uz9,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc4 depth map
figure(28)
scatter(ux10,uy10,10,uz10,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
=======
clc
clear
close all
%%
sc7 = pcread('ToF\no pol\RGBDPoints_20230602153549.ply');
sc8 = pcread('ToF\Tx pol 0\RGBDPoints_20230602154330.ply');
sc9 = pcread('ToF\Tx 0 + Rx 0\RGBDPoints_20230602154918.ply');
sc10 = pcread('ToF\Tx 0 + Rx 90\RGBDPoints_20230602155250.ply');

%% scene
load('ToF\no pol\ToFColor.mat');
close(figure(91))
figure(91)
image(schemepic);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0 0 1 1]);
set(gcf,'Color','k')
set(gca,'XTick',[])
set(gca,'YTick',[])

%% load and convert to depth coordinate
% sc7
xyzdata7 = sc7.Location;
x7 = xyzdata7(:,1);
y7 = xyzdata7(:,2);
z7 = xyzdata7(:,3);
% sc8
xyzdata8 = sc8.Location;
x8 = xyzdata8(:,1);
y8 = xyzdata8(:,2);
z8 = xyzdata8(:,3);
% sc9
xyzdata9 = sc9.Location;
x9 = xyzdata9(:,1);
y9 = xyzdata9(:,2);
z9 = xyzdata9(:,3);
% sc10
xyzdata10 = sc10.Location;
x10 = xyzdata10(:,1);
y10 = xyzdata10(:,2);
z10 = xyzdata10(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 313.179;
fy = 313.179;
cx = 319.482;
cy = 241.799;
factor = 1000;
% sc1 convert to depth map coordinate
uz7 = z7*factor;
ux7 = x7*fx./z7 + cx;
uy7 = y7*fy./z7 + cy;
% sc2 convert to depth map coordinate
uz8 = z8*factor;
ux8 = x8*fx./z8 + cx;
uy8 = y8*fy./z8 + cy;
% sc3 convert to depth map coordinate
uz9 = z9*factor;
ux9 = x9*fx./z9 + cx;
uy9 = y9*fy./z9 + cy;
% sc3 convert to depth map coordinate
uz10 = z10*factor;
ux10 = x10*fx./z10 + cx;
uy10 = y10*fy./z10 + cy;

%% sc1 depth map
load CustomColormap.mat;
figure(22)
scatter(ux7,uy7,10,uz7,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
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
figure(24)
scatter(ux8,uy8,10,uz8,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc3 depth map
figure(26)
scatter(ux9,uy9,10,uz9,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc4 depth map
figure(28)
scatter(ux10,uy10,10,uz10,'filled','s');
colormap(CustomColormap);
caxis([500 1200])
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'Position',[0 0 1 1]);