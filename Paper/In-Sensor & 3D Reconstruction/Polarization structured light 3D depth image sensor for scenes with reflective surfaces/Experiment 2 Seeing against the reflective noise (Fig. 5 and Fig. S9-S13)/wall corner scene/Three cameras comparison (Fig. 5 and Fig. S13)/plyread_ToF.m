<<<<<<< HEAD
clc
clear
close all
%%
sc7 = pcread('..\captured data\tof 24cm\RGBDPoints_20230602141012.ply');
sc8 = pcread('..\captured data\tof 40cm\RGBDPoints_20230602143136.ply');
sc9 = pcread('..\captured data\tof 56cm\RGBDPoints_20230602143938.ply');

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

%% sc1 depth map
figure(22)
scatter(ux7,uy7,10,uz7,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(24)
scatter(ux8,uy8,10,uz8,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc3 depth map
figure(26)
scatter(ux9,uy9,10,uz9,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
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
sc7 = pcread('..\captured data\tof 24cm\RGBDPoints_20230602141012.ply');
sc8 = pcread('..\captured data\tof 40cm\RGBDPoints_20230602143136.ply');
sc9 = pcread('..\captured data\tof 56cm\RGBDPoints_20230602143938.ply');

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

%% sc1 depth map
figure(22)
scatter(ux7,uy7,10,uz7,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc2 depth map
figure(24)
scatter(ux8,uy8,10,uz8,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% sc3 depth map
figure(26)
scatter(ux9,uy9,10,uz9,'filled','s');
colormap('jet');
caxis([200 1000])
set(gca,'Xdir','reverse')
set(gca,'Color','k')
xlim([30,605])
ylim([30,457])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'Position',[0 0 1 1]);