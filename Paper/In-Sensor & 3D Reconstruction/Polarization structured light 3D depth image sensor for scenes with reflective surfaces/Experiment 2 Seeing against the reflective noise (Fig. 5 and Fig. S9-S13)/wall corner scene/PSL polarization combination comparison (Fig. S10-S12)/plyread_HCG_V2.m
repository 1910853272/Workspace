<<<<<<< HEAD
clc
clear
close all

%% Select the height that needs to plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 24cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc1 = pcread('..\captured data\PSL TX pol x 24cm\RX no pol\berxelPoint3D_1.ply');
sc2 = pcread('..\captured data\PSL TX pol x 24cm\RX pol y\berxelPoint3D_1.ply');
sc3 = pcread('..\captured data\PSL TX pol x 24cm\RX pol x\berxelPoint3D_1.ply');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 40cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sc1 = pcread('..\captured data\PSL TX pol x 40cm\RX no pol\berxelPoint3D_1.ply');
% sc2 = pcread('..\captured data\PSL TX pol x 40cm\RX pol y\berxelPoint3D_1.ply');
% sc3 = pcread('..\captured data\PSL TX pol x 40cm\RX pol x\berxelPoint3D_1.ply');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 56cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sc1 = pcread('..\captured data\PSL TX pol x 56cm\RX no pol\berxelPoint3D_1.ply');
% sc2 = pcread('..\captured data\PSL TX pol x 56cm\RX pol y\berxelPoint3D_1.ply');
% sc3 = pcread('..\captured data\PSL TX pol x 56cm\RX pol x\berxelPoint3D_1.ply');

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
% sc3
xyzdata3 = sc3.Location;
x3 = xyzdata3(:,1);
y3 = xyzdata3(:,2);
z3 = xyzdata3(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 837.515564 / 2;
fy = 837.515564 / 2;
cx = 632.205383 / 2;
cy = 399.018280 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz1 = z1*factor;
ux1 = x1*fx./z1 + cx;
uy1 = y1*fy./z1 + cy;
% sc2 convert to depth map coordinate
uz2 = z2*factor;
ux2 = x2*fx./z2 + cx;
uy2 = y2*fy./z2 + cy;
% sc3 convert to depth map coordinate
uz3 = z3*factor;
ux3 = x3*fx./z3 + cx;
uy3 = y3*fy./z3 + cy;

%% sc1 depth map
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap('jet');
caxis([300 1000])
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
colormap('jet');
caxis([300 1000])
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

%% sc3 depth map
figure(6)
scatter(ux3,uy3,2,uz3,'filled','s');
colormap('jet');
caxis([300 1000])
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

%% Select the height that needs to plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 24cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc1 = pcread('..\captured data\PSL TX pol x 24cm\RX no pol\berxelPoint3D_1.ply');
sc2 = pcread('..\captured data\PSL TX pol x 24cm\RX pol y\berxelPoint3D_1.ply');
sc3 = pcread('..\captured data\PSL TX pol x 24cm\RX pol x\berxelPoint3D_1.ply');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 40cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sc1 = pcread('..\captured data\PSL TX pol x 40cm\RX no pol\berxelPoint3D_1.ply');
% sc2 = pcread('..\captured data\PSL TX pol x 40cm\RX pol y\berxelPoint3D_1.ply');
% sc3 = pcread('..\captured data\PSL TX pol x 40cm\RX pol x\berxelPoint3D_1.ply');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 56cm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sc1 = pcread('..\captured data\PSL TX pol x 56cm\RX no pol\berxelPoint3D_1.ply');
% sc2 = pcread('..\captured data\PSL TX pol x 56cm\RX pol y\berxelPoint3D_1.ply');
% sc3 = pcread('..\captured data\PSL TX pol x 56cm\RX pol x\berxelPoint3D_1.ply');

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
% sc3
xyzdata3 = sc3.Location;
x3 = xyzdata3(:,1);
y3 = xyzdata3(:,2);
z3 = xyzdata3(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 837.515564 / 2;
fy = 837.515564 / 2;
cx = 632.205383 / 2;
cy = 399.018280 / 2;
factor = 1000;
% sc1 convert to depth map coordinate
uz1 = z1*factor;
ux1 = x1*fx./z1 + cx;
uy1 = y1*fy./z1 + cy;
% sc2 convert to depth map coordinate
uz2 = z2*factor;
ux2 = x2*fx./z2 + cx;
uy2 = y2*fy./z2 + cy;
% sc3 convert to depth map coordinate
uz3 = z3*factor;
ux3 = x3*fx./z3 + cx;
uy3 = y3*fy./z3 + cy;

%% sc1 depth map
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap('jet');
caxis([300 1000])
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
colormap('jet');
caxis([300 1000])
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

%% sc3 depth map
figure(6)
scatter(ux3,uy3,2,uz3,'filled','s');
colormap('jet');
caxis([300 1000])
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
