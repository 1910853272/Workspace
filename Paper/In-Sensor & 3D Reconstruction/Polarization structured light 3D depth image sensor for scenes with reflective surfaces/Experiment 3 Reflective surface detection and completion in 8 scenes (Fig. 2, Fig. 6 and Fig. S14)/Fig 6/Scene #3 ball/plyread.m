<<<<<<< HEAD
clc
clear
close all

%% load pointcloud
% sc1 is pol 0, sc2 is pol 90, sc3 is benchmark
sc1 = pcread('berxelPoint3D_1.ply');
sc2 = pcread('berxelPoint3D_2.ply');
sc3 = pcread('berxelPoint3D_3.ply');

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
% sc3
xyzdata3 = sc3.Location;
x3 = xyzdata3(:,1);
y3 = xyzdata3(:,2);
z3 = xyzdata3(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 840.270020 / 2;
fy = 840.270020 / 2;
cx = 634.594482 / 2;
cy = 396.714142 / 2;
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
load CustomColormap.mat;
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap(CustomColormap);
caxis([300 1200])
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
caxis([300 1200])
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
figure(5)
scatter(ux3,uy3,2,uz3,'filled','s');
colormap(CustomColormap);
caxis([300 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Position',[0 0 1 1]);

%% boundary map
load("0080.mat");
glassboundary = imresize(glassboundary,[400 640]);
glassboundary = double(glassboundary)/255;
map = zeros(400,640);
map(1:400,1:640) = glassboundary;
figure(10)
imagesc(map);
colormap('gray');
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
set(gca,'LineWidth',1.5)

%% boundary map
map_x1fit = 1:640;
map_y1fit = 1:400;
[mapX1FIT,mapY1FIT] = meshgrid(map_x1fit,map_y1fit);
map_x_vector = reshape(mapX1FIT,[256000 1]);
map_y_vector = reshape(mapY1FIT,[256000 1]); 
%%%%%%%%%%%%%%%%%%%%% remove the bottom-right corner %%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [523,239]; rmpoint2 = [506,255];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)<0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%% remove the upper-right corner %%%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [528,1]; rmpoint2 = [554,42];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)>0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%% remove the upper-left corner %%%%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [84,72]; rmpoint2 = [117,1];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)>0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
map2 = reshape(map,[256000 1]);
logical_idmap = (map>0.9);
logical_idmap2 = (map2>0.9);
logical_idmap = (logical_idmap);
logical_idmap2 = (logical_idmap2);

map_to_show_id = logical_idmap2;
load('color\Color_1.mat');
close(figure(11))
figure(11)
imshow(schemepic);
hold on
scatter(map_x_vector(map_to_show_id),map_y_vector(map_to_show_id),...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','y','MarkerFaceAlpha',0.3);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% erodemap
% run erodemap.m

%% extract glass region
substract_uz = uz1-uz2;
figure(8)
scatter(ux1,uy1,2,substract_uz,'filled');
colorbar
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')

tempuz = abs(substract_uz);
temp_ux1 = ux1;
temp_uy1 = uy1;
logic_id_cam = (tempuz<300) | (substract_uz>800) | (~logical_erode_id_inmap) | (uz1>400);
tempuz(logic_id_cam)=NaN;
temp_ux1(logic_id_cam)=NaN;
temp_uy1(logic_id_cam)=NaN;
logic_id_cam_index = find(~logic_id_cam);

figure(9)
scatter(temp_ux1,temp_uy1,2,tempuz,'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%% fit plane
tofit_id = logic_id_cam_index;
tofit_x = x1(tofit_id);
tofit_y = y1(tofit_id);
tofit_z = z1(tofit_id);

[fitobject,gof] = fit([tofit_x,tofit_y],tofit_z,'poly22','Robust','Bisquare');
fitcoeffvalue = coeffvalues(fitobject);

%% plot the fitplane in world coordinate
% val(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
curvefit_b_world = [0.390377671442032	0.00332116973753826	0.326693222797862	1.28694578403700	0.0293768356775776	1.44091393858718];
b = curvefit_b_world;
x1fit = linspace(-0.4,0.4,400); % big enough to cover the glass region
y1fit = linspace(-0.4,0.4,400);
[X1FIT,Y1FIT] = ndgrid(x1fit,y1fit);
Z1FIT = b(1) + b(2)*X1FIT + b(3)*Y1FIT + b(4)*X1FIT.^2 + b(5)*X1FIT.*Y1FIT + b(6)*Y1FIT.^2;
X1FIT_vector = reshape(X1FIT,[length(x1fit)*length(y1fit) 1]);
Y1FIT_vector = reshape(Y1FIT,[length(x1fit)*length(y1fit) 1]);
Z1FIT_vector = reshape(Z1FIT,[length(x1fit)*length(y1fit) 1]);

%%%%%%%% plot the fitplane in Pir，fig12 is sc1 and fitplane, fig13 is sc3 and fitplane %%%%%%%%
close(figure(12))
figure(12)
pcshow(sc1.Location,[0.5 0.5 0.5])
hold on
pcshow([X1FIT_vector,Y1FIT_vector,Z1FIT_vector],'r')
hold off

close(figure(13))
figure(13)
pcshow(sc3.Location,[0.5 0.5 0.5])
hold on
pcshow([X1FIT_vector,Y1FIT_vector,Z1FIT_vector],'r')
hold off

%% convert the fit plane from Pir to Prgb
colorfx = 803.456970 / 2;
colorfy = 803.456970 / 2;
colorcx = 673.527100 / 2;
colorcy = 412.723907 / 2;

Rmat = [ 0.999587 -0.028488 0.003786
 0.028483 0.999593 0.001262
 -0.003820 -0.001154 0.999992];
Tmat = [-9.611872 -0.480608 1.107520] / 1000;

pcmat = [X1FIT_vector,Y1FIT_vector,Z1FIT_vector];
prgb = pcmat * Rmat' + Tmat;
fitplane_camrgb_x = prgb(:,1) * colorfx ./ prgb(:,3) + colorcx;
fitplane_camrgb_y = prgb(:,2) * colorfy ./ prgb(:,3) + colorcy;
fitplane_camrgb_z = prgb(:,3) * factor;

X = fitplane_camrgb_x;
Y = fitplane_camrgb_y;
V = fitplane_camrgb_z;
Xq = mapX1FIT;
Yq = mapY1FIT;
F = scatteredInterpolant(X,Y,V);
myfitz = F(Xq,Yq);

mykkx = mapX1FIT;
mykky = mapY1FIT;
mykkz = myfitz;
mykkx(~logical_erode_idmap) = nan;
mykky(~logical_erode_idmap) = nan;
mykkz(~logical_erode_idmap) = nan;
mykkx = reshape(mykkx,[256000 1]);
mykky = reshape(mykky,[256000 1]);
mykkz = reshape(mykkz,[256000 1]);

%% complete in color channel and convert it back to depth channel
filled_map_pt_incolor = [mykkx,mykky,mykkz];
filled_map_x_incolor = mykkx;
filled_map_y_incolor = mykky;
filled_map_z_incolor = mykkz;

filled_map_prgb_z = filled_map_z_incolor / factor;
filled_map_prgb_x = (filled_map_x_incolor - colorcx) .* filled_map_prgb_z / colorfx;
filled_map_prgb_y = (filled_map_y_incolor - colorcy) .* filled_map_prgb_z / colorfy;

filled_map_prgb = [filled_map_prgb_x,filled_map_prgb_y,filled_map_prgb_z];
filled_map_pir = (filled_map_prgb - Tmat) / (Rmat');
filled_map_pir_x = filled_map_pir(:,1);
filled_map_pir_y = filled_map_pir(:,2);
filled_map_pir_z = filled_map_pir(:,3);

filled_map_ux = filled_map_pir(:,1) * fx ./ filled_map_pir(:,3) + cx;
filled_map_uy = filled_map_pir(:,2) * fy ./ filled_map_pir(:,3) + cy;
filled_map_uz = filled_map_pir(:,3) * factor;

filled_map_ptcloud = pointCloud([filled_map_ux,filled_map_uy,filled_map_uz]);

%%%%%%%%%%%%%%%%%%%%% completed Pir and original sc1 %%%%%%%%%%%%%%%%%%%%%
close(figure(14))
figure(14)
pcshow(sc1.Location,[0.5,0.5,0.5]);
hold on
pcshow(filled_map_pir,'r');
hold off
view(0,-90)
%%%%%%%%%%%%%%%%%%%%% completed Pir and original sc3 %%%%%%%%%%%%%%%%%%%%%
close(figure(15))
figure(15)
pcshow(sc3.Location,[0.5,0.5,0.5]);
hold on
pcshow(filled_map_pir,'r');
hold off
view(0,-90)
%%%%%%%%%%%%%% completed depth pointcloud and original usc3 %%%%%%%%%%%%%%
close(figure(16))
figure(16)
pcshow(filled_map_ptcloud.Location,'r');
hold on
pcshow([ux3,uy3,uz3],[0.5 0.5 0.5]);
hold off
view(0,-90)
%%%%%%%%%%%%%% completed depth pointcloud and original usc1 %%%%%%%%%%%%%%
close(figure(17))
figure(17)
pcshow(filled_map_ptcloud.Location,'r');
hold on
pcshow([ux1,uy1,uz1],[0.5 0.5 0.5]);
hold off
view(0,-90)

%% 将map高于边界的点去掉
map_outside_depthBoundary_id = (filled_map_uy<5) | (filled_map_ux > 619) | (filled_map_uy > 394);
boundaryremoved_filled_map_ux = filled_map_ux;
boundaryremoved_filled_map_uy = filled_map_uy;
boundaryremoved_filled_map_uz = filled_map_uz;
boundaryremoved_filled_map_ux(map_outside_depthBoundary_id) = nan;
boundaryremoved_filled_map_uy(map_outside_depthBoundary_id) = nan;
boundaryremoved_filled_map_uz(map_outside_depthBoundary_id) = nan;
close(figure(18))
figure(18)
hold on
scatter(ux1,uy1,...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor',[0.5,0.5,0.5],'MarkerFaceAlpha',1);
scatter(boundaryremoved_filled_map_ux,boundaryremoved_filled_map_uy,...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','r','MarkerFaceAlpha',1);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Ydir','reverse')
set(gca,'Position',[0 0 1 1]);

%% plot the completed depth map
close(figure(19))
figure(19)
hold on
scatter(ux1,uy1,2,uz1,'filled','s');
scatter(boundaryremoved_filled_map_ux,boundaryremoved_filled_map_uy,...
    2,boundaryremoved_filled_map_uz,'filled','s');
hold off
colormap(CustomColormap);
caxis([300 1200])
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Ydir','reverse')
=======
clc
clear
close all

%% load pointcloud
% sc1 is pol 0, sc2 is pol 90, sc3 is benchmark
sc1 = pcread('berxelPoint3D_1.ply');
sc2 = pcread('berxelPoint3D_2.ply');
sc3 = pcread('berxelPoint3D_3.ply');

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
% sc3
xyzdata3 = sc3.Location;
x3 = xyzdata3(:,1);
y3 = xyzdata3(:,2);
z3 = xyzdata3(:,3);

%%%%%%%%%% convert parameters %%%%%%%%%%%%%%
fx = 840.270020 / 2;
fy = 840.270020 / 2;
cx = 634.594482 / 2;
cy = 396.714142 / 2;
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
load CustomColormap.mat;
figure(2)
scatter(ux1,uy1,2,uz1,'filled','s');
colormap(CustomColormap);
caxis([300 1200])
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
caxis([300 1200])
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
figure(5)
scatter(ux3,uy3,2,uz3,'filled','s');
colormap(CustomColormap);
caxis([300 1200])
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Position',[0 0 1 1]);

%% boundary map
load("0080.mat");
glassboundary = imresize(glassboundary,[400 640]);
glassboundary = double(glassboundary)/255;
map = zeros(400,640);
map(1:400,1:640) = glassboundary;
figure(10)
imagesc(map);
colormap('gray');
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
set(gca,'LineWidth',1.5)

%% boundary map
map_x1fit = 1:640;
map_y1fit = 1:400;
[mapX1FIT,mapY1FIT] = meshgrid(map_x1fit,map_y1fit);
map_x_vector = reshape(mapX1FIT,[256000 1]);
map_y_vector = reshape(mapY1FIT,[256000 1]); 
%%%%%%%%%%%%%%%%%%%%% remove the bottom-right corner %%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [523,239]; rmpoint2 = [506,255];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)<0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%% remove the upper-right corner %%%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [528,1]; rmpoint2 = [554,42];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)>0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%% remove the upper-left corner %%%%%%%%%%%%%%%%%%%%%%%%
rmpoint1 = [84,72]; rmpoint2 = [117,1];
line1b = polyfit([rmpoint1(1),rmpoint2(1)],[rmpoint1(2),rmpoint2(2)],1);
map_remove_id = ((line1b(1)*mapX1FIT+line1b(2)-mapY1FIT)>0);
map(map_remove_id) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
map2 = reshape(map,[256000 1]);
logical_idmap = (map>0.9);
logical_idmap2 = (map2>0.9);
logical_idmap = (logical_idmap);
logical_idmap2 = (logical_idmap2);

map_to_show_id = logical_idmap2;
load('color\Color_1.mat');
close(figure(11))
figure(11)
imshow(schemepic);
hold on
scatter(map_x_vector(map_to_show_id),map_y_vector(map_to_show_id),...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','y','MarkerFaceAlpha',0.3);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);

%% erodemap
% run erodemap.m

%% extract glass region
substract_uz = uz1-uz2;
figure(8)
scatter(ux1,uy1,2,substract_uz,'filled');
colorbar
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')

tempuz = abs(substract_uz);
temp_ux1 = ux1;
temp_uy1 = uy1;
logic_id_cam = (tempuz<300) | (substract_uz>800) | (~logical_erode_id_inmap) | (uz1>400);
tempuz(logic_id_cam)=NaN;
temp_ux1(logic_id_cam)=NaN;
temp_uy1(logic_id_cam)=NaN;
logic_id_cam_index = find(~logic_id_cam);

figure(9)
scatter(temp_ux1,temp_uy1,2,tempuz,'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%% fit plane
tofit_id = logic_id_cam_index;
tofit_x = x1(tofit_id);
tofit_y = y1(tofit_id);
tofit_z = z1(tofit_id);

[fitobject,gof] = fit([tofit_x,tofit_y],tofit_z,'poly22','Robust','Bisquare');
fitcoeffvalue = coeffvalues(fitobject);

%% plot the fitplane in world coordinate
% val(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
curvefit_b_world = [0.390377671442032	0.00332116973753826	0.326693222797862	1.28694578403700	0.0293768356775776	1.44091393858718];
b = curvefit_b_world;
x1fit = linspace(-0.4,0.4,400); % big enough to cover the glass region
y1fit = linspace(-0.4,0.4,400);
[X1FIT,Y1FIT] = ndgrid(x1fit,y1fit);
Z1FIT = b(1) + b(2)*X1FIT + b(3)*Y1FIT + b(4)*X1FIT.^2 + b(5)*X1FIT.*Y1FIT + b(6)*Y1FIT.^2;
X1FIT_vector = reshape(X1FIT,[length(x1fit)*length(y1fit) 1]);
Y1FIT_vector = reshape(Y1FIT,[length(x1fit)*length(y1fit) 1]);
Z1FIT_vector = reshape(Z1FIT,[length(x1fit)*length(y1fit) 1]);

%%%%%%%% plot the fitplane in Pir，fig12 is sc1 and fitplane, fig13 is sc3 and fitplane %%%%%%%%
close(figure(12))
figure(12)
pcshow(sc1.Location,[0.5 0.5 0.5])
hold on
pcshow([X1FIT_vector,Y1FIT_vector,Z1FIT_vector],'r')
hold off

close(figure(13))
figure(13)
pcshow(sc3.Location,[0.5 0.5 0.5])
hold on
pcshow([X1FIT_vector,Y1FIT_vector,Z1FIT_vector],'r')
hold off

%% convert the fit plane from Pir to Prgb
colorfx = 803.456970 / 2;
colorfy = 803.456970 / 2;
colorcx = 673.527100 / 2;
colorcy = 412.723907 / 2;

Rmat = [ 0.999587 -0.028488 0.003786
 0.028483 0.999593 0.001262
 -0.003820 -0.001154 0.999992];
Tmat = [-9.611872 -0.480608 1.107520] / 1000;

pcmat = [X1FIT_vector,Y1FIT_vector,Z1FIT_vector];
prgb = pcmat * Rmat' + Tmat;
fitplane_camrgb_x = prgb(:,1) * colorfx ./ prgb(:,3) + colorcx;
fitplane_camrgb_y = prgb(:,2) * colorfy ./ prgb(:,3) + colorcy;
fitplane_camrgb_z = prgb(:,3) * factor;

X = fitplane_camrgb_x;
Y = fitplane_camrgb_y;
V = fitplane_camrgb_z;
Xq = mapX1FIT;
Yq = mapY1FIT;
F = scatteredInterpolant(X,Y,V);
myfitz = F(Xq,Yq);

mykkx = mapX1FIT;
mykky = mapY1FIT;
mykkz = myfitz;
mykkx(~logical_erode_idmap) = nan;
mykky(~logical_erode_idmap) = nan;
mykkz(~logical_erode_idmap) = nan;
mykkx = reshape(mykkx,[256000 1]);
mykky = reshape(mykky,[256000 1]);
mykkz = reshape(mykkz,[256000 1]);

%% complete in color channel and convert it back to depth channel
filled_map_pt_incolor = [mykkx,mykky,mykkz];
filled_map_x_incolor = mykkx;
filled_map_y_incolor = mykky;
filled_map_z_incolor = mykkz;

filled_map_prgb_z = filled_map_z_incolor / factor;
filled_map_prgb_x = (filled_map_x_incolor - colorcx) .* filled_map_prgb_z / colorfx;
filled_map_prgb_y = (filled_map_y_incolor - colorcy) .* filled_map_prgb_z / colorfy;

filled_map_prgb = [filled_map_prgb_x,filled_map_prgb_y,filled_map_prgb_z];
filled_map_pir = (filled_map_prgb - Tmat) / (Rmat');
filled_map_pir_x = filled_map_pir(:,1);
filled_map_pir_y = filled_map_pir(:,2);
filled_map_pir_z = filled_map_pir(:,3);

filled_map_ux = filled_map_pir(:,1) * fx ./ filled_map_pir(:,3) + cx;
filled_map_uy = filled_map_pir(:,2) * fy ./ filled_map_pir(:,3) + cy;
filled_map_uz = filled_map_pir(:,3) * factor;

filled_map_ptcloud = pointCloud([filled_map_ux,filled_map_uy,filled_map_uz]);

%%%%%%%%%%%%%%%%%%%%% completed Pir and original sc1 %%%%%%%%%%%%%%%%%%%%%
close(figure(14))
figure(14)
pcshow(sc1.Location,[0.5,0.5,0.5]);
hold on
pcshow(filled_map_pir,'r');
hold off
view(0,-90)
%%%%%%%%%%%%%%%%%%%%% completed Pir and original sc3 %%%%%%%%%%%%%%%%%%%%%
close(figure(15))
figure(15)
pcshow(sc3.Location,[0.5,0.5,0.5]);
hold on
pcshow(filled_map_pir,'r');
hold off
view(0,-90)
%%%%%%%%%%%%%% completed depth pointcloud and original usc3 %%%%%%%%%%%%%%
close(figure(16))
figure(16)
pcshow(filled_map_ptcloud.Location,'r');
hold on
pcshow([ux3,uy3,uz3],[0.5 0.5 0.5]);
hold off
view(0,-90)
%%%%%%%%%%%%%% completed depth pointcloud and original usc1 %%%%%%%%%%%%%%
close(figure(17))
figure(17)
pcshow(filled_map_ptcloud.Location,'r');
hold on
pcshow([ux1,uy1,uz1],[0.5 0.5 0.5]);
hold off
view(0,-90)

%% 将map高于边界的点去掉
map_outside_depthBoundary_id = (filled_map_uy<5) | (filled_map_ux > 619) | (filled_map_uy > 394);
boundaryremoved_filled_map_ux = filled_map_ux;
boundaryremoved_filled_map_uy = filled_map_uy;
boundaryremoved_filled_map_uz = filled_map_uz;
boundaryremoved_filled_map_ux(map_outside_depthBoundary_id) = nan;
boundaryremoved_filled_map_uy(map_outside_depthBoundary_id) = nan;
boundaryremoved_filled_map_uz(map_outside_depthBoundary_id) = nan;
close(figure(18))
figure(18)
hold on
scatter(ux1,uy1,...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor',[0.5,0.5,0.5],'MarkerFaceAlpha',1);
scatter(boundaryremoved_filled_map_ux,boundaryremoved_filled_map_uy,...
    3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','r','MarkerFaceAlpha',1);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Ydir','reverse')
set(gca,'Position',[0 0 1 1]);

%% plot the completed depth map
close(figure(19))
figure(19)
hold on
scatter(ux1,uy1,2,uz1,'filled','s');
scatter(boundaryremoved_filled_map_ux,boundaryremoved_filled_map_uy,...
    2,boundaryremoved_filled_map_uz,'filled','s');
hold off
colormap(CustomColormap);
caxis([300 1200])
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','k')
set(gca,'Ydir','reverse')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'Position',[0 0 1 1]);