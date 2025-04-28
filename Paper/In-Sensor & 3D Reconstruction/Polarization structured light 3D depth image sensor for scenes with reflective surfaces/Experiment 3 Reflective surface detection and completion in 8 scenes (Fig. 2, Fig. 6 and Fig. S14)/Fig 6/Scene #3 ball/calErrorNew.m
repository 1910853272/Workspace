<<<<<<< HEAD
% this is used to calculate the errormap, run after plyread.m
%%
sc3_cover_logic = (uz3<425) & (z3~=0) & (uy3<244) & (uy3>10) & (ux3<475) & (ux3>125) & ...
    ~((uy3>190) & (uy3<245) & (ux3>123) & (ux3<271));
sc3_cover_logic_id = find(sc3_cover_logic);
figure(25)
scatter(ux3(sc3_cover_logic),uy3(sc3_cover_logic),2,uz3(sc3_cover_logic),'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%%
tofit_x3 = x3(sc3_cover_logic);
tofit_y3 = y3(sc3_cover_logic);
tofit_z3 = z3(sc3_cover_logic);
% fit poly11
[fitobject_sc3,gof_sc3] = fit([tofit_x3,tofit_y3],tofit_z3,'poly22','Robust','Bisquare');
fitcoeffvalue_sc3 = coeffvalues(fitobject_sc3);

%% plot fit plane in world coordinate
% val(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
curvefit_b_world_sc3 = [0.390636871739047	0.00666461232404349	0.340815354227339	1.30625816159378	0.0210667498667593	1.46224492677349];
b_sc3 = curvefit_b_world_sc3;
x3fit = linspace(-0.4,0.4,400);
y3fit = linspace(-0.4,0.4,400);
[X3FIT,Y3FIT] = ndgrid(x3fit,y3fit);
Z3FIT = b_sc3(1) + b_sc3(2)*X3FIT + b_sc3(3)*Y3FIT + b_sc3(4)*X3FIT.^2 + ...
    b_sc3(5)*X3FIT.*Y3FIT + b_sc3(6)*Y3FIT.^2;
X3FIT_vector = reshape(X3FIT,[160000 1]);
Y3FIT_vector = reshape(Y3FIT,[160000 1]);
Z3FIT_vector = reshape(Z3FIT,[160000 1]);
%%%%%%%%%%%%% plot fitplane in Pir，fig26 is sc3 and fitplane %%%%%%%%%%%%%
figure(26)
pcshow(sc3.Location,[0.5 0.5 0.5])
hold on
pcshow([X3FIT_vector,Y3FIT_vector,Z3FIT_vector],'r')
hold off
view(0,0)

%%
pcmat_sc3 = [X3FIT_vector,Y3FIT_vector,Z3FIT_vector];
prgb_sc3 = pcmat_sc3 * Rmat' + Tmat;
fitplane_camrgb_x_sc3 = prgb_sc3(:,1) * colorfx ./ prgb_sc3(:,3) + colorcx;
fitplane_camrgb_y_sc3 = prgb_sc3(:,2) * colorfx ./ prgb_sc3(:,3) + colorcy;
fitplane_camrgb_z_sc3 = prgb_sc3(:,3) * factor;

X3 = fitplane_camrgb_x_sc3;
Y3 = fitplane_camrgb_y_sc3;
V3 = fitplane_camrgb_z_sc3;
map_x3fit = 1:640;
map_y3fit = 1:400;
[mapX3FIT,mapY3FIT] = meshgrid(map_x3fit,map_y3fit);
X3q = mapX3FIT;
Y3q = mapY3FIT;
F3 = scatteredInterpolant(X3,Y3,V3);
myfitz3 = F3(X3q,Y3q);

mykkx3 = mapX3FIT;
mykky3 = mapY3FIT;
mykkz3 = myfitz3;
mykkx3(~logical_idmap) = nan;
mykky3(~logical_idmap) = nan;
mykkz3(~logical_idmap) = nan;
mykkx3 = reshape(mykkx3,[256000 1]);
mykky3 = reshape(mykky3,[256000 1]);
mykkz3 = reshape(mykkz3,[256000 1]);

%% plot the region that is used to calculate error
to_cal_id = logical_erode_idmap2 & (~isnan(mykkz3));
figure(23)
imshow(schemepic);
hold on
scatter(filled_map_x_incolor(to_cal_id),filled_map_y_incolor(to_cal_id),3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','y','MarkerFaceAlpha',0.3);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Position',[0 0 1 1]);

%% calculate the error
myerror_method2 = abs(filled_map_pt_incolor(to_cal_id,3) - mykkz3(to_cal_id));
temp_to_cal_id = to_cal_id;
close(figure(24))
figure(24)
imshow(schemepic);
hold on
scatter(filled_map_x_incolor(temp_to_cal_id),filled_map_y_incolor(temp_to_cal_id),2,myerror_method2,'filled','s');
hold off
set(gca,'Color','w')
xlim([0,640])
ylim([0,400])
colorbar('Linewidth',1.5)
caxis([0 max(myerror_method2)])
set(get(gca,'colorbar'),'Ticks',0:1:4)
set(get(gca,'colorbar'),'TickLabels',['','','','',''])
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0.05 0.05 0.85 0.9])
set(gca,'TickDir','out')
set(gcf,'Color','w')
=======
% this is used to calculate the errormap, run after plyread.m
%%
sc3_cover_logic = (uz3<425) & (z3~=0) & (uy3<244) & (uy3>10) & (ux3<475) & (ux3>125) & ...
    ~((uy3>190) & (uy3<245) & (ux3>123) & (ux3<271));
sc3_cover_logic_id = find(sc3_cover_logic);
figure(25)
scatter(ux3(sc3_cover_logic),uy3(sc3_cover_logic),2,uz3(sc3_cover_logic),'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%%
tofit_x3 = x3(sc3_cover_logic);
tofit_y3 = y3(sc3_cover_logic);
tofit_z3 = z3(sc3_cover_logic);
% fit poly11
[fitobject_sc3,gof_sc3] = fit([tofit_x3,tofit_y3],tofit_z3,'poly22','Robust','Bisquare');
fitcoeffvalue_sc3 = coeffvalues(fitobject_sc3);

%% plot fit plane in world coordinate
% val(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
curvefit_b_world_sc3 = [0.390636871739047	0.00666461232404349	0.340815354227339	1.30625816159378	0.0210667498667593	1.46224492677349];
b_sc3 = curvefit_b_world_sc3;
x3fit = linspace(-0.4,0.4,400);
y3fit = linspace(-0.4,0.4,400);
[X3FIT,Y3FIT] = ndgrid(x3fit,y3fit);
Z3FIT = b_sc3(1) + b_sc3(2)*X3FIT + b_sc3(3)*Y3FIT + b_sc3(4)*X3FIT.^2 + ...
    b_sc3(5)*X3FIT.*Y3FIT + b_sc3(6)*Y3FIT.^2;
X3FIT_vector = reshape(X3FIT,[160000 1]);
Y3FIT_vector = reshape(Y3FIT,[160000 1]);
Z3FIT_vector = reshape(Z3FIT,[160000 1]);
%%%%%%%%%%%%% plot fitplane in Pir，fig26 is sc3 and fitplane %%%%%%%%%%%%%
figure(26)
pcshow(sc3.Location,[0.5 0.5 0.5])
hold on
pcshow([X3FIT_vector,Y3FIT_vector,Z3FIT_vector],'r')
hold off
view(0,0)

%%
pcmat_sc3 = [X3FIT_vector,Y3FIT_vector,Z3FIT_vector];
prgb_sc3 = pcmat_sc3 * Rmat' + Tmat;
fitplane_camrgb_x_sc3 = prgb_sc3(:,1) * colorfx ./ prgb_sc3(:,3) + colorcx;
fitplane_camrgb_y_sc3 = prgb_sc3(:,2) * colorfx ./ prgb_sc3(:,3) + colorcy;
fitplane_camrgb_z_sc3 = prgb_sc3(:,3) * factor;

X3 = fitplane_camrgb_x_sc3;
Y3 = fitplane_camrgb_y_sc3;
V3 = fitplane_camrgb_z_sc3;
map_x3fit = 1:640;
map_y3fit = 1:400;
[mapX3FIT,mapY3FIT] = meshgrid(map_x3fit,map_y3fit);
X3q = mapX3FIT;
Y3q = mapY3FIT;
F3 = scatteredInterpolant(X3,Y3,V3);
myfitz3 = F3(X3q,Y3q);

mykkx3 = mapX3FIT;
mykky3 = mapY3FIT;
mykkz3 = myfitz3;
mykkx3(~logical_idmap) = nan;
mykky3(~logical_idmap) = nan;
mykkz3(~logical_idmap) = nan;
mykkx3 = reshape(mykkx3,[256000 1]);
mykky3 = reshape(mykky3,[256000 1]);
mykkz3 = reshape(mykkz3,[256000 1]);

%% plot the region that is used to calculate error
to_cal_id = logical_erode_idmap2 & (~isnan(mykkz3));
figure(23)
imshow(schemepic);
hold on
scatter(filled_map_x_incolor(to_cal_id),filled_map_y_incolor(to_cal_id),3,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor','y','MarkerFaceAlpha',0.3);
hold off
set(gca,'Color','k')
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'TickDir','out')
set(gcf,'Color','w')
set(gca,'Position',[0 0 1 1]);

%% calculate the error
myerror_method2 = abs(filled_map_pt_incolor(to_cal_id,3) - mykkz3(to_cal_id));
temp_to_cal_id = to_cal_id;
close(figure(24))
figure(24)
imshow(schemepic);
hold on
scatter(filled_map_x_incolor(temp_to_cal_id),filled_map_y_incolor(temp_to_cal_id),2,myerror_method2,'filled','s');
hold off
set(gca,'Color','w')
xlim([0,640])
ylim([0,400])
colorbar('Linewidth',1.5)
caxis([0 max(myerror_method2)])
set(get(gca,'colorbar'),'Ticks',0:1:4)
set(get(gca,'colorbar'),'TickLabels',['','','','',''])
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gca,'Position',[0.05 0.05 0.85 0.9])
set(gca,'TickDir','out')
set(gcf,'Color','w')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'linewidth',1)