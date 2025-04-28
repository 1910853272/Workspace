<<<<<<< HEAD
% run after the plyread.m
%% highlight the extracted glass region
tofit_id = ~logic_id_cam; 

extract_pc_x = ux1;
extract_pc_y = uy1;
extract_pc_z = uz1;
extract_pc_x(tofit_id) = nan;
extract_pc_y(tofit_id) = nan;
extract_pc_z(tofit_id) = nan;

tofit_x = ux1(tofit_id);
tofit_y = uy1(tofit_id);
tofit_z = uz1(tofit_id);

close(figure(102))
figure(102)
scatter(extract_pc_x,extract_pc_y,2,extract_pc_z,'filled','s');
load('CustomColormap_gd.mat')
colormap(CustomColormap_gd);
caxis([600 1300])
hold on
scatter(tofit_x,tofit_y,2,[1 0 0],'filled','s');
hold off
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

%% plot glass point and fitplane, which is for the pdf file
temp_show_x = x1(logic_id_cam_index);
temp_show_y = y1(logic_id_cam_index);
temp_show_z = z1(logic_id_cam_index);

temp_show_logical_id = (b(1) + b(2)*temp_show_x + b(3)*temp_show_y - temp_show_z) <0;
makercolormap = ones(length(temp_show_z),1)*[0.5 0.5 0.5]; 
makercolormap(temp_show_logical_id,:,:) = ones(length(find(temp_show_logical_id==1)),1)...
    * [0 0.4470 0.7410];

surf_show_x = linspace(1.5*min(temp_show_x)-0.5*max(temp_show_x),1.5*max(temp_show_x)-0.5*min(temp_show_x),10);
surf_show_y = linspace(1.5*min(temp_show_y)-0.5*max(temp_show_y),1.5*max(temp_show_y)-0.5*min(temp_show_y),10);
[temp_surf_X,temp_surf_Y] = meshgrid(surf_show_x,surf_show_y);
temp_surf_Z = b(1) + b(2)*temp_surf_X + b(3)*temp_surf_Y;

close(figure(103))
figure(103)
scatter3(temp_show_x,temp_show_y,temp_show_z,4,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor',[0 0.4470 0.7410],'MarkerFaceAlpha',1);
hold on
surf(temp_surf_X,temp_surf_Y,temp_surf_Z,'FaceColor',[0.5 0.5 0.5],...
    'FaceAlpha',0.4,'LineStyle','none')
hold off
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
set(gcf,'Color',[1 1 1])
view(34,33);
%
xlim([-0.15 0.7]);
ylim([-0.48 0.1]);
set(gca,'LineWidth',1.5)
set(gca,'FontSize',16)
fig103_xlabel = get(gca,'XLabel');
fig103_ylabel = get(gca,'YLabel');
set(fig103_xlabel,'Rotation',-14);
set(fig103_ylabel,'Rotation',35)
%
set(fig103_xlabel,'Position',[0.3380   -0.5246    0.4213]);
set(fig103_ylabel,'Position',[0.7927   -0.1928    0.4063]);
=======
% run after the plyread.m
%% highlight the extracted glass region
tofit_id = ~logic_id_cam; 

extract_pc_x = ux1;
extract_pc_y = uy1;
extract_pc_z = uz1;
extract_pc_x(tofit_id) = nan;
extract_pc_y(tofit_id) = nan;
extract_pc_z(tofit_id) = nan;

tofit_x = ux1(tofit_id);
tofit_y = uy1(tofit_id);
tofit_z = uz1(tofit_id);

close(figure(102))
figure(102)
scatter(extract_pc_x,extract_pc_y,2,extract_pc_z,'filled','s');
load('CustomColormap_gd.mat')
colormap(CustomColormap_gd);
caxis([600 1300])
hold on
scatter(tofit_x,tofit_y,2,[1 0 0],'filled','s');
hold off
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

%% plot glass point and fitplane, which is for the pdf file
temp_show_x = x1(logic_id_cam_index);
temp_show_y = y1(logic_id_cam_index);
temp_show_z = z1(logic_id_cam_index);

temp_show_logical_id = (b(1) + b(2)*temp_show_x + b(3)*temp_show_y - temp_show_z) <0;
makercolormap = ones(length(temp_show_z),1)*[0.5 0.5 0.5]; 
makercolormap(temp_show_logical_id,:,:) = ones(length(find(temp_show_logical_id==1)),1)...
    * [0 0.4470 0.7410];

surf_show_x = linspace(1.5*min(temp_show_x)-0.5*max(temp_show_x),1.5*max(temp_show_x)-0.5*min(temp_show_x),10);
surf_show_y = linspace(1.5*min(temp_show_y)-0.5*max(temp_show_y),1.5*max(temp_show_y)-0.5*min(temp_show_y),10);
[temp_surf_X,temp_surf_Y] = meshgrid(surf_show_x,surf_show_y);
temp_surf_Z = b(1) + b(2)*temp_surf_X + b(3)*temp_surf_Y;

close(figure(103))
figure(103)
scatter3(temp_show_x,temp_show_y,temp_show_z,4,'filled','MarkerEdgeColor','none',...
    'MarkerFaceColor',[0 0.4470 0.7410],'MarkerFaceAlpha',1);
hold on
surf(temp_surf_X,temp_surf_Y,temp_surf_Z,'FaceColor',[0.5 0.5 0.5],...
    'FaceAlpha',0.4,'LineStyle','none')
hold off
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
set(gcf,'Color',[1 1 1])
view(34,33);
%
xlim([-0.15 0.7]);
ylim([-0.48 0.1]);
set(gca,'LineWidth',1.5)
set(gca,'FontSize',16)
fig103_xlabel = get(gca,'XLabel');
fig103_ylabel = get(gca,'YLabel');
set(fig103_xlabel,'Rotation',-14);
set(fig103_ylabel,'Rotation',35)
%
set(fig103_xlabel,'Position',[0.3380   -0.5246    0.4213]);
set(fig103_ylabel,'Position',[0.7927   -0.1928    0.4063]);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
