<<<<<<< HEAD
%% erode boundary map
out_map = zeros(size(map)+2);
out_map(2:401,2:641) = map;
new_m = imerode(out_map,ones(10));
erode_map = new_m(2:401,2:641);
figure(101)
imagesc(erode_map);
colormap('gray');
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
set(gca,'LineWidth',1.5)

%% erode boundary map and color
erode_map2 = reshape(erode_map,[256000 1]);
logical_erode_idmap = (erode_map>0.5); 
logical_erode_idmap2 = (erode_map2>0.5);
erode_map_to_show_id = logical_erode_idmap2;
close(figure(111))
figure(111)
imshow(schemepic);
hold on
scatter(map_x_vector(erode_map_to_show_id),map_y_vector(erode_map_to_show_id),...
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

%% select the point of depth map that locate in the boundary
colorfx = 804.067200 / 2;
colorfy = 804.067200 / 2;
colorcx = 642.510803 / 2;
colorcy = 406.336792 / 2;

Rmat = [  0.999658 -0.026112 0.001343
 0.026113 0.999658 -0.001089
 -0.001314 0.001124 0.999999];
Tmat = [-9.958817 -0.388572 1.521843] / 1000;

pcnewx_temp = x1;
pcnewy_temp = y1;
pcnewz_temp = z1;
pcmat_temp = [pcnewx_temp,pcnewy_temp,pcnewz_temp];
prgb_temp = pcmat_temp * Rmat' + Tmat;
myx = prgb_temp(:,1) * colorfx ./ prgb_temp(:,3) + colorcx;
myy = prgb_temp(:,2) * colorfy ./ prgb_temp(:,3) + colorcy;
myz = prgb_temp(:,3) * factor;

myx_int_temp = floor(myx);
myy_int_temp = floor(myy);
C = [myx_int_temp,myy_int_temp];
rgb_x_temp = 1:640;
rgb_y_temp = 1:400;
[rgb_Xmat_temp,rgb_Ymat_temp] = meshgrid(rgb_x_temp,rgb_y_temp);
rgb_Xvector_temp = reshape(rgb_Xmat_temp,[256000 1]);
rgb_Yvector_temp = reshape(rgb_Ymat_temp,[256000 1]);
[lia_temp, locb_temp] = ismember(C,[rgb_Xvector_temp,rgb_Yvector_temp],'rows');
lia_temp = lia_temp & (x1~=0);

logical_erode_id_inmap = zeros(256000,1);
for ii= 1:length(C)
    if lia_temp(ii)==1
        if logical_erode_idmap2(locb_temp(ii))==1
            logical_erode_id_inmap(ii)=1;
        end
    end
end
=======
%% erode boundary map
out_map = zeros(size(map)+2);
out_map(2:401,2:641) = map;
new_m = imerode(out_map,ones(10));
erode_map = new_m(2:401,2:641);
figure(101)
imagesc(erode_map);
colormap('gray');
xlim([0,640])
ylim([0,400])
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gcf,'Units','pixel','Position',[449.8,301,640,400])
set(gcf,'Color','k')
set(gca,'Position',[0 0 1 1]);
set(gca,'LineWidth',1.5)

%% erode boundary map and color
erode_map2 = reshape(erode_map,[256000 1]);
logical_erode_idmap = (erode_map>0.5); 
logical_erode_idmap2 = (erode_map2>0.5);
erode_map_to_show_id = logical_erode_idmap2;
close(figure(111))
figure(111)
imshow(schemepic);
hold on
scatter(map_x_vector(erode_map_to_show_id),map_y_vector(erode_map_to_show_id),...
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

%% select the point of depth map that locate in the boundary
colorfx = 804.067200 / 2;
colorfy = 804.067200 / 2;
colorcx = 642.510803 / 2;
colorcy = 406.336792 / 2;

Rmat = [  0.999658 -0.026112 0.001343
 0.026113 0.999658 -0.001089
 -0.001314 0.001124 0.999999];
Tmat = [-9.958817 -0.388572 1.521843] / 1000;

pcnewx_temp = x1;
pcnewy_temp = y1;
pcnewz_temp = z1;
pcmat_temp = [pcnewx_temp,pcnewy_temp,pcnewz_temp];
prgb_temp = pcmat_temp * Rmat' + Tmat;
myx = prgb_temp(:,1) * colorfx ./ prgb_temp(:,3) + colorcx;
myy = prgb_temp(:,2) * colorfy ./ prgb_temp(:,3) + colorcy;
myz = prgb_temp(:,3) * factor;

myx_int_temp = floor(myx);
myy_int_temp = floor(myy);
C = [myx_int_temp,myy_int_temp];
rgb_x_temp = 1:640;
rgb_y_temp = 1:400;
[rgb_Xmat_temp,rgb_Ymat_temp] = meshgrid(rgb_x_temp,rgb_y_temp);
rgb_Xvector_temp = reshape(rgb_Xmat_temp,[256000 1]);
rgb_Yvector_temp = reshape(rgb_Ymat_temp,[256000 1]);
[lia_temp, locb_temp] = ismember(C,[rgb_Xvector_temp,rgb_Yvector_temp],'rows');
lia_temp = lia_temp & (x1~=0);

logical_erode_id_inmap = zeros(256000,1);
for ii= 1:length(C)
    if lia_temp(ii)==1
        if logical_erode_idmap2(locb_temp(ii))==1
            logical_erode_id_inmap(ii)=1;
        end
    end
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
logical_erode_id_inmap = logical(logical_erode_id_inmap);