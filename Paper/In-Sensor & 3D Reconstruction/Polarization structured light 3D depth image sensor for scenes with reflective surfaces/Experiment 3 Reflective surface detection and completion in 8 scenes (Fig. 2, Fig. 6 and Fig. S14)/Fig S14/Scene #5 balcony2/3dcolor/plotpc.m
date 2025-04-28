<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_balconydoor2_beforeCompletion.ply');
sc8 = pcread('withglass_pir_balconydoor2_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0 max(sc8.Location(:,3))])
view(-0.7451171875,-25.396484375)
set(gca,'CameraPosition',[-0.241895390758655,-9.330243407823593,-8.242878380159416])
set(gca,'CameraTarget',[0.009728916763747,-0.101421138241251,1.317361914327423])
set(gca,'CameraViewAngle',5.527523426185996)
set(gca,'CameraUpVector',[-0.018108003265039,-0.725546967091094,0.687934370970566])
%% pointcloud after completion with glass area denoted by red color
length_red = 0;
for ii = 1:length(sc8.Color)
    if sc8.Color(ii,:,:) == uint8([255 0 0])
        length_red = length_red + 1;
    end
end
sc_length = length(sc8.Location);
close(figure(610))
figure(610)
pcshow(sc8.Location(1:sc_length-length_red,:,:))
hold on
pcshow(select(sc8,(sc_length-length_red+1):sc_length))
hold off
caxis([0 max(sc8.Location(:,3))])
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
view(-0.7451171875,-25.396484375)
set(gca,'CameraPosition',[-0.241895390758655,-9.330243407823593,-8.242878380159416])
set(gca,'CameraTarget',[0.009728916763747,-0.101421138241251,1.317361914327423])
set(gca,'CameraViewAngle',5.527523426185996)
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_balconydoor2_beforeCompletion.ply');
sc8 = pcread('withglass_pir_balconydoor2_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0 max(sc8.Location(:,3))])
view(-0.7451171875,-25.396484375)
set(gca,'CameraPosition',[-0.241895390758655,-9.330243407823593,-8.242878380159416])
set(gca,'CameraTarget',[0.009728916763747,-0.101421138241251,1.317361914327423])
set(gca,'CameraViewAngle',5.527523426185996)
set(gca,'CameraUpVector',[-0.018108003265039,-0.725546967091094,0.687934370970566])
%% pointcloud after completion with glass area denoted by red color
length_red = 0;
for ii = 1:length(sc8.Color)
    if sc8.Color(ii,:,:) == uint8([255 0 0])
        length_red = length_red + 1;
    end
end
sc_length = length(sc8.Location);
close(figure(610))
figure(610)
pcshow(sc8.Location(1:sc_length-length_red,:,:))
hold on
pcshow(select(sc8,(sc_length-length_red+1):sc_length))
hold off
caxis([0 max(sc8.Location(:,3))])
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
view(-0.7451171875,-25.396484375)
set(gca,'CameraPosition',[-0.241895390758655,-9.330243407823593,-8.242878380159416])
set(gca,'CameraTarget',[0.009728916763747,-0.101421138241251,1.317361914327423])
set(gca,'CameraViewAngle',5.527523426185996)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraUpVector',[-0.018108003265039,-0.725546967091094,0.687934370970566])