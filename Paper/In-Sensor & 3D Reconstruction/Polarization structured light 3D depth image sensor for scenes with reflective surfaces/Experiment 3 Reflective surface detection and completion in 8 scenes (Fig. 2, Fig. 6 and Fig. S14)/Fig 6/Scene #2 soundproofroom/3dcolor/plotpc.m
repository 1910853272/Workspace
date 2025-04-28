<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_soundproof2_beforeCompletion.ply');
sc8 = pcread('withglass_pir_soundproof2_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0 max(sc8.Location(:,3))])
view(-2.7509765625,-18.564453125)
set(gca,'CameraPosition',[-0.605521518773506,-12.360939411739876,-4.153566061125016])
set(gca,'CameraTarget',[0.106942223994681,-0.125541616407346,1.385666545239466])
set(gca,'CameraUpVector',[-0.041630764883881,-0.415641393918591,0.908575319429597])
set(gca,'CameraViewAngle',5.92886783722228)
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
caxis([0 max(sc8.Location(:,3))])
view(-2.7509765625,-18.564453125)
set(gca,'CameraPosition',[-0.605521518773506,-12.360939411739876,-4.153566061125016])
set(gca,'CameraTarget',[0.106942223994681,-0.125541616407346,1.385666545239466])
set(gca,'CameraUpVector',[-0.041630764883881,-0.415641393918591,0.908575319429597])
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_soundproof2_beforeCompletion.ply');
sc8 = pcread('withglass_pir_soundproof2_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0 max(sc8.Location(:,3))])
view(-2.7509765625,-18.564453125)
set(gca,'CameraPosition',[-0.605521518773506,-12.360939411739876,-4.153566061125016])
set(gca,'CameraTarget',[0.106942223994681,-0.125541616407346,1.385666545239466])
set(gca,'CameraUpVector',[-0.041630764883881,-0.415641393918591,0.908575319429597])
set(gca,'CameraViewAngle',5.92886783722228)
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
caxis([0 max(sc8.Location(:,3))])
view(-2.7509765625,-18.564453125)
set(gca,'CameraPosition',[-0.605521518773506,-12.360939411739876,-4.153566061125016])
set(gca,'CameraTarget',[0.106942223994681,-0.125541616407346,1.385666545239466])
set(gca,'CameraUpVector',[-0.041630764883881,-0.415641393918591,0.908575319429597])
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraViewAngle',5.92886783722228)