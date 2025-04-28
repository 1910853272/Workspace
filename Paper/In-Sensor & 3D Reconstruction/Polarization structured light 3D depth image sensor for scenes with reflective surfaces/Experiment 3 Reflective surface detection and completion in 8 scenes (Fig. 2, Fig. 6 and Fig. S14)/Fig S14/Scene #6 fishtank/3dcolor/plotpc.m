<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_fishtank_beforeCompletion.ply');
sc8 = pcread('withglass_pir_fishtank_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.3 1.9])
view(-3.189453125,-74.515625)
set(gca,'CameraPosition',[-0.06407226694207,-1.043892144061376,-11.493275835313296])
set(gca,'CameraTarget',[0.088398444761099,-0.049102329646608,1.918763814546683])
set(gca,'CameraUpVector',[-0.043208723806462,-0.99763509129321,0.053453071075531])
set(gca,'CameraViewAngle',3.152862174007214)
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
caxis([0.3 1.9])
view(-3.189453125,-74.515625)
set(gca,'CameraPosition',[-0.06407226694207,-1.043892144061376,-11.493275835313296])
set(gca,'CameraTarget',[0.088398444761099,-0.049102329646608,1.918763814546683])
set(gca,'CameraUpVector',[-0.043208723806462,-0.99763509129321,0.053453071075531])
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_fishtank_beforeCompletion.ply');
sc8 = pcread('withglass_pir_fishtank_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.3 1.9])
view(-3.189453125,-74.515625)
set(gca,'CameraPosition',[-0.06407226694207,-1.043892144061376,-11.493275835313296])
set(gca,'CameraTarget',[0.088398444761099,-0.049102329646608,1.918763814546683])
set(gca,'CameraUpVector',[-0.043208723806462,-0.99763509129321,0.053453071075531])
set(gca,'CameraViewAngle',3.152862174007214)
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
caxis([0.3 1.9])
view(-3.189453125,-74.515625)
set(gca,'CameraPosition',[-0.06407226694207,-1.043892144061376,-11.493275835313296])
set(gca,'CameraTarget',[0.088398444761099,-0.049102329646608,1.918763814546683])
set(gca,'CameraUpVector',[-0.043208723806462,-0.99763509129321,0.053453071075531])
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraViewAngle',3.152862174007214)