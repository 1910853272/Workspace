<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_frontdoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_frontdoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.4 1.8])
view(-3.544921875,-32.099609375)
set(gca,'CameraPosition',[-0.671582575835681,-12.043298340986727,-4.587753250216942])
set(gca,'CameraTarget',[-0.020761511292998,-0.035480037300497,1.070992307909132])
set(gca,'CameraViewAngle',4.031029245067117)
set(gca,'CameraUpVector',[-0.022583712626898,-0.419405801548976,0.90751790591208])
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
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.4 1.8])
view(-3.544921875,-32.099609375)
set(gca,'CameraPosition',[-0.671582575835681,-12.043298340986727,-4.587753250216942])
set(gca,'CameraTarget',[-0.020761511292998,-0.035480037300497,1.070992307909132])
set(gca,'CameraViewAngle',4.031029245067117)
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_frontdoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_frontdoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.4 1.8])
view(-3.544921875,-32.099609375)
set(gca,'CameraPosition',[-0.671582575835681,-12.043298340986727,-4.587753250216942])
set(gca,'CameraTarget',[-0.020761511292998,-0.035480037300497,1.070992307909132])
set(gca,'CameraViewAngle',4.031029245067117)
set(gca,'CameraUpVector',[-0.022583712626898,-0.419405801548976,0.90751790591208])
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
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.4 1.8])
view(-3.544921875,-32.099609375)
set(gca,'CameraPosition',[-0.671582575835681,-12.043298340986727,-4.587753250216942])
set(gca,'CameraTarget',[-0.020761511292998,-0.035480037300497,1.070992307909132])
set(gca,'CameraViewAngle',4.031029245067117)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraUpVector',[-0.022583712626898,-0.419405801548976,0.90751790591208])