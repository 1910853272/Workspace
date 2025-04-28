<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_mirror_beforeCompletion.ply');
sc8 = pcread('withglass_pir_mirror_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.4])
view(0.16015625,-30.455078125)
set(gca,'CameraPosition',[-0.017709781203362,-18.38275125939006,-24.12555351244493])
set(gca,'CameraTarget',[-0.059570521857907,0.684816759956095,2.122467425443609])
set(gca,'CameraUpVector',[-0.013424670009777,-0.828292589218005,0.560134952383497])
set(gca,'CameraViewAngle',1.72305264360857)
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
caxis([0 4.5])
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.4])
view(0.16015625,-30.455078125)
set(gca,'CameraPosition',[-0.017709781203362,-18.38275125939006,-24.12555351244493])
set(gca,'CameraTarget',[-0.059570521857907,0.684816759956095,2.122467425443609])
set(gca,'CameraUpVector',[-0.013424670009777,-0.828292589218005,0.560134952383497])
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_mirror_beforeCompletion.ply');
sc8 = pcread('withglass_pir_mirror_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.4])
view(0.16015625,-30.455078125)
set(gca,'CameraPosition',[-0.017709781203362,-18.38275125939006,-24.12555351244493])
set(gca,'CameraTarget',[-0.059570521857907,0.684816759956095,2.122467425443609])
set(gca,'CameraUpVector',[-0.013424670009777,-0.828292589218005,0.560134952383497])
set(gca,'CameraViewAngle',1.72305264360857)
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
caxis([0 4.5])
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.4])
view(0.16015625,-30.455078125)
set(gca,'CameraPosition',[-0.017709781203362,-18.38275125939006,-24.12555351244493])
set(gca,'CameraTarget',[-0.059570521857907,0.684816759956095,2.122467425443609])
set(gca,'CameraUpVector',[-0.013424670009777,-0.828292589218005,0.560134952383497])
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraViewAngle',1.72305264360857)