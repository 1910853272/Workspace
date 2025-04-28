<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_balconydoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_balconydoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.5 1.7])
set(gca,'ColorScale','log')
view(-2.7421875,-35.2998046875)
set(gca,'CameraPosition',[-0.535319959966499,-9.07810734519598,-8.485126295301846])
set(gca,'CameraTarget',[0.01477039185147,0.133506880983674,1.079194575356448])
set(gca,'CameraViewAngle',3.628210616863957)
set(gca,'CameraUpVector',[0.102810402029054,-0.738435514999175,0.666440553554874])
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
caxis([0.5 1.7])
set(gca,'ColorScale','log')
view(-2.7421875,-35.2998046875)
set(gca,'CameraPosition',[-0.535319959966499,-9.07810734519598,-8.485126295301846])
set(gca,'CameraTarget',[0.01477039185147,0.133506880983674,1.079194575356448])
set(gca,'CameraViewAngle',3.628210616863957)
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_balconydoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_balconydoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.5 1.7])
set(gca,'ColorScale','log')
view(-2.7421875,-35.2998046875)
set(gca,'CameraPosition',[-0.535319959966499,-9.07810734519598,-8.485126295301846])
set(gca,'CameraTarget',[0.01477039185147,0.133506880983674,1.079194575356448])
set(gca,'CameraViewAngle',3.628210616863957)
set(gca,'CameraUpVector',[0.102810402029054,-0.738435514999175,0.666440553554874])
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
caxis([0.5 1.7])
set(gca,'ColorScale','log')
view(-2.7421875,-35.2998046875)
set(gca,'CameraPosition',[-0.535319959966499,-9.07810734519598,-8.485126295301846])
set(gca,'CameraTarget',[0.01477039185147,0.133506880983674,1.079194575356448])
set(gca,'CameraViewAngle',3.628210616863957)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraUpVector',[0.102810402029054,-0.738435514999175,0.666440553554874])