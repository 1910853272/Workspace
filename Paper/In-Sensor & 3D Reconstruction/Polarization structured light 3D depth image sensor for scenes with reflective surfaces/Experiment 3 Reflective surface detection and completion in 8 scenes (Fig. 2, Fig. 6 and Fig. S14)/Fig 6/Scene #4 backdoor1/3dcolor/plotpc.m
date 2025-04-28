<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_backdoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_backdoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.8])
set(gca,'ColorScale','log')
view(-1.2802734375,-33.369140625)
set(gca,'CameraPosition',[-0.127301959547339,-9.291364146607332,-8.293129303126547])
set(gca,'CameraTarget',[0.017608976314468,0.072720773047184,1.136906974755344])
set(gca,'CameraViewAngle',3.628210616863968)
set(gca,'CameraUpVector',[-0.005352041885713,-0.729807031492159,0.683632249409181])
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
caxis([0 max(sc8.Location(:,3))])
caxis([0.6 1.8])
set(gca,'ColorScale','log')
view(-1.2802734375,-33.369140625)
set(gca,'CameraPosition',[-0.127301959547339,-9.291364146607332,-8.293129303126547])
set(gca,'CameraTarget',[0.017608976314468,0.072720773047184,1.136906974755344])
set(gca,'CameraViewAngle',3.628210616863968)
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_backdoor1_beforeCompletion.ply');
sc8 = pcread('withglass_pir_backdoor1_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.6 1.8])
set(gca,'ColorScale','log')
view(-1.2802734375,-33.369140625)
set(gca,'CameraPosition',[-0.127301959547339,-9.291364146607332,-8.293129303126547])
set(gca,'CameraTarget',[0.017608976314468,0.072720773047184,1.136906974755344])
set(gca,'CameraViewAngle',3.628210616863968)
set(gca,'CameraUpVector',[-0.005352041885713,-0.729807031492159,0.683632249409181])
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
caxis([0 max(sc8.Location(:,3))])
caxis([0.6 1.8])
set(gca,'ColorScale','log')
view(-1.2802734375,-33.369140625)
set(gca,'CameraPosition',[-0.127301959547339,-9.291364146607332,-8.293129303126547])
set(gca,'CameraTarget',[0.017608976314468,0.072720773047184,1.136906974755344])
set(gca,'CameraViewAngle',3.628210616863968)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraUpVector',[-0.005352041885713,-0.729807031492159,0.683632249409181])