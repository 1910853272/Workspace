<<<<<<< HEAD
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_ball_beforeCompletion.ply');
sc8 = pcread('withglass_pir_ball_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.2 1.3])
view(4.9892578125,-16.765625)
set(gca,'CameraPosition',[2.859528168729841,-6.24504377542348,-10.107925921552722])
set(gca,'CameraTarget',[-0.115523165001408,0.186219224227565,1.135765445386712])
set(gca,'CameraViewAngle',2.939188330640521)
set(gca,'CameraUpVector',[0.280597964888408,-0.799237825443845,0.531491938302245])
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
caxis([0.2 1.3])
view(4.9892578125,-16.765625)
set(gca,'CameraPosition',[2.859528168729841,-6.24504377542348,-10.107925921552722])
set(gca,'CameraTarget',[-0.115523165001408,0.186219224227565,1.135765445386712])
set(gca,'CameraViewAngle',2.939188330640521)
=======
% This file is used to show the pointcloud before and after completion
%%
sc3 = pcread('pir_ball_beforeCompletion.ply');
sc8 = pcread('withglass_pir_ball_afterCompletion_combined_red.ply');

%% pointcloud before completion
figure(609)
pcshow(sc3.Location)
set(gcf,'Color','w')
set(gca,'Color','w')
set(gca,'XColor','w','YColor','w','ZColor','w')
caxis([0.2 1.3])
view(4.9892578125,-16.765625)
set(gca,'CameraPosition',[2.859528168729841,-6.24504377542348,-10.107925921552722])
set(gca,'CameraTarget',[-0.115523165001408,0.186219224227565,1.135765445386712])
set(gca,'CameraViewAngle',2.939188330640521)
set(gca,'CameraUpVector',[0.280597964888408,-0.799237825443845,0.531491938302245])
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
caxis([0.2 1.3])
view(4.9892578125,-16.765625)
set(gca,'CameraPosition',[2.859528168729841,-6.24504377542348,-10.107925921552722])
set(gca,'CameraTarget',[-0.115523165001408,0.186219224227565,1.135765445386712])
set(gca,'CameraViewAngle',2.939188330640521)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(gca,'CameraUpVector',[0.280597964888408,-0.799237825443845,0.531491938302245])