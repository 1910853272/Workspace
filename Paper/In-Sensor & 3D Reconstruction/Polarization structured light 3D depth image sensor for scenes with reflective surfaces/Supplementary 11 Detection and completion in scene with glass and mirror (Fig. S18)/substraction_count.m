<<<<<<< HEAD
tempuz_forhist = abs(substract_uz);
temp_ux1 = ux1;
temp_uy1 = uy1;
logic_id_cam_forhist = (tempuz_forhist<300) | (~logical_erode_id_inmap);
tempuz_forhist(logic_id_cam_forhist)=NaN;
temp_ux1(logic_id_cam_forhist)=NaN;
temp_uy1(logic_id_cam_forhist)=NaN;

figure(92)
scatter(temp_ux1,temp_uy1,2,tempuz_forhist,'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
colorbar
%%
close(figure(93))
figure(93)
h93 = histogram(tempuz_forhist,'EdgeColor','none');
set(gca,'LineWidth',1.5,'FontSize',16);
set(gcf,'Color','w')
xlabel('Depth (mm)')
ylabel('Count')
set(gca,'TickDir','out');
set(gca,'XColor','k','YColor','k');

box off
ax1_complement = axes('Position',get(gca,'Position'),...
    'Color','none',...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'XColor','k','YColor','k','LineWidth',1.5);
set(ax1_complement,'YTick', []);
=======
tempuz_forhist = abs(substract_uz);
temp_ux1 = ux1;
temp_uy1 = uy1;
logic_id_cam_forhist = (tempuz_forhist<300) | (~logical_erode_id_inmap);
tempuz_forhist(logic_id_cam_forhist)=NaN;
temp_ux1(logic_id_cam_forhist)=NaN;
temp_uy1(logic_id_cam_forhist)=NaN;

figure(92)
scatter(temp_ux1,temp_uy1,2,tempuz_forhist,'filled');
colormap('jet');
set(gca,'Ydir','reverse')
set(gca,'Color','k')
xlim([0,627])
ylim([0,400])
set(gcf,'Color','w')
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
colorbar
%%
close(figure(93))
figure(93)
h93 = histogram(tempuz_forhist,'EdgeColor','none');
set(gca,'LineWidth',1.5,'FontSize',16);
set(gcf,'Color','w')
xlabel('Depth (mm)')
ylabel('Count')
set(gca,'TickDir','out');
set(gca,'XColor','k','YColor','k');

box off
ax1_complement = axes('Position',get(gca,'Position'),...
    'Color','none',...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'XColor','k','YColor','k','LineWidth',1.5);
set(ax1_complement,'YTick', []);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
set(ax1_complement,'XTick', []);