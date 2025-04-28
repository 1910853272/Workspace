<<<<<<< HEAD
clc
clear
close all

%% load error data
% namelist = dir('errormat\*.mat');
% len = length(namelist);
% for i = 1:len
%     load ('./errormat/' + namelist(i).name);
% end
load errormat\backdoor1_error.mat
load errormat\balcony1_error.mat
load errormat\balcony2_error.mat
load errormat\ball_error.mat
load errormat\fishtank_error.mat
load errormat\frontdoor1_error.mat
load errormat\mirror_error.mat
load errormat\soundproofroom_error.mat

%% Put error mat in a matrix for statistical calculation
Error_mat = [balcony1_error',soundproofroom_error',ball_error',backdoor1_error',...
    balcony2_error',fishtank_error',mirror_error',frontdoor1_error'];
g_mat = [1*ones(size(balcony1_error))',2*ones(size(soundproofroom_error))',3*ones(size(ball_error))',...
    4*ones(size(backdoor1_error))',5*ones(size(balcony2_error))',...
    6*ones(size(fishtank_error))',7*ones(size(mirror_error))',8*ones(size(frontdoor1_error))'];

%% Gray boxplot for getting the position of the box body
close(figure(1))
figure(1)
bp = boxplot(Error_mat,g_mat,'Symbol','.k','Colors','k');
set(gcf,'Color','w')
set(gca,'FontSize',16);
set(gca,'LineWidth',1.5);
ylabel('Error (mm)');
set(gca,'TickDir','out');
xticklabels({'a','b','c','d','e','f','g','h'})
xlim([0.5,8.5]);
ylim([-0.2,4.5])

boxobj = findobj(gca,'Tag','Box');
for ii = 1:length(boxobj)
    X(ii,:) = get(boxobj(ii),'XData');
    Y(ii,:) = get(boxobj(ii),'YData');
end

%% calculate the mean value
meanvalue = grpstats(Error_mat,g_mat,'mean');

%% final boxplot in Fig. 7
close(figure(2))
figure(2)
ax2 = gca;
for jj=1:length(boxobj)
    patch(X(jj,:),Y(jj,:),[102 153 153]/255,'EdgeColor','k','LineWidth',1,'FaceAlpha',1)
end
hold on
boxplot(Error_mat,g_mat,'Symbol','.k','Colors','k');
plot(1:8,meanvalue,'--o','LineWidth',1,'Color',[0.9290 0.6940 0.1250]); % [244,177,131]/255
set(gcf,'Color','w')
set(gca,'FontSize',12);
set(gca,'LineWidth',1);
ylabel('Error (mm)');
set(gca,'TickDir','out');
xticklabels({'a','b','c','d','e','f','g','h'})
xlim([0.5,8.5]);
ylim([-0.2,4.5])
box off
ax1_complement = axes('Position',get(gca,'Position'),...
    'Color','none',...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'XColor','k','YColor','k','LineWidth',1);
set(ax1_complement,'YTick', []);
set(ax1_complement,'XTick', []);

%
boxplotobj = findobj(ax2,'Tag','boxplot');
obj = get(boxplotobj,'children');
for ii=9:56
    obj(ii).LineWidth = 1;
end
for ii=1:8
    obj(ii).MarkerEdgeColor = [0.5 0.5 0.5];
end
=======
clc
clear
close all

%% load error data
% namelist = dir('errormat\*.mat');
% len = length(namelist);
% for i = 1:len
%     load ('./errormat/' + namelist(i).name);
% end
load errormat\backdoor1_error.mat
load errormat\balcony1_error.mat
load errormat\balcony2_error.mat
load errormat\ball_error.mat
load errormat\fishtank_error.mat
load errormat\frontdoor1_error.mat
load errormat\mirror_error.mat
load errormat\soundproofroom_error.mat

%% Put error mat in a matrix for statistical calculation
Error_mat = [balcony1_error',soundproofroom_error',ball_error',backdoor1_error',...
    balcony2_error',fishtank_error',mirror_error',frontdoor1_error'];
g_mat = [1*ones(size(balcony1_error))',2*ones(size(soundproofroom_error))',3*ones(size(ball_error))',...
    4*ones(size(backdoor1_error))',5*ones(size(balcony2_error))',...
    6*ones(size(fishtank_error))',7*ones(size(mirror_error))',8*ones(size(frontdoor1_error))'];

%% Gray boxplot for getting the position of the box body
close(figure(1))
figure(1)
bp = boxplot(Error_mat,g_mat,'Symbol','.k','Colors','k');
set(gcf,'Color','w')
set(gca,'FontSize',16);
set(gca,'LineWidth',1.5);
ylabel('Error (mm)');
set(gca,'TickDir','out');
xticklabels({'a','b','c','d','e','f','g','h'})
xlim([0.5,8.5]);
ylim([-0.2,4.5])

boxobj = findobj(gca,'Tag','Box');
for ii = 1:length(boxobj)
    X(ii,:) = get(boxobj(ii),'XData');
    Y(ii,:) = get(boxobj(ii),'YData');
end

%% calculate the mean value
meanvalue = grpstats(Error_mat,g_mat,'mean');

%% final boxplot in Fig. 7
close(figure(2))
figure(2)
ax2 = gca;
for jj=1:length(boxobj)
    patch(X(jj,:),Y(jj,:),[102 153 153]/255,'EdgeColor','k','LineWidth',1,'FaceAlpha',1)
end
hold on
boxplot(Error_mat,g_mat,'Symbol','.k','Colors','k');
plot(1:8,meanvalue,'--o','LineWidth',1,'Color',[0.9290 0.6940 0.1250]); % [244,177,131]/255
set(gcf,'Color','w')
set(gca,'FontSize',12);
set(gca,'LineWidth',1);
ylabel('Error (mm)');
set(gca,'TickDir','out');
xticklabels({'a','b','c','d','e','f','g','h'})
xlim([0.5,8.5]);
ylim([-0.2,4.5])
box off
ax1_complement = axes('Position',get(gca,'Position'),...
    'Color','none',...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'XColor','k','YColor','k','LineWidth',1);
set(ax1_complement,'YTick', []);
set(ax1_complement,'XTick', []);

%
boxplotobj = findobj(ax2,'Tag','boxplot');
obj = get(boxplotobj,'children');
for ii=9:56
    obj(ii).LineWidth = 1;
end
for ii=1:8
    obj(ii).MarkerEdgeColor = [0.5 0.5 0.5];
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
