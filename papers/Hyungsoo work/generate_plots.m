% time = pos.time;
position_obstacle = pos_obs.signals.values;

% traj_1 = traj_1.Data; %trajectory
position_1 = pos_1.signals.values; %actual position
position_desired_1 = pos_d_1.signals.values; %target position

% traj_2 = traj_2.Data;
position_2 = pos_2.signals.values;
position_desired_2 = pos_d_2.signals.values;

% traj_3 = traj_3.Data;
position_3 = pos_3.signals.values;
position_desired_3 = pos_d_3.signals.values;

% video = VideoWriter('circle.mp4','MPEG-4');
% video.FrameRate = 60;
% open(video)

close all
figure
cmap =lines;

% figure('units','normalized','outerposition',[0 0 1 1])

for j=1:10:2290%length(tout)
    plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
    hold on
    plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
    hold on
    plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
    hold on
%     plot(position_1(max(1,j-100):j,1),position_1(max(1,j-100):j,2),'Color',cmap(1,:),'LineWidth',2)
%     hold on
%     plot(position_2(max(1,j-100):j,1),position_2(max(1,j-100):j,2),'Color',cmap(1,:),'LineWidth',2)
%     hold on
%     plot(position_3(max(1,j-100):j,1),position_3(max(1,j-100):j,2),'Color',cmap(1,:),'LineWidth',2)
%     hold on
    plot(position_1(1:j,1),position_1(1:j,2),'Color',cmap(3,:),'LineWidth',1.5)
    hold on
    plot(position_2(1:j,1),position_2(1:j,2),'Color',cmap(5,:),'LineWidth',1.5)
    hold on
    plot(position_3(1:j,1),position_3(1:j,2),'Color','b','LineWidth',1.5)
    hold on
    grid off
    axis([-20 50 -10 50])
%     plot(position_obstacle(max(1,j-200):j,1),position_obstacle(max(1,j-200):j,2),'Color',cmap(9,:),'LineWidth',2)
%     hold on
    head = plot(position_desired_1 (j,1),position_desired_1(j,2),'o','Color',cmap(3,:));
    hold on
    head = plot(position_desired_2 (j,1),position_desired_2(j,2),'o','Color',cmap(5,:));
    hold on
    head = plot(position_desired_3 (j,1),position_desired_3(j,2),'o','Color','b');
    hold on
    
%     plot([position_desired_1(j,1) position_1(j,1)],[position_desired_1(j,2) position_1(j,2)],'k')
%     hold on
%     plot([position_desired_2(j,1) position_2(j,1)],[position_desired_2(j,2) position_2(j,2)],'k')
%     hold on
%     plot([position_desired_3(j,1) position_3(j,1)],[position_desired_3(j,2) position_3(j,2)],'k')
%     hold on
    
%     F = getframe(gcf);
%     writeVideo(video, F);
    
    hold off
    pause(0.01)
%     delete(head)
    grid off
end

% close(video)












% time = pos.time;
% 
% position = pos.signals.values;
% position_obstacle = pos_obs.signals.values;
% position_desired = pos_d.signals.values;
% close all
% figure
% cmap =lines;
% for j=1:10:length(time)
%     plot(position_desired (:,1),position_desired (:,2),'-.','Color',cmap(1,:),'LineWidth',2)
%     hold on
%     plot(position(max(1,j-150):j,1),position(max(1,j-150):j,2),'Color',cmap(1,:),'LineWidth',2)
%     hold on
%     grid on
%     axis([0 50 0 30])
%     plot(position_obstacle(max(1,j-150):j,1),position_obstacle(max(1,j-150):j,2),'Color',cmap(9,:),'LineWidth',2)
%     hold on
%     hold on
%     head = scatter(position_desired (j,1),position_desired(j,2),'g');
%     hold on
%   %  F(k) = getframe(gcf);
%     hold off
%     pause(0.01)
%     grid on
%     delete(head)
% end



% video = VideoWriter('circle.mp4','MPEG-4');
% video.FrameRate = 30;
% open(video)
% writeVideo(video, F);
