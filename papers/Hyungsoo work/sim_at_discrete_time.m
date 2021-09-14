traj_1 = traj_1;%.Data; %trajectory
position_1 = pos_1.signals.values; %actual position
position_desired_1 = pos_d_1.signals.values; %target position

traj_2 = traj_2;%.Data;
position_2 = pos_2.signals.values;
position_desired_2 = pos_d_2.signals.values;

traj_3 = traj_3;%.Data;
position_3 = pos_3.signals.values;
position_desired_3 = pos_d_3.signals.values;

close all
figure
cmap =lines;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:1,1),position_1(1:1,2),'.','Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:1,1),position_2(1:1,2),'.','Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:1,1),position_3(1:1,2),'.','Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (1,1),position_desired_1(1,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (1,1),position_desired_2(1,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (1,1),position_desired_3(1,2),'o','Color','b');
hold on

hold off
grid off
% title('t=0sec')
% xlabel('t=0sec','FontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:201,1),position_1(1:201,2),'Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:201,1),position_2(1:201,2),'Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:201,1),position_3(1:201,2),'Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (201,1),position_desired_1(201,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (201,1),position_desired_2(201,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (201,1),position_desired_3(201,2),'o','Color','b');
hold on
    
hold off
grid off
% title('t=2sec')
% xlabel('t=2sec','FontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:301,1),position_1(1:301,2),'Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:301,1),position_2(1:301,2),'Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:301,1),position_3(1:301,2),'Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (301,1),position_desired_1(301,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (301,1),position_desired_2(301,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (301,1),position_desired_3(301,2),'o','Color','b');
hold on
    
hold off
grid off
% title('t=6sec')
% xlabel('t=6sec','FontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:1201,1),position_1(1:1201,2),'Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:1201,1),position_2(1:1201,2),'Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:1201,1),position_3(1:1201,2),'Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (1201,1),position_desired_1(1201,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (1201,1),position_desired_2(1201,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (1201,1),position_desired_3(1201,2),'o','Color','b');
hold on
    
hold off
grid off
% title('t=10sec')
% xlabel('t=10sec','FontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:1501,1),position_1(1:1501,2),'Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:1501,1),position_2(1:1501,2),'Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:1501,1),position_3(1:1501,2),'Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (1501,1),position_desired_1(1501,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (1501,1),position_desired_2(1501,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (1501,1),position_desired_3(1501,2),'o','Color','b');
hold on
    
hold off
grid off
% title('t=15sec')
% xlabel('t=15sec','FontWeight','bold')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
plot(traj_1(:,1),traj_1(:,2),'-.','Color',cmap(3,:),'LineWidth',1)
hold on
plot(traj_2(:,1),traj_2(:,2),'-.','Color',cmap(5,:),'LineWidth',1)
hold on
plot(traj_3(:,1),traj_3(:,2),'-.','Color','b','LineWidth',1)
hold on

plot(position_1(1:2290,1),position_1(1:2290,2),'Color',cmap(3,:),'LineWidth',1.5)
hold on
plot(position_2(1:2290,1),position_2(1:2290,2),'Color',cmap(5,:),'LineWidth',1.5)
hold on
plot(position_3(1:2290,1),position_3(1:2290,2),'Color','b','LineWidth',1.5)
hold on
grid off
axis([-20 50 -10 50])

head = plot(position_desired_1 (2290,1),position_desired_1(2290,2),'o','Color',cmap(3,:));
hold on
head = plot(position_desired_2 (2290,1),position_desired_2(2290,2),'o','Color',cmap(5,:));
hold on
head = plot(position_desired_3 (2290,1),position_desired_3(2290,2),'o','Color','b');
hold on
    
hold off
grid off
% title('t=22.6sec')
% xlabel('t=22.1sec','FontWeight','bold')