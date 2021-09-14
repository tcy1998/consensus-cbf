close all;

figure
axes('FontSize',10);
plot(topology.time,topology.signals.values,'linewidth',2);
set(gca,'FontSize',20)
yticks(0:1:10.5);
xlabel('time [s]');
ylabel('topology');
ylim([0 10.5]);
% axis([0 25 0 10.5]);

% figure
% axes('FontSize',10);
% plot(xi.time,xi.signals.values,'linewidth',1);
% xlabel('time [s]');
% ylabel('$x_i$','Interpreter','latex','FontSize',15);
% legend('$x_1$','$x_2$','$x_3$','Interpreter','latex','Location','southeast','FontSize',11)

figure
axes('FontSize',10);
hold on;
plot(xi.time,xi.signals.values(:,1)-xi.signals.values(:,2),'linewidth',2);
hold on;
plot(xi.time,xi.signals.values(:,2)-xi.signals.values(:,3),'linewidth',2);
hold on;
plot(xi.time,xi.signals.values(:,3)-xi.signals.values(:,1),'linewidth',2);
set(gca,'FontSize',20)
% yticks(-0.15:0.075:0.15);
xlabel('time [s]');
ylabel('$x_i-x_j$','Interpreter','latex','FontSize',30);
legend('$x_1-x_2$','$x_2-x_3$','$x_3-x_1$','Interpreter','latex','Location','southeast','FontSize',20)
axis([0 25 -0.15 0.151])

figure
axes('FontSize',10);
hold on;
plot(xi_dot.time,xi_dot.signals.values,'linewidth',2);
set(gca,'FontSize',20)
xlabel('time [s]');
ylabel('$\dot{x}_i$','Interpreter','latex','FontSize',30);
legend('$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$','Interpreter','latex','Location','southeast','FontSize',20)
axis([0 25 0.85 1.25])

figure
axes('FontSize',10);
hold on;
plot(chi.time,chi.signals.values,'linewidth',2);
set(gca,'FontSize',20)
xlabel('time [s]');
ylabel('$\chi_i$','Interpreter','latex','FontSize',30);
legend('$\chi_2$','$\chi_3$','Interpreter','latex','Location','southeast','FontSize',20)
axis([0 25 0.65 1.2])
