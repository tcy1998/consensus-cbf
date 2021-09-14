k_1=0.2;
k_2=0.1;
Gamma=10;
theta_a=pi/4;
k_delta=1;

% vehicle 1 initial offset
x_F_0_1=-6;
y_F_0_1=0;
theta_0_1=0;

% vehicle 2 initial offset
x_F_0_2=-4;
y_F_0_2=-2;
theta_0_2=0;

% vehicle 3 initial offset
x_F_0_3=-5;
y_F_0_3=-4;
theta_0_3=0;

r = 2.0;
tau_r = 1.0;
vr_max = 1.5;
vo_max = 0.75;
psio_max = 0.045;
L = (vr_max + vo_max)/r;
u_ca_max = 10;
vr_min = 1.0;
beta = pi/tau_r;
omega = r + (vr_max + vo_max)*pi/beta +0.1;
gamma = tau_r*beta + pi;
u_tr_max = min(beta,beta^2*omega/(vr_max*gamma)); 
k_2 = gamma^2/beta;
ar_max = beta^2*omega/gamma/pi*0.999;
k = (gamma/beta)^2*(vo_max*psio_max) +0.1;

%u_worst = gamma^2/beta*(k*(1/tau_r)^2 + (vr_max*L)/(r+(vr_min + ...
%    vo_max)/(pi/beta))) + 2*gamma^2/vr_min/tau_r + L
% Generate random control points.
%ctrl_points = [ [0 0]' rand(2,4)*15 [10 10]' ];
%
sim('multi_ca_sim_1.slx')
%generate_plots;
%fancy_plots