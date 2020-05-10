% Problem 3

% Define system parameters
A = [0 1;1 0];
B = [0; 1];
Q = [3 0;0 3];
R = 1;
Qf = [0 0;0 0];
C = [0 1];

tf = 15;
dt = 0.01;
%%  Part a

% Continuous LQR Riccati equation
% Reshape final condition as column vector for ode45
Qf_col = Qf(:);
% Solve Riccati ODE
[tV, V_col] = ode45(@(tV,V_col)cont_lqr_riccati(tV,V_col,A,B,Q,R), [tf:-dt:0], Qf_col);
ep_length = size(V_col,1);

LQR_gains = zeros(2, ep_length);
% Use solution of Riccati equation to compute gain
for t=1:ep_length
   V_curr = reshape(V_col(t, :),[2,2]);
   L = (1/R) * B.' * V_curr;
   LQR_gains(:,t) = L;
end

% Continuous Kalman filter
sigma0 = eye(2) * 5;
xhat0 = [0, 0];
Sw = [0 0;0 4]; % Process noise covariance
Sv = 0.5; % Measurement noise covariance

[tK, S_col] = ode45(@(tK,S_col)cont_kalman_sigma(tK,S_col,A,C,Sw,Sv), [0:dt:tf], sigma0(:));

% Compute Kalman gain
K_gains = zeros(2, ep_length);
for t=1:ep_length
   S_curr = reshape(S_col(t, :),[2,2]);
   K = S_curr * C.' * (1/Sv);
   K_gains(:,t) = K(:);
end

fig = figure;
plot(K_gains');
legend("L1","L2")
xlabel("time")
title("Kalman gains")
saveas(fig, "p3_Kgains.png")

% Plot gains
fig = figure;
plot(LQR_gains');
legend("L1","L2")
xlabel("time")
title("LQR Gains")
saveas(fig, "p3_LQRgains.png")

%% Part c

% Simulate closed-loop system 

x_0 = [10; -10];
xc_0 = [0; 0];
Sys0 = [x_0; xc_0];
 
% For now, use steady-state gains (change to time-dependent when this
% converges)

m_noise = randn(size(V_col,1),1) * Sv;
p_noise = randn(size(V_col,1),2);
[t,Sys] = ode45(@(t,Sys)closed_loop(t, Sys, A, B, C, R, V_col, K_gains, tV, tK, m_noise, p_noise), [0:dt:tf], Sys0);

fig = figure;
plot(Sys)
legend("x1", "x2", "xc1", "xc2")
title("True state and state estimates")
saveas(fig, "p3.png")

% Solve coupled ODEs for dxcdt and dxdt
% Sys is a (4,1) column vector that stacks x and xc
function dSysdt = closed_loop(t, Sys, A, B, C, R, V_col, K_gains, tV, tK,m,p)
    x = Sys(1:2);
    xc = Sys(3:4);

    closest_t_ind_V = interp1(tV,1:length(tV),t,"nearest");
    closest_t_ind_K = interp1(tK,1:length(tK),t,"nearest");
    
    V = reshape(V_col(closest_t_ind_V, :),[2,2]);
    K = reshape(K_gains(:, closest_t_ind_K),[2,1]);

    Ac = A - B*(1/R)*B.'*V - K*C;
    Bc = K;
    Cc = (1/R)*B.'*V;
    Sw = [0 0;0 4]; % Process noise covariance
    dxdt = A*x - B*Cc*xc + Sw*p(closest_t_ind_V,:).';
    dxcdt = Ac*xc + Bc*C*x + m(closest_t_ind_V);
    
    dSysdt = [dxdt;dxcdt];
end