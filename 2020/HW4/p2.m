% HW4 problem 2
clear; close all; clc; 

% Define system parameters - these stay constant for all question parts
n = 2; % State dimension
m = 1; % Control dimension
A = [1 1;0 1];
B = [0; 1];
Q = eye(2);


% Part b
% R = 0.01;
% ep_length = 15;
% xbar = 5;
% ubar = 0.5;
% T=3; % Horizon
% P = eye(2);
% x0 = [-4.5; 2]; % first x0
% % x0 = [-4.5; 3]; % second x0
% Xf = inf;
% [Xallmpc, Uallmpc, iters] = run_mpc(x0, T, ep_length, A, B, Q, R, P, xbar, ubar, n, m, Xf);
% fig = figure;
% plot(Xallmpc')
% legend('x1','x2')
% saveas(fig, 'p2b_x0_1.png')

% Part c
R = 10;
[K, Pinf, e] = lqr(A, B, Q, R, 0);
ep_length = 10;
xbar = 10;
ubar = 1;
T=2;
%x0 = [1; 0]; % first x0
Xf = 0;
step_size = 1;
save_str = 'p2c.png';
find_feas_x0(step_size, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf, save_str)

% Part d
T=6;
save_str = 'p2d.png';
find_feas_x0(step_size, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf, save_str)

% Part e
T=2;
Xf = inf;
save_str = 'p2e.png';
find_feas_x0(step_size, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf, save_str)

% Part f
T=6;
Xf = inf;
save_str = 'p2f.png';
find_feas_x0(step_size, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf, save_str)

% Search through the discretized state space for feasible intial conditions
function find_feas_x0(step_size, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf, save_str)
    feasible_x0 = [];
    infeasible_x0 = [];
    for x1=-xbar:step_size:xbar
        for x2=-xbar:step_size:xbar
            x0 = [x1;x2]
            [Xallmpc, Uallmpc, iters] = run_mpc(x0, T, ep_length, A, B, Q, R, Pinf, xbar, ubar, n, m, Xf);
            if iters == ep_length
                feasible_x0 = [feasible_x0, [x1;x2]];
            else
                infeasible_x0 = [infeasible_x0, [x1;x2]];
            end
        end
    end
    fig = figure('visible','off');
    hold on
    feas = scatter(feasible_x0(1,:),feasible_x0(2,:),150,'filled','g');
    inf = scatter(infeasible_x0(1,:),infeasible_x0(2,:),150,'filled','r');
    legend([feas,inf],{'Feasible', 'Infeasible'});
    hold off
    saveas(fig, save_str)
end
