% Problem 2 - continuous LQR

% Define system parameters
dt = 0.1;

r = 3;
q = 1;
h = 4;
Q = [q 0; 0 0];
R = r;
A = [0, 1; 0 -1];
B = [0; 1];
tf = 10;
Qf = [0 0;0 h];
tspan = [tf:-dt:0];
x0 = [1;1];

% Reshape final condition as column vector for ode45
Qf_col = Qf(:);
% Solve Riccati ODE
[tV, V_col] = ode45(@(tV,V_col)cont_lqr_riccati(tV,V_col,A,B,Q,R), tspan, Qf_col);
ep_length = size(V_col,1);

% Here, we use Matlab's built in LQR function to confirm our results
[K,S,E] = lqr(A, B, Q, R, 0);

gains = zeros(2, ep_length);
for i=1:ep_length
    gains(:,i) = (1/R)*B.'*reshape(V_col(i,:), [2,2]);
end

% Use solution of Riccati equation to compute gain, and trajectory
[t, x] = ode45(@(t,x)dyn(t,x,A,B,V_col,tV,R), [0:dt:tf], x0);

fig = figure; 
plot(x)
legend("x1", "x2")
xlabel("time")
title(sprintf("state trajectory for tf = %s", int2str(tf)))
saveas(fig, sprintf("p2_state_tf_%s.png", int2str(tf)))

fig = figure;
plot(gains');
legend("l1","l2")
xlabel("time")
title(sprintf("gains for tf = %s", int2str(tf)))
saveas(fig, sprintf("p2_gains_tf_%s.png", int2str(tf)))

function dxdt = dyn(t,x,A,B,V_col,tV,R)
    closest_t = interp1(tV,1:length(tV),t,"nearest");
    V_curr = reshape(V_col(closest_t,:), [2,2]);
    K_curr = (1/R)*B.'*V_curr;
    dxdt = A*x + B*-K_curr*x;
end
