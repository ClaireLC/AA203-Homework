function [L, P] = lqr_infinite_horizon_solution(Q, R)

%% find the infinite horizon L and P through running LQR back-ups
%%   until norm(L_new - L_current, 2) <= 1e-4  
dt = 0.1;
mc = 10; mp = 2.; l = 1.; g= 9.81;

% TODO write A,B matrices
n = 4; % State dimension
a1 = mp * g / mc;
a2 = (mc + mp)*g/(l*mc);

df_ds = [0 0 1 0; 0 0 0 1; 0 a1 0 0; 0 a2 0 0];
df_du = [0; 0; 1/mc; 1/(l*mc)];

A = eye(n) + dt * df_ds
B = dt * df_du;

% TODO implement Riccati recursio

% Initialize
P = Q;
L_prev = zeros(size(R));
L_curr = -inv(R + B.'*P*B)*B.'*P*A;
while(norm(L_prev - L_curr) > 1e-5)
    P = Q + L_curr.'*R*L_curr + (A + B*L_curr).'*P*(A+B*L_curr)
    L_prev = L_curr;
    L_curr = -inv(R + B.'*P*B)*B.'*P*A;
end

L = L_curr;