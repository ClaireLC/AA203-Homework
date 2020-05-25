% HW4 problem 4

% Part a
% Define system parameters
A = [0.99 1;0 0.99];
B = [0; 1];
xlim = 5;
ulim = 0.5;
n = 2; % State dimensions
m = 1; % Control dimensions

Q = eye(n);
R = eye(m);

% Define LTI system for MPT
system = LTISystem('A', A, 'B', B);
system.x.min = ones(n,1)*-xlim;
system.x.max = ones(n,1)*xlim;
system.u.min = -ulim;
system.u.max = ulim;
% Weights on states and inputs
system.x.penalty = QuadFunction(Q);
system.u.penalty = QuadFunction(R);
% Compute terminal set
Xf = system.LQRSet;
%Xf.plot()
% Compute terminal penalty
P = system.LQRPenalty;

% Part b
% Add terminal set and terminal penalty to system
system.x.with('terminalSet');
system.x.terminalSet = Xf;
system.x.with('terminalPenalty');
system.x.terminalPenalty = P;
% Formulate finite horizon MPC problem
N = 4; % horizon
start_time = cputime;
mpc = MPCController(system,N);
openloop_setup_time = cputime - start_time

% Simulate with same system
start_time = cputime;
x0 = [-4.7;2];
sim_model = system;
loop = ClosedLoop(mpc, sim_model);
Nsim = 10;
data = loop.simulate(x0, Nsim);
openloop_sim_time = cputime - start_time
x_traj = data.X;
fig = figure('visible', 'off');
plot(x_traj')
legend('x1','x2')
saveas(fig, 'p4b.png');

% Part c - explicit MPC controller
start_time = cputime;
exp_mpc = mpc.toExplicit();
explicit_setup_time = cputime - start_time
start_time = cputime;
exp_loop = ClosedLoop(exp_mpc, sim_model);
exp_data = exp_loop.simulate(x0, Nsim);
explicit_sim_time = cputime - start_time
exp_x_traj = exp_data.X;
fig = figure('visible', 'off');
plot(exp_x_traj')
legend('x1','x2')
saveas(fig, 'p4c.png');

fig = figure()
exp_mpc.partition.plot()
saveas(fig, 'p4e.png');


