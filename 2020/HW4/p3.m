% computes a control invariant set for LTI system x^+ = A*x+B*u
A = [1 1 0; 0 0.9 1; 0 0.2 0];
B = [0;1;0];
xmin = ones(3,1) * -5;
xmax = ones(3,1) * 5;

system = LTISystem('A', A, 'B', B);
system.x.min = xmin;
system.x.max = xmax;
system.u.min = -0.5;
system.u.max = 0.5;
InvSet = system.invariantSet()
InvSet.plot()