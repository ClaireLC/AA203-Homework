% Dynamics of the problem

function [xDyn,yDyn] = fDyn(x,y,u)

% Put here the dynamics
xDyn = -x.*u + y.*u.*u;
yDyn = x.*u - 3.*y.*u.*u;
