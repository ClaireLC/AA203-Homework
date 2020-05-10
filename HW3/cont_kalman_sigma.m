% sigma ode for continuous time Kalman filter

function dSdt_col = cont_kalman_sigma(t,S_col,A,C,Sw,Sv)
% Shape S_col into [2,2] matrix
S = reshape(S_col, [2,2]);

dSdt = A*S + S*A.' + Sw - S*C.'*Sv*C*S;

%Reshape dVdt into column vector (4,1)
dSdt_col = dSdt(:); 
end