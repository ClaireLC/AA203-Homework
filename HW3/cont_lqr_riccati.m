% Ricatti equation for continuous LQR

function dVdt_col = cont_lqr_riccati(t,V_col,A,B,Q,R)
% Shape V_col into [2,2] matrix
V = reshape(V_col, [2,2]);

dVdt = -1 * (Q - V*B*(1/R)*B.'*V + V*A + A.'*V);

%Reshape dVdt into column vector (4,1)
dVdt_col = dVdt(:); 
end