function [x,u] = scp(x_old, u_old, u_lb, u_ub, f, linearize_dynamics, Q,R, Qf, goal_state, x0, num_steps, dt)
    % README:
    % We will use CVX to solve the convex approximations to the optimal
    % control problem in our SCP implementation.

    % VARIABLE NAMES
    % CVX will only solve problems with 1 decision variable, so to adhere
    % with this, we will form a single, combined variable z = [x,u] by
    % concatenating the state and control variables.

    % INDEXING:
    % Recall that the state at any given time is a 4-dimensional vector.
    % x = (x_1, x_2, x_3, ... x_num_steps) is a concatenation of the state vectors at each time.
    % Therefore, x_1 = z(1:4), z_2 = x(5:8), and in general, x_i = z((i-1)*4 : i*4)
    % For your convenience, we have provided some indexing functions so
    % that x_i = z(x_start(i) : x_end(i)) for each i.
    % similarly, you can access the controls via
    % u_i = z(u_start(i),u_end(i))

    z_old = [x_old;u_old]; % concatenate the nominal state and control to form a nominal z vector.

    %% SET UP INDEXING FUNCTIONS

    n = size(Q,1); % get the state dimension
    m = size(R,1); % get the control dimension
    u_shift = n*num_steps; % what is the last index of z that represents a state? (This is used for indexing purposes). 

    x_start = @(i) (i-1)*n + 1; % get the first entry of x_i in the vector z
    x_end = @(i) i*n; % get the last entry of x_i in the vector z
    u_start = @(i) u_shift + (i-1)*m + 1; % get the first entry of u_i in the vector z
    u_end = @(i) u_shift + i*m; % get the last entry of u_i in the vector z

    %% SET CONSTRAINTS FOR CONTROL EFFORT

    % set the box constraints for x and u
    lb = -1000*ones((n+m)*num_steps,1); % x is the first n*num_steps entries, and u is the remaining m*num_steps entries.
    ub = 1000*ones((n+m)*num_steps,1);
    z0 = zeros((n+m)*num_steps,1);

    lb(u_shift+1:u_shift+m*num_steps) = u_lb*ones(m*num_steps,1); % set lower bound for control
    ub(u_shift+1:u_shift+m*num_steps) = u_ub*ones(m*num_steps,1); % set upper bound for control

    % SET THE TARGET STATE FOR z
    % (the optimization will try to minimize a weighted distance from z to z0 subject to constraints
    for i=1:num_steps
        z0((i-1)*n+1 :i*n) = goal_state;
    end

    %% BUILD THE COST MATRIX
    % so that the cost is (z-z0)'*M*(z-z0)
    M = zeros((n+m)*num_steps);

    for i=1:num_steps
        if (i < num_steps)
            M(x_start(i): x_end(i), x_start(i): x_end(i)) = Q;
        else
            M(x_start(i): x_end(i), x_start(i): x_end(i)) = Qf;
        end
        M(u_start(i): u_end(i), u_start(i): u_end(i)) = R;
    end

    %% Build the linear constraints (which are local approximation to the nonlinear dynamics constraints)
    num_constr = n*num_steps;

    % constraints will have the form C*x == d
    C = zeros(num_constr, (n+m)*num_steps);
    d = zeros(num_constr,1);

    % Fill C,d with the appropriate dynamics matrices to enforce the
    % constraints.
    % Specifically, for each timestep: x_bar,u_bar is the state and control
    % from the nominal trajectory at this timestep.
    % Using euler integration, we first approximate x_t+1 = x_t + dt*f(x_t,u_t)
    % then, we need to linearize f(x_t,u_t) as f(x_bar,u_bar) + other terms
        % TODO - Please initialize and fill up C,d so that C*x = d
        % represents the linearized equality constraints.
    % Fill in 4 rows at a time (for state s at each timestep)
    for t_i = 1:num_steps
        % Linearized dynamics
        x_ref = z_old(x_start(t_i) : x_end(t_i));
        u_ref = z_old(u_start(t_i) : u_end(t_i));
        [A, B, c] = linearize_dynamics(x_ref, u_ref, dt);
        
        % Full dynamics at reference trajectory
        sdot = (f(x_ref, u_ref, dt) - x_ref) / dt;
        
        % Compute noteterm, the term that will go in d vector
        note_term = dt * sdot - B*u_ref - (A - eye(n))*x_ref;
        
        % Fill in 4 rows of d with note_term rows
        d(x_start(t_i):x_end(t_i)) = note_term;
        
        assignin('base','sdot',sdot)
        assignin('base','note_term',note_term);
       
        % Fill in 4 rows of C with corresponding coefficients for s_t,
        % s_t+1, and u_t
        C(x_start(t_i):x_end(t_i),x_start(t_i):x_end(t_i)) = -A;
        C(x_start(t_i):x_end(t_i),x_start(t_i+1):x_end(t_i+1)) = eye(n);
        C(x_start(t_i):x_end(t_i),u_start(t_i):u_end(t_i)) = -B.';
    end
    % initial condition constraint
        % TODO - Please add the initial codition constraint
        % We need additional rows in C and d for the initial conditions x0
    % Add n rows to C and d for initial conditions
    C = [C ; zeros(n,(n+m)*num_steps)];
    d = [d ; zeros(n, 1)];
    
    C(end-3:end,x_start(1):x_end(1)) = eye(n);
    assignin('base','C',C);
    assignin('base','d',d);
    
    %% Build CVX instance of our optimization problem
    cvx_begin quiet

        variable z((n+m)*num_steps); % our decision variable will be z = [x,u] a concatenation of the state and control variables.
        cost = quad_form(z-z0,M);

        minimize(cost)
        subject to
            % TODO
            % TODO: Include linearized dynamics constraints 
            C * z == d;
            lb <= z <= ub; % control effort bounds
            % TODO - Add trust region constraints.
    cvx_end

    var = z; % retrieve the optimal solution from CVX

    x = var(1:n*num_steps); % split z back into x and u
    u = var(u_shift+1:(n+m)*num_steps);

    % If you want, you can uncomment the next line to see the objective
    % value for this iteration.
    fprintf("Current Objective value: %f \n" ,(var-z0)'*M*(var-z0))

end
