from model import dynamics, cost
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []

A_true = dynfun.A
B_true = dynfun.B

Q_true = costfun.Q
R_true = costfun.R

# For testing, fix Q and R from costfun
#Q = Q_true
#R = R_true

# True L
L_true,P_true = dynfun.Riccati(A_true,B_true,Q_true,R_true)

plot = True

# Initialize matrices for least squares for dynamics model
A0 = np.random.rand(dynfun.n, dynfun.n)
B0 = np.random.rand(dynfun.n, dynfun.m)
Chat = np.vstack((A0.T,B0.T))
Phat = np.eye(dynfun.m+dynfun.n)

# Initialize matrices for least squares for cost function
Q0 = np.eye(dynfun.n)
Q0[0,1] = 2
R0 = np.eye(dynfun.m)
Chat_cost = np.expand_dims(np.concatenate((Q0.flatten(), R0.flatten())),1)
Phat_cost = np.eye(dynfun.m**2+dynfun.n**2)

# Intialize P and L for Riccati update
A = Chat[0:dynfun.n, :].T
B = Chat[dynfun.n:, :].T
P = Q0
L = -np.linalg.inv(R0 + B.T @ P @ B) @ (B.T @ P @ A)

L_diff = [] # List to hold errors between L_true and L

for n in tqdm(range(N)):
    costs = []
    
    x = dynfun.reset()

    if plot and n == N-1:
      x_list = []
      
    for t in range(T):
        # TODO compute policy
        # Get A and B from Chat
        A = Chat[0:dynfun.n, :].T
        B = Chat[dynfun.n:, :].T

        # Get Q and R from Chat_cost
        Q = Chat_cost[:dynfun.n**2, :].reshape(Q_true.shape)
        R = Chat_cost[dynfun.n**2:, :].reshape(R_true.shape)

        #L, P = dynfun.Riccati(A, B, Q, R)
        P = Q + L.T @ R @ L + (A + B @ L).T @ P @ (A + B @ L) 
        L = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # TODO compute action
        u = (L @ x)
        L_diff.append(np.linalg.norm(L-L_true))
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
  
        if plot and n == N-1:
          x_list.append(x)
        
        # TODO implement recursive least squares update
        
        # Iterative least squares update for dynamics mode 
        # Least squares output is next state
        y = np.expand_dims(xp,1)
        # Stack x and u for least squares input
        s = np.expand_dims(np.concatenate((x, u)),1)# least squares input
        Chat += ((Phat @ s) @ (y.T - s.T @ Chat)) / (1 + s.T @ Phat @ s)
        Phat -= (Phat @ s @ s.T @ Phat) / (1 + s.T @ Phat @ s)

        # Iterative least squares update fro cost
        y_cost = c # Output is true cost
    
        # Form s
        # x-sqrd terms
        x = np.array([1,2,3,4]) # for testing
        x_sqrd = np.zeros((dynfun.n, dynfun.n))
        for i in range(dynfun.n):
          for j in range(dynfun.n):
            x_sqrd[i,j] = x[i] * x[j]
        x_vec = x_sqrd.flatten()
        # u-sqrd terms
        u_sqrd = np.zeros((dynfun.m, dynfun.m))
        for i in range(dynfun.m):
          for j in range(dynfun.m):
            u_sqrd[i,j] = u[i] * u[j]
        u_vec = u_sqrd.flatten()
        z = np.expand_dims(np.concatenate((x_vec, u_vec),0),1)

        Chat_cost += ((Phat_cost @ z) @ (y_cost.T - z.T @ Chat_cost)) / (1 + z.T @ Phat_cost @ z)
        Phat_cost -= (Phat_cost @ z @ z.T @ Phat_cost) / (1 + z.T @ Phat_cost @ z)
      
        # Update x
        x = xp.copy()
            
    total_costs.append(sum(costs))

if plot:
  # Plot state to check if it goes to 0
  x_arr = np.asarray(x_list)
  plt.figure()
  for i in range(4):
    plt.plot(np.linspace(0, T, T), x_arr[:,i])
  plt.xlabel("Time")
  plt.ylabel("x")
  plt.title("State (last episode)")
  plt.savefig("p1b_state.png")

  plt.figure()
  plt.plot(range(len(L_diff)), L_diff)
  plt.title("L error vs. iteration number")
  plt.xlabel("Iteration")
  plt.savefig("p1b_Lerr.png")

  # Plot cost vs. episode
  plt.figure()
  plt.title("cost vs. episode number")
  plt.xlabel("Episode")
  plt.ylabel("Cost")
  plt.plot(range(N), total_costs)
  plt.show()

print(np.mean(total_costs))
print(total_costs[-1])
