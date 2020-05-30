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

# For testing, fix Q and R from costfun
Q_true = costfun.Q
R_true = costfun.R
Q = Q_true
R = R_true

# True L
L_true,P_true = dynfun.Riccati(A_true,B_true,Q_true,R_true)
#L = L_true

plot = True

# Reinitialize A and P for iterative least squares
A0 = np.random.rand(dynfun.m, dynfun.m)
B0 = np.random.rand(dynfun.m, dynfun.n)
Chat = np.vstack((A0.T,B0.T))
A = Chat[0:dynfun.m, :].T
B = Chat[dynfun.m:, :].T
P = Q
L = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

Phat = np.eye(dynfun.m+dynfun.n)

L_diff = []

for n in tqdm(range(N)):
    costs = []
    
    x = dynfun.reset()

    if plot and n == N-1:
      x_list = []
      
    for t in range(T):
        # TODO compute policy
        # Get A and B from Chat
        A = Chat[0:dynfun.m, :].T
        B = Chat[dynfun.m:, :].T

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
        # TODO: need to add zero-mean noise?
        
        # Iterative least squares update for dynamics mode 
        # Least squares output is next state
        y = np.expand_dims(xp,1)
        # Stack x and u for least squares input
        s = np.expand_dims(np.concatenate((x, u)),1)# least squares input
        Chat = Chat + ((Phat @ s) @ (y.T - s.T @ Chat)) / (1 + s.T @ Phat @ s)
        Phat = Phat - (Phat @ s @ s.T @ Phat) / (1 + s.T @ Phat @ s)
        x = xp.copy()
        
    total_costs.append(sum(costs))

#print(x_list)
if plot:
  # Plot state to check if it goes to 0
  x_arr = np.asarray(x_list)
  plt.figure()
  for i in range(4):
    plt.plot(np.linspace(0, T, T), x_arr[:,i])
  plt.xlabel("Time")
  plt.ylabel("x")
  plt.title("Plot to check")
  #plt.savefig("p1b.png")

  plt.figure()
  plt.plot(range(len(L_diff)), L_diff)
  plt.title("L error")
  plt.show()

print(np.mean(total_costs))
print(total_costs[-1])
