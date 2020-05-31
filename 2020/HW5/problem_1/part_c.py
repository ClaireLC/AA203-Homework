from model import dynamics, cost
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 100
gamma = 0.95 # discount factor

total_costs = []

# True L
L_true,P_true = dynfun.Riccati(dynfun.A,dynfun.B,costfun.Q,costfun.R)
L_diff = [] # List to hold errors between L_true and L

plot = True

# Initialize policy randomly
L = np.zeros((dynfun.m, dynfun.n))
#L = np.random.rand(dynfun.m, dynfun.n)
H_dim = dynfun.m + dynfun.n
theta_dim = H_dim*(H_dim+1)//2
theta_hat = np.expand_dims(np.random.rand(theta_dim),1)

for n in tqdm(range(N)):
    costs = []
    
    # Initialize recursive least squares
    P = np.eye(theta_dim)

    x = dynfun.reset()

    if plot and n == N-1:
      x_list = []

    for t in range(T):
        # TODO compute action
        u = L@x + np.random.rand(dynfun.m) * 0.1

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)

        # TODO recursive least squares policy evaluation step
        # Build phi vector
        # Form xu_bar and xUx_bar, which hav same dimension as theta_hat
        xu_vec = np.concatenate((x,u),0)
        xu_bar = np.zeros((theta_dim, 1))
        xUx_vec = np.concatenate((xp, L@x),0)
        xUx_bar = np.zeros((theta_dim, 1))
        ind = 0
        for i in range(xu_vec.shape[0]):
          for j in range(i,xu_vec.shape[0]):
            xu_bar[ind, 0] = xu_vec[i] * xu_vec[j]
            xUx_bar[ind, 0] = xUx_vec[i] * xUx_vec[j]
            ind += 1

        phi = xu_bar - gamma * xUx_bar

        theta_hat += (P @ phi) @ (c.T - phi.T @ theta_hat) / (1 + phi.T @ P @ phi)
        P -= (P @ phi @ phi.T @ P) / (1 + phi.T  @ P @ phi)

        # Save x value
        if plot and n == N-1:
          x_list.append(x)

        x = xp.copy()
    
    # TODO policy improvement step
    # Find matrix H that corresponds to theta. Iterate through theta_hat
    H = np.zeros((dynfun.m + dynfun.n, dynfun.m + dynfun.n))
    ind = 0
    for i in range(H.shape[0]):
      for j in range(i,H.shape[0]):
        if i != j:
          H[i,j] = theta_hat[ind,0] / 2
          H[j,i] = theta_hat[ind,0] / 2
        else:
          H[i,j] = theta_hat[ind,0]
        ind += 1
    
    # Policy improvement
    H_22 = H[-dynfun.m:,-dynfun.m:]
    H_21 = H[-dynfun.m:,:dynfun.n]
    L = -1*np.linalg.inv(H_22) @ H_21
    L_diff.append(np.linalg.norm(L-L_true))

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

  plt.figure()
  plt.plot(range(len(L_diff)), L_diff)
  plt.title("L error vs. iteration number")
  plt.xlabel("Iteration")

  plt.show()

