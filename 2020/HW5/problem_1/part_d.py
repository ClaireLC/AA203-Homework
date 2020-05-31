from model import dynamics, cost
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

replay = True

if replay:
  N = 1
else:
  N = 15000

T = 100
gamma = 0.95 # discount factor
alpha = 1e-13

total_costs = []

# Initialize W, which parametrizes our policy pi_W
W = np.zeros((dynfun.m, dynfun.n))

# Initialize sigma
std_dev = 0.1
sigma = np.eye(dynfun.m) * std_dev

# True L
L_true,P_true = dynfun.Riccati(dynfun.A,dynfun.B,costfun.Q,costfun.R)
L_diff = [] # List to hold errors between L_true and L

if replay:
  npz_file = np.load("p1d.npz")
  W = npz_file["W"]

plot = True

def get_pi_grad(W,u,x):
  return 0.5 * np.expand_dims(np.linalg.inv(sigma) @ (-u + W@x),1) @ np.expand_dims(x,1).T

for n in tqdm(range(N)):
    costs = []
    x_list = []
    u_list = []
    
    x = dynfun.reset()
    
    # Rollout an entire episode
    for t in range(T):

        # TODO compute action
        u = np.random.normal(W@x, std_dev,size=(dynfun.m))

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)

        x_list.append(x)
        u_list.append(u)
        
        # dynamics step
        xp = dynfun.step(u)

        x = xp.copy()

    total_costs.append(sum(costs))

    if replay:
      break

    # TODO update policy
    for t in range(T):
        # Estimate return since timestep t
        G = 0
        pw = 0
        for r in costs[t:]:
          G += gamma**pw * r
          pw += 1

        # Update W
        u_t = u_list[t]
        x_t = x_list[t]
        W += alpha * G * get_pi_grad(W, u_t, x_t)

    W_err = np.linalg.norm(W-L_true)
    L_diff.append(W_err)
    print("W error: {}".format(W_err))


# Save W
np.savez("p1d", W=W)
print(total_costs)

if plot:
  # Plot state to check if it goes to 0
  x_arr = np.asarray(x_list)
  plt.figure()
  for i in range(4):
    plt.plot(np.linspace(0, T, T), x_arr[:,i])
  plt.xlabel("Time")
  plt.ylabel("x")
  plt.title("State (last episode)")
  plt.savefig("p1d_state.png")

  if not replay:
    plt.figure()
    plt.plot(range(len(L_diff)), L_diff)
    plt.title("L error vs. iteration number")
    plt.xlabel("Iteration")
    plt.savefig("p1d_Lerr.png")

  plt.show()


