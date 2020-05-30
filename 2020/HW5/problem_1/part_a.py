from model import dynamics, cost
import numpy as np
import matplotlib.pyplot as plt

dynfun = dynamics(stochastic=False)
# dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

costfun = cost()


T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor


plot = True

A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

L,P = dynfun.Riccati(A,B,Q,R)

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()

    if plot:
      x_list = []

    for t in range(T):
        
        # policy 
        u = (L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
    
        # dynamics step
        x = dynfun.step(u)
        
        if plot:
          x_list.append(x)
        
    total_costs.append(sum(costs))

    if plot:
      # Plot state to check if it goes to 0
      x_arr = np.asarray(x_list)
      plt.figure()
      for i in range(4):
        plt.plot(np.linspace(0, T, T), x_arr[:,i])
      plt.xlabel("Time")
      plt.ylabel("x")
      plt.title("Plot to check LQR")
      plt.savefig("p1a.png")
      plt.show()
      quit()
    
print("LQR average cost: {}".format(np.mean(total_costs)))
