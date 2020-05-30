from model import dynamics, cost
import numpy as np
import matplotlib.pyplot as plt


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []

# For testing, fix Q and R from costfun
Q = costfun.Q
R = costfun.R

plot = True

for n in range(N):
    costs = []
    
    x = dynfun.reset()

    if plot and n == N-1:
      x_list = []
      
    # Reinitialize A and P for iterative least squares
    Ahat = np.zeros((dynfun.m+dynfun.n,dynfun.m+dynfun.n))
    Phat = np.eye(dynfun.m+dynfun.n)
    for t in range(T):

        # TODO compute policy
        # Use the Ricatti recursion. I need some intial guess for A and B?
        # Get A and B from Ahat
        #A = Ahat[:, 0:dynfun.m]
        #B = Ahat[:, dynfun.m:]
        A = Ahat[0:4, 0:dynfun.m]
        B = Ahat[0:4, dynfun.m:]
        print(Ahat)
        print(A)
        print(B) # TODO something is wrong with B, it's always all zeros
        if t == 2:
          quit()
        L, P = dynfun.Riccati(A, B, Q, R)
        
        # TODO compute action
        u = (-L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
  
        if plot and n == N-1:
          x_list.append(x)
        
        # TODO implement recursive least squares update
        # TODO: need to add zero-mean noise?
        # See hint to use iterative least squres
        # Lecture 12 page 14??
        # Least squares output is next state
        y = np.expand_dims(np.pad(xp, (0,2), "constant"),1)
        # Stack x and u for least squares output
        s = np.expand_dims(np.concatenate((x, u)),1)# least squares input
        Ahat += ((Phat @ s) @ (y.T - s.T @ Ahat)) / (1 + s.T @ Phat @ s)
        print(((y.T - s.T @ Ahat)))
        print(Ahat)
        quit()
        Phat -= (Phat @ s @ s.T @ Phat) / (1 + s.T @ Phat @ s)
        x = xp.copy()
        
    total_costs.append(sum(costs))

print(x_list)
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
  plt.show()

print(np.mean(total_costs))
print(total_costs[-1])
