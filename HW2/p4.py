# Gridworld quadrotor in a storm
# Value iteration
import numpy as np
import seaborn
import matplotlib.pyplot as plt

sigma = 10
gamma = 0.95

n = 20
eye  = np.array([15, 15]) # Location of center of storm
goal = np.array([9, 9])

#n = 5
#eye  = np.array([4, 4]) # Location of center of storm
#goal = np.array([2, 2])

# Action enumeration
actions = {
          0: "u",
          1: "d",
          2: "l",
          3: "r"
          }

# Return next grid location and reward after taking action a from state s
def take_action(s, a_num):
  a = actions[a_num]
  s_next = np.copy(s)
  if a == "u":
    s_next[0] -= 1
  elif a == "d":
    s_next[0] += 1
  elif a == "l":
    s_next[1] -= 1
  elif a == "r":
    s_next[1] += 1

  # Cap s_next at grid boundaries
  if (s_next[0] >= n or s_next[1] >= n or s_next[0] < 0 or s_next[1] < 0):
    s_next = s
  
  # Determine reward
  r = 0
  if (s_next[0] == goal[0] and s_next[1] == goal[1]):
    r = 1
  return s_next, r

# Values at each iteration
V_prev = np.zeros((n,n))
V = np.zeros((n,n))

A = np.zeros((n,n)) # Action table

delta = 1
while(delta >= 1e-2):
  for x1 in range(n):
    for x2 in range(n):
      max_V = -np.inf
      max_a = None
      s = np.array([x1,x2])
      px = np.exp(-np.linalg.norm(s-eye)**2 / (2*sigma**2))
      for a in range(4):
        total = 0
        # with probability 1 - px, take action a
        s_next, r = take_action(s, a)
        total += ((1-px) * (r + gamma*V_prev[s_next[0], s_next[1]]))
        # with probability px/4, take each of the four possible actions
        for a_sub in range(4):
          s_next, r = take_action(s, a_sub)
          total += ((px/4) * (r + gamma*V_prev[s_next[0], s_next[1]]))
        # Save best value and action
        if total > max_V:
          max_V = total
          max_a = a
      V[s[0], s[1]] = max_V
      A[s[0], s[1]] = max_a
      delta = np.amax(abs(V - V_prev))
  V_prev = np.copy(V)

eye  = np.array([15, 15]) # Location of center of storm
goal = np.array([9, 9])

#seaborn.set_palette(seaborn.color_palette("RdYlGn"))
seaborn.heatmap(V, annot=False, linewidths=0.5, cmap="RdYlGn")

# Get trajectory given policy A
start = np.array([9, 19])
curr = start
traj = []
while(np.linalg.norm(curr - goal) > 0):
  a = A[curr[0], curr[1]]
  curr, r = take_action(curr, A[curr[0], curr[1]])
  traj.append(actions[a])
    
print(traj)

plt.show()
