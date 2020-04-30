import numpy as np

# Problem 3
# Value iteration

p = 100 # Gross profit
m = -20 # Maintenance cost
r = -40 # Repair cost
repl = -150 # Replacement cost


# Define transition probabilities each (s, a, s'), and rewards
# For state NOT BROKEN (s0), there are 2 actions, ABSTAIN (a0) and MAINTAIN (a1)
T0 = np.array([
               [0.3, 0.7], # ABSTAIN next state [s0, s1]
               [0.6, 0.4], # MAINTAIN next state [s0, s1]
               ])
# For state BROKEN (s1), there are 2 actions, REPAIR (a0) and REPLACE (a1)
T1 = np.array([
               [0.6, 0.4], # REPAIR next state [s0, s1]
               [1, 0], # REPLACE next state [s0, s1]
               ])
T = np.array([T0, T1])

R0 = np.array([
               [p, 0], # ABSTAIN next state [s0, s1]
               [m+p, m], # MAINTAIN next state [s0, s1]
               ])
# For state BROKEN (s1), there are 2 actions, REPAIR (a0) and REPLACE (a1)
R1 = np.array([
               [r+p, r], # REPAIR next state [s0, s1]
               [repl + p, repl], # REPLACE next state [s0, s1]
               ])
R = np.array([R0, R1])

H = 4

# Values at each iteration for states NOT BROKEN (s0) and BROKEN (s1)
# Element (i,j) is value at time i at state j
V = np.zeros((H+1,2))
A_opt = np.zeros((H+1,2))


for i in range(1, H+1):
  for s in range(2):
    max_V = -np.inf
    max_a = None
    for a in range(2):
      total = 0
      for s_next in range(2): 
        total += (T[s, a, s_next] * (R[s, a, s_next] + V[i-1, s_next]))
      if total > max_V:
        max_V = total
        max_a = a
    V[i,s] = max_V
    A_opt[i,s] = max_a

print(V)
print(A_opt)

