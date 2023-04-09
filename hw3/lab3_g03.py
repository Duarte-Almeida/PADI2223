#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 3: Partially observable Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (``File >> Download as >> Python (.py)``). Make sure that the resulting script includes all code written in the tasks marked as "**Activity n. N**", together with any replies to specific questions posed. Your file should be named `padi-labKK-groupXXX.py`, where `KK` corresponds to the lab number and the `XXX` corresponds to your group number. Similarly, your homework should consist of a single pdf file named `padi-hwKK-groupXXX.pdf`. You should create a zip file with the lab and homework files and submit it in Fenix **at most 30 minutes after your lab is over**.
# 
# Make sure to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The POMDP model
# 
# Consider once again the garbage collection problem described in the homework and for which you wrote a partially observable Markov decision problem model. In this lab, you will consider a larger version of that same problem, described by the diagram:
# 
# <img src="garbage-big.png">
# 
# Recall that the POMDP should describe the decision-making process of the truck driver. In the above domain,
# 
# * At any time step, garbage is _at most_ in one of the cells marked with a garbage bin. 
# * When the garbage truck picks up the garbage from one of the bins, it becomes ``loaded``. 
# * While the truck is loaded, no garbage appears in any of the marked locations.
# * The driver has six actions available: `Up`, `Down`, `Left`, `Right`, `Pick`, and `Drop`. 
# * Each movement action moves the truck to the adjacent stop in the corresponding direction, if there is one. Otherwise, it has no effect. 
# * The `Pick` action succeeds when the truck is in a location with garbage. In that case, the truck becomes "loaded".
# * The `Drop` action succeeds when the loaded truck is at the recycling plant. After a successful drop, the truck becomes empty, and garbage may now appear in any of the marked cells with a total probability of 0.3.
# * The driver cannot observe whether there is garbage in any of these locations unless if it goes there.
# 
# In this lab you will use a POMDP based on the aforementioned domain and investigate how to simulate a partially observable Markov decision problem and track its state. You will also compare different MDP heuristics with the optimal POMDP solution.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.99$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings. In the garbage collection environment above, for example, there is a total of 462 states, each describing the location of the truck in the environment, the location of the garbage (or `None` if no garbage exists in the environment), and whether the truck is `loaded` or `empty`. Each state is, therefore, a string of the form `"(p, g, t)"`, where:
#     * `p` is one of `0`, ..., `32`, indicating the location of the truck;
#     * `g` is either `None` or one of `1`, `9`, `10`, `11`, `18`, `19`, `20`, `21`, `23`, `27`, `28`, `29`, indicating that no garbage exists (`None`), or that there is garbage in one of the listed stops;
#     * `t` is either `empty` or `loaded`, indicating whether the truck is loaded or not.
# * An array `A` that contains all the actions in the POMDP, also represented as strings. In the garbage collection environment above, for example, each action is represented as a string `"Up"`, `"Down"`, `"Left"`, `"Right"`, `"Pick"`, and `"Drop"`.
# * An array `Z` that contains all the observations in the POMDP, also represented as strings. In the garbage collection environment above, for example, there is a total of 78 observations, each describing the location of truck in the environment, whether it is loaded or empty, and whether it sees garbage in its current location. This means that the strings describing the observations take the form `"(p, g, t)"`, where:
#     * `p` is one of `0`, ..., `32`, indicating the location of the truck;
#     * `g` is either `no garbage` or `full` indicating, respectively, that the driver sees no garbage (either because there is no garbage bin in the current location or because the one in it is empty) or sees a full garbage bin;
#     * `t` is either `empty` or `loaded`, indicating whether the truck is loaded or not.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `O` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(Z)` and  corresponding to the observation probability matrix for one action.
# * An array `c` with dimension `len(X)` &times; `len(A)` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., len(A)), (Oa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `O` is a tuple with `len(A)` elements, where `O[a]` is an `np.array` corresponding to the observation probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# ---

# In[1]:


import numpy as np

def load_pomdp(filename, gamma):
    data = np.load(filename)
    states = ()
    for i in range(data["X"].shape[0]):
        states += (data["X"][i], )
    actions = ()
    for i in range(data["A"].shape[0]):
        actions += (data["A"][i], )
    observations = ()
    for i in range(data["Z"].shape[0]):
        observations += (data["Z"][i], )
    probabilities = ()
    for i in range(data["P"].shape[0]):
        probabilities += (data["P"][i], )
    probabilities_obs = ()
    for i in range(data["O"].shape[0]):
        probabilities_obs += (data["O"][i], )
    
    return (states, actions, observations, probabilities, probabilities_obs, data["c"], float(gamma)) 


# We provide below an example of application of the function with the file `garbage-big.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_pomdp('garbage-big.npz', 0.99)
# 
# rand.seed(42)
# 
# # States
# print('= State space (%i states) =' % len(M[0]))
# print('\nStates:')
# for i in range(min(10, len(M[0]))):
#     print(M[0][i]) 
# 
# print('...')
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('\nRandom state: s =', M[0][s])
# 
# # Last state
# print('\nLast state:', M[0][-1])
# 
# # Actions
# print('= Action space (%i actions) =' % len(M[1]))
# for i in range(len(M[1])):
#     print(M[1][i]) 
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('\nRandom action: a =', M[1][a])
# 
# # Observations
# print('= Observation space (%i observations) =' % len(M[2]))
# print('\nObservations:')
# for i in range(min(10, len(M[2]))):
#     print(M[2][i]) 
# 
# print('...')
# 
# # Random observation
# z = rand.randint(len(M[2]))
# print('\nRandom observation: z =', M[2][z])
# 
# # Last state
# print('\nLast observation:', M[2][-1])
# 
# # Transition probabilities
# print('\n= Transition probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[3][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[3][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][s], M[1][a]))
# print("s' in", np.array(M[0])[np.where(M[3][a][s, :] > 0)])
# 
# # Observation probabilities
# print('\n= Observation probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nObservation probability matrix dimensions (action %s):' % M[1][i], M[4][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[4][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) yields observation(s)' % (M[0][s], M[1][a]))
# print("z in", np.array(M[2])[np.where(M[4][a][s, :] > 0)])
# 
# # Cost
# print('\n= Costs =')
# 
# print('\nCost for the state-action pair (%s, %s):' % (M[0][s], M[1][a]))
# print('c(s, a) =', M[5][s, a])
# 
# # Discount
# print('\n= Discount =')
# print('\ngamma =', M[6])
# ```
# 
# Output:
# 
# ```
# = State space (462 states) =
# 
# States:
# (0, None, empty)
# (0, 1, empty)
# (0, 9, empty)
# (0, 10, empty)
# (0, 11, empty)
# (0, 18, empty)
# (0, 19, empty)
# (0, 20, empty)
# (0, 21, empty)
# (0, 23, empty)
# ...
# 
# Random state: s = (7, 28, empty)
# 
# Last state: (32, None, loaded)
# = Action space (6 actions) =
# Up
# Down
# Left
# Right
# Pick
# Drop
# 
# Random action: a = Right
# = Observation space (78 observations) =
# 
# Observations:
# (8, no garbage, empty)
# (31, no garbage, loaded)
# (27, full, empty)
# (9, no garbage, loaded)
# (2, no garbage, loaded)
# (29, full, empty)
# (4, no garbage, empty)
# (23, no garbage, empty)
# (23, full, empty)
# (11, no garbage, loaded)
# ...
# 
# Random observation: z = (0, no garbage, loaded)
# 
# Last observation: (12, no garbage, loaded)
# 
# = Transition probabilities =
# 
# Transition probability matrix dimensions (action Up): (462, 462)
# Dimensions add up for action "Up"? True
# 
# Transition probability matrix dimensions (action Down): (462, 462)
# Dimensions add up for action "Down"? True
# 
# Transition probability matrix dimensions (action Left): (462, 462)
# Dimensions add up for action "Left"? True
# 
# Transition probability matrix dimensions (action Right): (462, 462)
# Dimensions add up for action "Right"? True
# 
# Transition probability matrix dimensions (action Pick): (462, 462)
# Dimensions add up for action "Pick"? True
# 
# Transition probability matrix dimensions (action Drop): (462, 462)
# Dimensions add up for action "Drop"? True
# 
# State-action pair ((7, 28, empty), Right) transitions to state(s)
# s' in ['(8, 28, empty)']
# 
# = Observation probabilities =
# 
# Observation probability matrix dimensions (action Up): (462, 78)
# Dimensions add up for action "Up"? True
# 
# Observation probability matrix dimensions (action Down): (462, 78)
# Dimensions add up for action "Down"? True
# 
# Observation probability matrix dimensions (action Left): (462, 78)
# Dimensions add up for action "Left"? True
# 
# Observation probability matrix dimensions (action Right): (462, 78)
# Dimensions add up for action "Right"? True
# 
# Observation probability matrix dimensions (action Pick): (462, 78)
# Dimensions add up for action "Pick"? True
# 
# Observation probability matrix dimensions (action Drop): (462, 78)
# Dimensions add up for action "Drop"? True
# 
# State-action pair ((7, 28, empty), Right) yields observation(s)
# z in ['(7, no garbage, empty)']
# 
# = Costs =
# 
# Cost for the state-action pair ((7, 28, empty), Right):
# c(s, a) = 0.501
# 
# = Discount =
# 
# gamma = 0.99
# ```
# 
# **Note:** For debug purposes, we also provide a second file, `garbage-small.npz`, that contains a 6-state POMDP that you can use to verify if your results make sense.

# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n + 1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# 3. The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n + 1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# In[2]:


import numpy as np

def gen_trajectory(pomdp, x0, n):
    (X, A, Z, P, O, c, g) = pomdp
    x = np.zeros(n + 1, dtype = np.int16)
    a = np.zeros(n, dtype = np.int16)
    z = np.zeros(n, dtype = np.int16)
    x[0] = x0
    
    for i in range(n):
        a[i] = np.random.choice(len(A))
        x[i + 1] = np.random.choice(len(X), p = P[a[i]][x[i], :])
        z[i] = np.random.choice(len(Z), p = O[a[i]][x[i + 1], :])
        
    return (x, a, z)


# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# 
# rand.seed(42)
# 
# # Number of steps and initial state
# steps = 10
# s0    = 106 # State (18, 0, 2)
# 
# # Generate trajectory
# t = gen_trajectory(M, s0, steps)
# 
# # Check shapes
# print('Shape of state trajectory:', t[0].shape)
# print('Shape of state trajectory:', t[1].shape)
# print('Shape of state trajectory:', t[2].shape)
# 
# # Print trajectory
# for i in range(steps):
#     print('\n- Time step %i -' % i)
#     print('State:', M[0][t[0][i]], '(state %i)' % t[0][i])
#     print('Action selected:', M[1][t[1][i]], '(action %i)' % t[1][i])
#     print('Resulting state:', M[0][t[0][i+1]], '(state %i)' % t[0][i+1])
#     print('Observation:', M[2][t[2][i]], '(observation %i)' % t[2][i])
# ```
# 
# Output:
# 
# ```
# Shape of state trajectory: (11,)
# Shape of state trajectory: (10,)
# Shape of state trajectory: (10,)
# 
# - Time step 0 -
# State: (8, 9, empty) (state 106)
# Action selected: Right (action 3)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# 
# - Time step 1 -
# State: (10, 9, empty) (state 132)
# Action selected: Pick (action 4)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# 
# - Time step 2 -
# State: (10, 9, empty) (state 132)
# Action selected: Left (action 2)
# Resulting state: (8, 9, empty) (state 106)
# Observation: (8, no garbage, empty) (observation 0)
# 
# - Time step 3 -
# State: (8, 9, empty) (state 106)
# Action selected: Left (action 2)
# Resulting state: (7, 9, empty) (state 93)
# Observation: (7, no garbage, empty) (observation 64)
# 
# - Time step 4 -
# State: (7, 9, empty) (state 93)
# Action selected: Right (action 3)
# Resulting state: (8, 9, empty) (state 106)
# Observation: (8, no garbage, empty) (observation 0)
# 
# - Time step 5 -
# State: (8, 9, empty) (state 106)
# Action selected: Right (action 3)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# 
# - Time step 6 -
# State: (10, 9, empty) (state 132)
# Action selected: Drop (action 5)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# 
# - Time step 7 -
# State: (10, 9, empty) (state 132)
# Action selected: Left (action 2)
# Resulting state: (8, 9, empty) (state 106)
# Observation: (8, no garbage, empty) (observation 0)
# 
# - Time step 8 -
# State: (8, 9, empty) (state 106)
# Action selected: Right (action 3)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# 
# - Time step 9 -
# State: (10, 9, empty) (state 132)
# Action selected: Drop (action 5)
# Resulting state: (10, 9, empty) (state 132)
# Observation: (10, no garbage, empty) (observation 41)
# ```

# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n + 1` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief, and should also be considered). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Suggestion:** You may want to define an auxiliary function `belief_update` that receives a POMDP, a belief, an action and an observation and returns the updated belief.
# 
# **Note:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# In[3]:


import numpy as np

def sample_beliefs(pomdp, n):
    (X, A, Z, P, O, c, g) = pomdp
    b_0 = np.ones((1, len(X))) / len(X)
    x_0 = np.random.choice(len(X))
    (x_traj, a_traj, z_traj) = gen_trajectory(pomdp, x_0, n)
    res = (b_0, )
    B = np.zeros((n + 1, len(X)))
    B[0] = b_0
    
    def belief_update(pomdp, b, a, z):
        (X, A, Z, P, O, c, g) = pomdp
        new_b = b @ P[a] @ np.diag(O[a][:, z])
        return new_b / np.sum(new_b)
    
    prev = b_0
    count = 1
    tol = 10 ** -3
    
    for i in range(n):
        curr = belief_update(pomdp, prev, a_traj[i], z_traj[i])
        if (np.all(np.linalg.norm(curr - B[: count, :], axis = 1) > tol)):
            B[count, :] = curr
            count += 1
            res += (curr, )
        prev = curr
    return res


# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# rand.seed(42)
# 
# # 3 sample beliefs + initial belief
# B = sample_beliefs(M, 3)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(np.round(B[i], 3))
#     print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))
# 
# # 100 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled.' % len(B))
# ```
# 
# Output:
# 
# ```
# 4 beliefs sampled:
# [[0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
#   0.002 0.002 0.002 0.002 0.002 0.002]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.054 0.079 0.079 0.079
#   0.079 0.079 0.079 0.079 0.079 0.079 0.079 0.079 0.079 0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.038 0.08  0.08  0.08
#   0.08  0.08  0.08  0.08  0.08  0.08  0.08  0.08  0.08  0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.026 0.081 0.081 0.081 0.081
#   0.081 0.081 0.081 0.081 0.081 0.081 0.081 0.081 0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# 40 beliefs sampled.
# ```

# ### 3. MDP-based heuristics
# 
# In this section you are going to compare different heuristic approaches for POMDPs discussed in class.

# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. Feel free to reuse one of the functions you implemented in Lab 2 (for example, value iteration).
# 
# ---

# In[4]:


import numpy as np

def solve_mdp(pomdp):
    (X, A, _, P, _, c, g) = pomdp

    tol = 10 ** -8

    # small trick to leverage parallelization
    all_P_a = np.zeros((len(A), len(X), len(X)))
    for i in range(len(A)):
        all_P_a[i] = P[i]
    c_T = c.T

    # initial cost-to-go
    J = np.zeros(len(X))

    while True:
        old_J = J
        J = (c_T + g *(all_P_a @ J)).min(axis = 0)
        if (np.linalg.norm(J - old_J) < tol):
            break
    return (c + g * (all_P_a @ J).T)


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# Q = solve_mdp(M)
# 
# s = 115 # State (8, 28, empty)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# 
# s = 429 # (0, None, loaded)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# 
# s = 239 # State (18, 18, empty)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# ```
# 
# Output:
# 
# ```
# Q-values at state (8, 28, empty): [39.945 39.738 39.945 39.945 40.341 40.341]
# Best action at state (8, 28, empty): Down
# 
# Q-values at state (0, None, loaded): [38.318 37.966 38.318 38.318 38.318 37.695]
# Best action at state (0, None, loaded): Drop
# 
# Q-values at state (18, 18, empty): [38.422 38.803 38.803 38.803 38.185 38.803]
# Best action at state (18, 18, empty): Pick
# ```

# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`. *In all heuristics, ties should be broken randomly, i.e., when maximizing/minimizing, you should randomly select between all maximizers/minimizers*.
# 
# ---

# In[5]:


import numpy as np

def get_heuristic_action(belief, Q_star, strategy):
    if strategy == "mls":
        q_optimal = Q[np.argmax(belief), :]
        return np.random.choice(np.argwhere(q_optimal == np.amin(q_optimal)).flatten())
    
    elif strategy == "av":
        votes = np.zeros(Q_star.shape[1])
        pi = np.argmin(Q_star, axis = 1)
        for a in range(Q_star.shape[1]):
            votes[a] = b[:, pi == a].sum()
        return np.random.choice(np.argwhere(votes == np.amax(votes)).flatten())
        
    elif strategy == "q-mdp":
        aux = (b @ Q_star).flatten()
        return np.random.choice(np.argwhere(aux == np.amin(aux)).flatten())
        
    else:
        return -1


# For example, if you run your function in the examples from **Activity 3** using the $Q$-function from **Activity 4**, you can observe the following interaction.
# 
# ```python
# rand.seed(42)
# 
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Belief (approx.) uniform
# MLS action: Down; AV action: Right; Q-MDP action: Right
# 
# Belief: [ (27, None, empty) : 0.058, (27, 1, empty) : 0.086, (27, 9, empty) : 0.086, (27, 10, empty) : 0.086, (27, 11, empty) : 0.086, (27, 18, empty) : 0.086, (27, 19, empty) : 0.086, (27, 20, empty) : 0.086, (27, 21, empty) : 0.086, (27, 23, empty) : 0.086, (27, 28, empty) : 0.086, (27, 29, empty) : 0.086]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.041, (27, 1, empty) : 0.087, (27, 9, empty) : 0.087, (27, 10, empty) : 0.087, (27, 11, empty) : 0.087, (27, 18, empty) : 0.087, (27, 19, empty) : 0.087, (27, 20, empty) : 0.087, (27, 21, empty) : 0.087, (27, 23, empty) : 0.087, (27, 28, empty) : 0.087, (27, 29, empty) : 0.087]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.029, (27, 1, empty) : 0.088, (27, 9, empty) : 0.088, (27, 10, empty) : 0.088, (27, 11, empty) : 0.088, (27, 18, empty) : 0.088, (27, 19, empty) : 0.088, (27, 20, empty) : 0.088, (27, 21, empty) : 0.088, (27, 23, empty) : 0.088, (27, 28, empty) : 0.088, (27, 29, empty) : 0.088]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.020, (27, 1, empty) : 0.089, (27, 9, empty) : 0.089, (27, 10, empty) : 0.089, (27, 11, empty) : 0.089, (27, 18, empty) : 0.089, (27, 19, empty) : 0.089, (27, 20, empty) : 0.089, (27, 21, empty) : 0.089, (27, 23, empty) : 0.089, (27, 28, empty) : 0.089, (27, 29, empty) : 0.089]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.014, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.010, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.007, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.005, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (27, None, empty) : 0.003, (27, 1, empty) : 0.091, (27, 9, empty) : 0.091, (27, 10, empty) : 0.091, (27, 11, empty) : 0.091, (27, 18, empty) : 0.091, (27, 19, empty) : 0.091, (27, 20, empty) : 0.091, (27, 21, empty) : 0.091, (27, 23, empty) : 0.091, (27, 28, empty) : 0.091, (27, 29, empty) : 0.091]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# ```

# You will now implement the last heuristic, the "Fast Informed Bound" (or FIB) heuristic. To that purpose, you will write a function to compute the FIB Q-function.
# 
# ---
# 
# #### Activity 6
# 
# Write a function `solve_fib` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **FIB $Q$-function**, that verifies the recursion
# 
# $$Q_{FIB}(x,a)=c(x,a)+\gamma\sum_{z\in\mathcal{Z}}\min_{a'\in\mathcal{A}}\sum_{x'\in\mathcal{X}}\mathbf{P}(x'\mid x,a)\mathbf{O}(z\mid x',a)Q_{FIB}(x',a').$$
# 
# Stop the algorithm when the error between iterations is smaller than $10^{-1}$. Run the example code below to compare all the heuristics. What can you conclude from the results?
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# In[6]:


import copy
import numpy as np 

def solve_fib(pomdp):
    (X, A, Z, P, O, c, g) = pomdp
    tol = 10 ** -1
    Q_fib = solve_mdp(pomdp)
    while True:
        Q_old = copy.deepcopy(Q_fib)
        for x in range(len(X)):
            for a in range(len(A)):
                aux_z = np.zeros(len(Z))
                for z in range(len(Z)):
                    aux_z[z] = np.amin(P[a][[x], :] @ (Q_old * O[a][:, [z]]), axis = 1)
                Q_fib[x, a] = c[x, a] + g * np.sum(aux_z)
        if (np.linalg.norm(Q_fib - Q_old) < tol):
            break
    
    return Q_fib


# <font color="blue">Every heuristic chooses the same action.</font>

# Using the function `solve_fib` in the function from `get_heuristic_action` from Activity 5 for the beliefs in the example from **Activity 3**, you can observe the following interaction.
# 
# ```python
# Qfib = solve_fib(M)
# 
# s = 115 # State (8, 28, empty)
# print('\nQ-values at state %s:' % M[0][s], np.round(Qfib[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Qfib[s, :])])
# 
# s = 429 # (0, None, loaded)
# print('\nQ-values at state %s:' % M[0][s], np.round(Qfib[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Qfib[s, :])])
# 
# s = 239 # State (18, 18, empty)
# print('\nQ-values at state %s:' % M[0][s], np.round(Qfib[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Qfib[s, :])])
# 
# print()
# 
# rand.seed(42)
# 
# # Comparing the prescribed actions
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
#     print('FIB action:', M[1][get_heuristic_action(b, Qfib, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Q-values at state (8, 28, empty): [39.876 39.673 39.876 39.876 40.274 40.274]
# Best action at state (8, 28, empty): Down
# 
# Q-values at state (0, None, loaded): [38.281 37.927 38.281 38.281 38.281 37.659]
# Best action at state (0, None, loaded): Drop
# 
# Q-values at state (18, 18, empty): [38.372 38.754 38.754 38.754 38.137 38.754]
# Best action at state (18, 18, empty): Pick
# 
# Belief (approx.) uniform
# MLS action: Down; AV action: Right; Q-MDP action: Right; FIB action: Right
# 
# Belief: [ (27, None, empty) : 0.058, (27, 1, empty) : 0.086, (27, 9, empty) : 0.086, (27, 10, empty) : 0.086, (27, 11, empty) : 0.086, (27, 18, empty) : 0.086, (27, 19, empty) : 0.086, (27, 20, empty) : 0.086, (27, 21, empty) : 0.086, (27, 23, empty) : 0.086, (27, 28, empty) : 0.086, (27, 29, empty) : 0.086]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.041, (27, 1, empty) : 0.087, (27, 9, empty) : 0.087, (27, 10, empty) : 0.087, (27, 11, empty) : 0.087, (27, 18, empty) : 0.087, (27, 19, empty) : 0.087, (27, 20, empty) : 0.087, (27, 21, empty) : 0.087, (27, 23, empty) : 0.087, (27, 28, empty) : 0.087, (27, 29, empty) : 0.087]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.029, (27, 1, empty) : 0.088, (27, 9, empty) : 0.088, (27, 10, empty) : 0.088, (27, 11, empty) : 0.088, (27, 18, empty) : 0.088, (27, 19, empty) : 0.088, (27, 20, empty) : 0.088, (27, 21, empty) : 0.088, (27, 23, empty) : 0.088, (27, 28, empty) : 0.088, (27, 29, empty) : 0.088]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.020, (27, 1, empty) : 0.089, (27, 9, empty) : 0.089, (27, 10, empty) : 0.089, (27, 11, empty) : 0.089, (27, 18, empty) : 0.089, (27, 19, empty) : 0.089, (27, 20, empty) : 0.089, (27, 21, empty) : 0.089, (27, 23, empty) : 0.089, (27, 28, empty) : 0.089, (27, 29, empty) : 0.089]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.014, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.010, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.007, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.005, (27, 1, empty) : 0.090, (27, 9, empty) : 0.090, (27, 10, empty) : 0.090, (27, 11, empty) : 0.090, (27, 18, empty) : 0.090, (27, 19, empty) : 0.090, (27, 20, empty) : 0.090, (27, 21, empty) : 0.090, (27, 23, empty) : 0.090, (27, 28, empty) : 0.090, (27, 29, empty) : 0.090]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# 
# Belief: [ (27, None, empty) : 0.003, (27, 1, empty) : 0.091, (27, 9, empty) : 0.091, (27, 10, empty) : 0.091, (27, 11, empty) : 0.091, (27, 18, empty) : 0.091, (27, 19, empty) : 0.091, (27, 20, empty) : 0.091, (27, 21, empty) : 0.091, (27, 23, empty) : 0.091, (27, 28, empty) : 0.091, (27, 29, empty) : 0.091]
# MLS action: Down; AV action: Down; Q-MDP action: Down; FIB action: Down
# ```
