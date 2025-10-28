# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.

### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3:
After training, derive the optimal policy from the Q-table.

### Step 4:
Implement the Monte Carlo method to estimate state values.

### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name: Lokhnath J
### Register Number: 212223240079
```
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

```
## OUTPUT:
<img width="748" height="309" alt="Screenshot 2025-10-28 160914" src="https://github.com/user-attachments/assets/1e80e73c-4339-45c1-8ef0-d682918c8d67" />
<img width="929" height="736" alt="Screenshot 2025-10-28 160926" src="https://github.com/user-attachments/assets/0b9eca50-1d92-4fa0-8d05-56d90d06e664" />
<img width="454" height="313" alt="Screenshot 2025-10-28 160938" src="https://github.com/user-attachments/assets/3d715898-2efe-45ed-9900-4e462b3b01ff" />
<img width="849" height="729" alt="Screenshot 2025-10-28 160950" src="https://github.com/user-attachments/assets/6178c6ed-e568-40f9-bba0-b4bdc4a743e7" />
<img width="1258" height="510" alt="Screenshot 2025-10-28 161034" src="https://github.com/user-attachments/assets/9a778ba9-d611-4436-9cbd-2f14591cd19f" />
<img width="1260" height="503" alt="Screenshot 2025-10-28 161041" src="https://github.com/user-attachments/assets/ded3fb03-de9b-4440-aff6-b7ea91749e71" />



## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given RL environment using Q-Learning and compared the state values with the Monte Carlo method.
