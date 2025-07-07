import numpy as np
import random

# Environment parameters
n_states = 5            # positions 0 through 4
goal_state = 4
actions = [0, 1]        # 0 = left, 1 = right

# Q-table initialization
Q = np.zeros((n_states, len(actions)))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.1    # exploration rate
episodes = 100

# Define environment step
def step(state, action):
    if action == 0:  # move left
        next_state = max(state - 1, 0)
    else:            # move right
        next_state = min(state + 1, n_states - 1)

    reward = 1 if next_state == goal_state else 0
    done = next_state == goal_state
    return next_state, reward, done

# Training loop
for episode in range(episodes):
    state = 0
    done = False

    while not done:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-learning update
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state

# Show learned Q-values
print("Q-table after training:")
print(Q)
