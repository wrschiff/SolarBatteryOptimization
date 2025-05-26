import matplotlib.pyplot as plt
import numpy as np
from parameters import *

def plot_state_cost(state,memo):
    fig, ax = plt.subplots()
    stages = [a[0] for a in memo.keys() if a[1] == state]
    costs = [memo[(stage, 0)][1] for stage in stages]
    ax.plot(stages, costs, marker='o')
    ax.set_xlabel('State')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Function for Stage 1')
def plot_cost_function(memo):
    # derive cost functio 
    fig, ax = plt.subplots()
    all_stages = []
    all_states = []
    all_costs = []
    for stage in range(MAX_STAGE + 1):
        states = [a[1] for a in memo.keys() if a[0] == stage]
        costs = [memo[(stage, state)][1] for state in states]
        all_stages.extend([stage] * len(states))
        all_states.extend(states)
        all_costs.extend(costs)
    
    # finite costs vs inf costs
    finite_indices = [i for i, cost in enumerate(all_costs) if np.isfinite(cost)]
    inf_indices = [i for i, cost in enumerate(all_costs) if not np.isfinite(cost)]

    # finit is dots
    scatter = ax.scatter(
        [all_stages[i] for i in finite_indices],
        [all_states[i] for i in finite_indices],
        c=[all_costs[i] for i in finite_indices],
        cmap='viridis'
    )
    # infinite costs as x
    ax.scatter(
        [all_stages[i] for i in inf_indices],
        [all_states[i] for i in inf_indices],
        marker='x',
        color='red',
        label='Infinite cost'
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cost')

    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Cost Function for ' + CITY + ' with ' + STRUCTURE + ' structure')
def extract_policy(memo):
    policy = {(stage, state): memo[(stage, state)][0][0] for stage in range(24) for state in state_space}
    return policy
def plot_policy_lines(policy,next_state):
    fig, ax = plt.subplots()
    for state in state_space:
        states = [state]
        controls = []
        for stage in range(24):
            control = policy[(stage, states[-1])]
            states.append(next_state(states[-1], control))
            controls.append(control)
        ax.plot(states, label=str(state))
    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Policy Lines for ' + CITY + ' with ' + STRUCTURE + ' structure')
def plot_policy_states(policy,next_state):
    fig, ax = plt.subplots()
    states = state_space
    stages = np.arange(24)
    states_grid, stages_grid = np.meshgrid(states, stages, indexing='ij')
    next_states = np.array([[next_state(states_grid[i,j], policy[(stages_grid[i,j], states_grid[i,j])]) for j in range(len(stages))] for i in range(len(states))])
    dx = np.zeros_like(states_grid)+1
    dy = next_states - states_grid
    length = np.sqrt(dx**2 + dy**2)
    ax.quiver(stages_grid, states_grid, dx/length, dy/length, angles='xy', scale_units='xy', scale=1, color='black')
    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Policy States for ' + CITY + ' with ' + STRUCTURE + ' structure')