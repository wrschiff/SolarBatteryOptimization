import matplotlib.pyplot as plt
import numpy as np
from parameters_seasonal import Parameters

def plot_state_cost(state, memo, parameters: Parameters):
    fig, ax = plt.subplots()
    stages = [a[0] for a in memo.keys() if a[1] == state]
    costs = [memo[(stage, 0)][1] for stage in stages]
    ax.plot(stages, costs, marker='o')
    ax.set_xlabel('State')
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost Function for starting at stage {state}, ' + parameters.CITY + 'seasonal')
def plot_cost_function(memo, parameters: Parameters):
    # derive cost function
    fig, ax = plt.subplots()
    all_stages = []
    all_states = []
    all_costs = []
    for stage in range(parameters.MAX_STAGE + 1):
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
    ax.set_title('Cost Function for ' + parameters.CITY + 'seasonal')
def extract_policy(memo, parameters: Parameters):
    policy = {(stage, state): memo[(stage, state)][0][0] for stage in range(8760) for state in parameters.state_space}
    return policy
def get_day_cost(memo, parameters: Parameters):
    return memo[(0, 0)][1]/(parameters.MAX_STAGE/24) # divide by number of days
def plot_policy_lines(policy, next_state, parameters: Parameters):
    fig, ax = plt.subplots()
    for state in [p[1] for p in policy.keys() if p[0] == 0]:
        states = [state]
        controls = []
        for stage in range(8760):
            control = policy[(stage, states[-1])]
            states.append(next_state(states[-1], control, parameters))
            controls.append(control)
        ax.plot(states, label=str(state))
    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Policy Lines for ' + parameters.CITY + ' seasonal')
def plot_policy_states(policy, next_state, parameters: Parameters):
    fig, ax = plt.subplots()
    states = [state[1] for state in policy if state[0] == 0]
    stages = np.arange(8760)
    states_grid, stages_grid = np.meshgrid(states, stages, indexing='ij')
    next_states = np.array([[next_state(states_grid[i, j], policy[(stages_grid[i, j], states_grid[i, j])], parameters) for j in range(len(stages))] for i in range(len(states))])
    dx = np.zeros_like(states_grid)+1
    dy = next_states - states_grid
    length = np.sqrt(dx**2 + dy**2)
    ax.quiver(stages_grid, states_grid, dx/length, dy/length, angles='xy', scale_units='xy', scale=1, color='black')
    ax.plot([0, 0], [max(states)+0.5, 25], color='red', label='Unreachable states')
    ax.legend()
    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Policy States for ' + parameters.CITY + ' seasonal')

def plot_policy_boxes(policy, parameters: Parameters):
    pmin_states = []
    pmax_states = []
    nmax_states = []
    nmin_states = []
    p_stages = []
    n_stages = []
    for stage in range(parameters.MAX_STAGE):
        pos_states = [state for (s, state), control in policy.items() if s == stage and control > 0]
        neg_states = [state for (s, state), control in policy.items() if s == stage and control < 0]
        if pos_states:
            min_state = min(pos_states)
            max_state = max(pos_states)
            if max_state-min_state < 1:  # if the range is too small, skip
              pmin_states.append(0)
              pmax_states.append(0)
            else:
                pmin_states.append(min_state)
                pmax_states.append(max_state)
            p_stages.append(stage)
            nmax_states.append(0)
            nmin_states.append(0)
            n_stages.append(stage)

        elif neg_states:
            min_state = min(neg_states)
            max_state = max(neg_states)
            if max_state-min_state < 1:  # if the range is too small, skip
              nmin_states.append(0)
              nmax_states.append(0)
            else:  
                nmin_states.append(min_state)
                nmax_states.append(max_state)
            n_stages.append(stage)
            p_stages.append(stage)
            pmin_states.append(0)
            pmax_states.append(0)
    pheights = [pmax_states[i] - pmin_states[i] for i in range(len(p_stages))]
    nheights = [nmax_states[i] - nmin_states[i] for i in range(len(n_stages))]
    
    fig, ax = plt.subplots()
    ax.bar(p_stages, pheights, width=0.8, align='center', color='green', edgecolor='black', bottom=pmin_states, label='Charging')
    ax.bar(n_stages, nheights, width=0.8, align='center', color='red', edgecolor='black', bottom=nmin_states, label='Discharging')
    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Policy Thresholds for ' + parameters.CITY + ' seasonal')
    ax.legend(loc='upper left')

def plot_tester_states(states):
    plt.figure()
    for state_list in states:
        plt.plot(state_list)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('States Over Time')

def plot_tester_costs(costs):
    plt.figure()
    for cost_list in costs:
        plt.plot(cost_list)
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.title('Costs Over Time')

def plot_tester_cum_costs(costs):
    plt.figure()
    cs = np.cumsum(costs,axis=1)
    for i,cost_list in enumerate(cs):
        plt.plot(cost_list, label='Simulation ' + str(i+1))
    plt.xlabel('Days')
    plt.xticks(np.linspace(0, len(costs[0]), 13), [str(int(x//24)) for x in np.linspace(0, len(costs[0]), 13)])
    plt.ylabel('Cost')
    plt.title('Cumulative Costs Over Time')
    avg_cum_cost = np.mean(cs[:,-1])
    plt.text(0.01, max(cs[:,-1])/2, 
             f'Avg. tot.: {avg_cum_cost:.2f}\nAvg. per day: {avg_cum_cost/(len(costs[0])/24):.2f}', 
             ha='left', va='top')
    plt.legend()

def from_arr_to_dict(arr, parameters: Parameters):
    return {(i, parameters.state_space[j]): arr[i, j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}
