import numpy as np
import matplotlib.pyplot as plt
import pickle

N_BATT = 5
BATT_CAP = 5
N_SOLAR = 5
AREA_SOLAR = 2
ETA = 0.949
SOL_EFFICIENCY = 0.15*0.82
STRUCTURE = 'B'
CITY = 'Seattle'
MAX_STAGE = 24 * 14
N_STATE_DISC = 20

#EPOCH_COUNT = 1 # times to use starting costs as terminal costs

state_space = np.linspace(0,N_BATT*BATT_CAP,N_STATE_DISC)

term_points = {0: 0, N_BATT*BATT_CAP: 0}
def term_cost(state):
    lower_state = max([k for k in term_points.keys() if k <= state])
    upper_state = min([k for k in term_points.keys() if k >= state])
    if lower_state == upper_state:
        return term_points[lower_state]
    else:
        lower_cost = term_points[lower_state]
        upper_cost = term_points[upper_state]
        return lower_cost + (state - lower_state) / (upper_state - lower_state) * (upper_cost - lower_cost)

memo = dict()
def solve(stage: int, state: float):
    if (stage, state) in memo:
        return memo[(stage, state)]
    else:
        result = _solve(stage, state)
        memo[(stage, state)] = result
        return result

def _solve(stage: int, state: float):
    if stage == MAX_STAGE:
        return [(), term_cost(state)]
    irr, load = get_irr_and_load(stage, CITY)
    controls_to_costs = dict()
    for next in state_space:
        control = control_from_state(state,next) # compute control needed to get to given state
        if control is None: # control is inadmissible
            continue
        next_controls, next_cost = solve(stage+1, next)
        cost = arbitrage_cost(stage, control, load, irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY) + next_cost
        controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])
def get_irr_and_load(stage, city):
    stage = stage % 24
    if city == "Phoenix":
        minVars = [0.8, 0.9, 0.85, 0.8]
        maxVars = [1.2, 1.1, 1.15, 1.2]
        means = [0, 0, 0, 0, 0, 0.032, 0.178, 0.410, 0.632,
            0.812, 0.942, 1.016, 1.028, 0.974, 0.862, 0.698, 0.498, 0.285, 0.124,
            0.018, 0, 0, 0, 0]
    elif city == "Sacramento":
        minVars = [0.7, 0.8, 0.75, 0.7]
        maxVars = [1.3, 1.2, 1.25, 1.3]
        means = [0, 0, 0, 0, 0, 0.015, 0.142, 0.356, 0.556,
                 0.712, 0.825, 0.891, 0.902, 0.855, 0.756, 0.612, 0.436, 0.245, 0.098,
                 0.008, 0, 0, 0, 0]
    elif city == "Seattle":
        minVars = [0.6, 0.65, 0.6, 0.55]
        maxVars = [1.40, 1.35, 1.40, 1.45]
        means = [0, 0, 0, 0, 0, 0, 0.072, 0.224, 0.367, 0.498,
                 0.594, 0.654, 0.676, 0.644, 0.562, 0.442, 0.302, 0.158, 0.054, 0,
                 0, 0, 0, 0]
    else:
        raise ValueError("City not recognized. Please use 'Phoenix', 'Sacramento', or 'Seattle'.")
        
    consump = [0.52, 0.42, 0.38, 0.35, 0.32, 0.38, 0.62, 0.98, 0.85, 0.68, 0.62, 0.65,
        0.75, 0.68, 0.65, 0.72, 0.95, 1.42, 1.95, 1.65, 1.38, 1.15, 0.88, 0.65]
    consumpVarMin = 0.8
    consumpVarMax = 1.2
    
    zone = [stage < 8, stage < 12, stage < 26, 1].index(1)

    irr = means[stage]
    load = consump[stage]
    return irr, load
    
def next_state(state: float, control):
    return state + control * (1/ETA if control < 0 else ETA)
def control_from_state(current:float, next: float):
    needed = (next - current) * (ETA if current > next else 1/ETA)
    if needed < max(-N_BATT*2,-current) or needed > min(2*N_BATT,5*N_BATT-current):
        return None
    return needed
def arbitrage_cost(stage, control, load, solar):
    stage = stage % 24
    p_grid = load - solar + control
    [buy, sell] = buy_sell_rates(stage, STRUCTURE)
    rate = buy if p_grid > 0 else sell
    return p_grid * rate

def buy_sell_rates(stage, structure):
    if structure == 'A':
        return 0.15, 0.15
    if structure == 'B':
        arr = [[0.1,0.1],[0.15,0.15],[0.3,0.3],[0.1,0.1]]
    else:
        arr = [[0.12,0.06],[0.18,0.09],[0.35,0.07],[0.12,0.06]]
    
    zone = [stage < 8, stage < 16, stage < 20, 1]
    return arr[zone.index(1)]

def plot_state_cost(state):
    fig, ax = plt.subplots()
    stages = [a[0] for a in memo.keys() if a[1] == state]
    costs = [memo[(stage, 0)][1] for stage in stages]
    ax.plot(stages, costs, marker='o')
    ax.set_xlabel('State')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Function for Stage 1')
def plot_cost_function():
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
    # Plot all at once
    scatter = ax.scatter(all_stages, all_states, c=all_costs, cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cost')

    ax.set_xlabel('Stage')
    ax.set_ylabel('State')
    ax.set_title('Cost Function for ' + CITY + ' with ' + STRUCTURE + ' structure')
def extract_policy(memo):
    policy = {(stage, state): memo[(stage, state)][0][0] for stage in range(24) for state in state_space}
    return policy
def plot_policy_lines(policy):
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

if __name__ == "__main__":
    term_states = np.linspace(0, N_BATT*BATT_CAP, N_STATE_DISC)
    raws = []
    costs = dict()
    memo.clear()
    for state in term_states:
        out = solve(stage=0, state=state)
        costs[state] = out[1]
    plot_cost_function()
    policy = extract_policy(memo)
    plot_policy_lines(policy)
    filename = CITY + '_' + STRUCTURE + '_' + str(N_BATT) + '_' + str(N_SOLAR) + '_policy.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(policy, f)
    plt.show()

