import numpy as np
import matplotlib.pyplot as plt

N_BATT = 5
BATT_CAP = 5
N_SOLAR = 5
AREA_SOLAR = 2
ETA = 0.949
SOL_EFFICIENCY = 0.15*0.82
STRUCTURE = 'A'
CITY = 'Seattle'
MAX_STAGE = 24 * 7
N_STATE_DISC = 500

state_space = np.linspace(0, N_BATT * BATT_CAP, N_STATE_DISC)
cost_to_go = np.zeros((MAX_STAGE + 1, N_STATE_DISC))
policy = np.zeros((MAX_STAGE + 1, N_STATE_DISC))

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

def buy_sell_rates(stage, structure):
    if structure == 'A':
        return 0.15, 0.15
    if structure == 'B':
        arr = [[0.1,0.1],[0.15,0.15],[0.3,0.3],[0.1,0.1]]
    else:
        arr = [[0.12,0.06],[0.18,0.09],[0.35,0.07],[0.12,0.06]]
    
    zone = [stage < 8, stage < 16, stage < 20, 1]
    return arr[zone.index(1)]

def arbitrage_cost(stage, control, load, solar):
    stage = stage % 24
    p_grid = load - solar + control
    [buy, sell] = buy_sell_rates(stage, STRUCTURE)
    rate = buy if p_grid > 0 else sell
    return p_grid * rate

def next_state(state: float, control, irr, load):
    solar = irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY
    return state + control * (1/ETA if control < 0 else ETA)
    
def control_from_state(current:float, next: float):
    needed = (next - current) * (ETA if current > next else 1/ETA)
    if needed < max(-N_BATT*2,-current) or needed > min(2*N_BATT,5*N_BATT-current):
        return None
    return needed

def solve():
    for stage in range(MAX_STAGE, -1, -1):
        for state_index, state in enumerate(state_space):
            if stage == MAX_STAGE:
                cost_to_go[stage, state_index] = 0
                continue
            
            min_cost = float('inf')

            # Find the index of the largest element in state_space that is smaller than lower_bound_from_state
            lower_bound_from_state = state - N_BATT * 2          
            lower_idx = np.searchsorted(state_space, lower_bound_from_state, side='right') - 1
            lower_idx = max(lower_idx, 0)
            
            for i in range(lower_idx, len(state_space)):
                control = control_from_state(state, state_space[i])
                if control is None:
                    continue
                if state_space[i] > state + N_BATT*2:
                    break
                
                irr, load = get_irr_and_load(stage, CITY)
                solar = irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY
                if control > solar:
                    continue
                next_st = next_state(state, control, irr, load)
                
                if next_st is None or next_st < 0 or next_st > N_BATT * BATT_CAP:
                    continue
                
                next_index = state_space.searchsorted(state_space[i])
                cost = arbitrage_cost(stage, control, load, irr) + cost_to_go[stage + 1, next_index]
                
                if cost < min_cost:
                    min_cost = cost
                    policy[stage, state_index] = control
            
            cost_to_go[stage, state_index] = min_cost
    
    return cost_to_go

if __name__ == "__main__":
    print(solve()[0, 0])
    print(policy[:,0])
