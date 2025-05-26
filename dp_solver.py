import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable

N_BATT = 5
BATT_CAP = 5
N_SOLAR = 10
AREA_SOLAR = 2
ETA = 0.949
SOL_EFFICIENCY = 0.15*0.82
STRUCTURE = 'A'
CITY = 'Sacramento'

N_CONT_DISC = 10
control_space = np.linspace(-N_BATT*2,2*N_BATT,N_CONT_DISC)
if 0 not in control_space:
    control_space = np.insert(control_space, 0, 0)
print(control_space)
def term_cost(state):
    return 0

memo = dict()
def solve(stage: int, state: float):
    if (stage, state) in memo:
        return memo[(stage, state)]
    else:
        result = _solve(stage, state)
        memo[(stage, state)] = result
        return result

def _solve(stage: int, state: float):
    if stage == 24:
        return [(), term_cost(state)]
    if stage == 1:
        print("I am at stage 1")
    all_controls = [c for c in control_space if -state <= c <= 5*N_BATT - state]
    irr, load = get_irr_and_load(stage, CITY)
    controls_to_costs = dict()
    for control in all_controls:
        next = next_state(stage, state, control)
        next_controls, next_cost = solve(stage+1, next)
        cost = arbitrage_cost(stage, control, load, irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY) + next_cost
        controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])
def get_irr_and_load(stage, city):
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
    
def next_state(stage: int, state: float, control):
    return state + control * (1/ETA if control < 0 else ETA)

def arbitrage_cost(stage, control, load, solar):
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

if __name__ == "__main__":
    print(solve(stage=0, state=0))
    