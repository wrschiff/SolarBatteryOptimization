import numpy as np
import matplotlib.pyplot as plt
import dynamics
from parameters import *

state_space = np.linspace(0, N_BATT * BATT_CAP, N_STATE_DISC)
cost_to_go = np.zeros((MAX_STAGE + 1, N_STATE_DISC))
policy = np.zeros((MAX_STAGE + 1, N_STATE_DISC))

def solve():
    for stage in range(MAX_STAGE, -1, -1):
        for state_index, state in enumerate(state_space):
            if stage == MAX_STAGE:
                cost_to_go[stage, state_index] = 0
                continue
            
            min_cost = float('inf')

            # Find the index of the largest element in state_space that is smaller than lower_bound_from_state
            lower_bound_from_state = state - N_BATT * 2 / ETA        
            lower_idx = np.searchsorted(state_space, lower_bound_from_state, side='right') - 1
            lower_idx = max(lower_idx, 0)
            
            for i in range(lower_idx, len(state_space)):
                control = dynamics.control_from_state(state, state_space[i])
                if control is None:
                    continue
                if state_space[i] > state + N_BATT*2:
                    break
                
                irr, load = dynamics.get_expected_irr_and_load(stage, CITY)
                solar = irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY
                
                next_index = state_space.searchsorted(state_space[i])
                cost = dynamics.arbitrage_cost(stage, control, load, solar) + cost_to_go[stage + 1, next_index]
                
                if cost < min_cost:
                    min_cost = cost
                    policy[stage, state_index] = control
            
            cost_to_go[stage, state_index] = min_cost
    
    return cost_to_go

if __name__ == "__main__":
    print(solve()[0, 0])
    print(policy[:,0])
