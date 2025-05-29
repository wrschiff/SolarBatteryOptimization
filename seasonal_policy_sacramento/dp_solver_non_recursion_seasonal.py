import numpy as np
import matplotlib.pyplot as plt
import dynamics_seasonal_sacramento 
from parameters_seasonal import *
import plotting_seasonal

parameters = Parameters(N_SOLAR=20, MAX_STAGE=8760, N_BATT=5)
state_space = np.linspace(0, parameters.N_BATT * parameters.BATT_CAP, parameters.N_STATE_DISC)
cost_to_go = np.zeros((parameters.MAX_STAGE + 1, parameters.N_STATE_DISC))
policy = np.zeros((parameters.MAX_STAGE + 1, parameters.N_STATE_DISC))

def solve():
    for stage in range(parameters.MAX_STAGE, -1, -1):
        for state_index, state in enumerate(state_space):
            if stage == parameters.MAX_STAGE:
                cost_to_go[stage, state_index] = 0
                continue
            
            min_cost = float('inf')

            # Find the index of the largest element in state_space that is smaller than lower_bound_from_state
            lower_bound_from_state = state - parameters.N_BATT * 2 / parameters.ETA        
            lower_idx = np.searchsorted(state_space, lower_bound_from_state, side='right') - 1
            lower_idx = max(lower_idx, 0)
            
            for i in range(lower_idx, len(state_space)):
                control = dynamics_seasonal_sacramento.control_from_state(state, state_space[i], parameters)
                if control is None:
                    continue
                if state_space[i] > state + parameters.N_BATT*2:
                    break
                
                irr, load = dynamics_seasonal_sacramento.get_expected_irr_and_load(stage, parameters)
                solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
                
                next_index = state_space.searchsorted(state_space[i])
                cost = dynamics_seasonal_sacramento.arbitrage_cost(stage, control, load, solar, parameters) + cost_to_go[stage + 1, next_index]
                
                if cost < min_cost:
                    min_cost = cost
                    policy[stage, state_index] = control
            
            cost_to_go[stage, state_index] = min_cost
    
    return cost_to_go

if __name__ == "__main__":
    solve()
    memo = plotting_seasonal.from_arr_to_dict(policy, parameters)
    plotting_seasonal.plot_policy_states(memo, dynamics_seasonal_sacramento.next_state, parameters)
    plt.show()
