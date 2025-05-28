import numpy as np
import matplotlib.pyplot as plt
import dynamics
from parameters import *
from scipy.integrate import quad
parameters = Parameters(N_SOLAR=20, STRUCTURE='C', MAX_STAGE=24*7, CITY='Phoenix')

state_space = parameters.state_space
cost_to_go = np.zeros((parameters.MAX_STAGE + 1, parameters.N_STATE_DISC))
policy = np.zeros((parameters.MAX_STAGE + 1, parameters.N_STATE_DISC))

def solve():
    for stage in range(parameters.MAX_STAGE, -1, -1):
        for state_index, state in enumerate(state_space):
            if stage == parameters.MAX_STAGE:
                cost_to_go[stage, state_index] = 0
                continue

            # Find the index of the largest element in state_space that is smaller than lower_bound_from_state
            lower_bound_from_state = state - parameters.N_BATT * 2 / parameters.ETA        
            lower_idx = np.searchsorted(state_space, lower_bound_from_state, side='right') - 1
            lower_idx = max(lower_idx, 0)
            costs = []
            controls = []
            for i in range(lower_idx, len(state_space)):
                control = dynamics.control_from_state(state, state_space[i], parameters)

                if control is None:
                    continue
                if state_space[i] > state + parameters.N_BATT*2:
                    break
                
                irr_mean, load_mean = dynamics.get_expected_irr_and_load(stage, parameters)
                # solar = irr_mean * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
                
                irr_range, _ = dynamics.get_irr_and_load_range(stage, parameters)
                irr_range[0] = irr_range[0] * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
                irr_range[1] = irr_range[1] * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY

                next_index = state_space.searchsorted(state_space[i])
                if irr_range[0] == irr_range[1]:
                    integral = arbitrage_cost(stage, control, load_mean, irr_range[0], parameters) \
                        + cost_to_go[stage+1, next_state_index_with_solar(state, control, irr_range[0], next_index, parameters)]
                else:
                    prob = 1 / (irr_range[1] - irr_range[0])
                    integral = quad(lambda solar: prob*arbitrage_cost(stage, control, load_mean, solar, parameters) \
                        + prob*cost_to_go[stage+1, next_state_index_with_solar(state, control, solar, next_index, parameters)], 
                            irr_range[0], irr_range[1])[0]
                
                cost = integral
                costs.append(cost)
                controls.append(control)
            cost_to_go[stage, state_index] = min(costs)
            policy[stage, state_index] = controls[np.argmin(costs)] if costs else 0
    
    return cost_to_go

def next_state_index_with_solar(state, control, solar, nxt_state_index, parameters):
    if control > 0:
        if control > solar:
            control = solar
        nxt_state = state + control * parameters.ETA
        lower_bound = state_space.searchsorted(nxt_state)
        if abs(state_space[lower_bound]-nxt_state) < 1e-8:
            return lower_bound
        else:
            return max(lower_bound-1, 0)
    else:
        return nxt_state_index

def arbitrage_cost(stage, control, load, solar, parameters):
    if control > solar:
        control = solar
    return dynamics.arbitrage_cost(stage, control, load, solar, parameters)
    
if __name__ == "__main__":
    solve()
    print(cost_to_go[0, 25])
    plt.imshow(policy.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Control')
    plt.show()
