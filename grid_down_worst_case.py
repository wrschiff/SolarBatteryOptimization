import numpy as np
import matplotlib.pyplot as plt
import dynamics
from parameters import *
import plotting
import pickle
from testpolicyworstcase import *

parameters = Parameters(N_SOLAR=5,STRUCTURE='A', MAX_STAGE=24)
cities = ['Phoenix', 'Sacramento', 'Seattle']
structures = ['A', 'B', 'C']
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
                control = dynamics.control_from_state(state, state_space[i], parameters)
                if state_space[i] < dynamics.get_grid_down_energy_threshold(stage+1, parameters):
                    continue
                if control is None:
                    continue
                if state_space[i] > state + parameters.N_BATT*2:
                    break
                
                irr, load = dynamics.get_expected_irr_and_load(stage, parameters)
                solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
                
                next_index = state_space.searchsorted(state_space[i])
                cost = dynamics.arbitrage_cost(stage, control, load, solar, parameters) + cost_to_go[stage + 1, next_index]
                
                if cost < min_cost:
                    min_cost = cost
                    policy[stage, state_index] = control
            
            if min_cost == float('inf'):
                print(f"Warning: No valid control found for stage {stage}, state {state}.")
            cost_to_go[stage, state_index] = min_cost
    
    return cost_to_go

if __name__ == "__main__":
    for city in cities:
        parameters.CITY = city
        for structure in structures:
            parameters.STRUCTURE = structure
            #print(f"Solving for {city} with structure {structure}...")
            #cost_to_go = solve()
            #memo = plotting.from_arr_to_dict(policy, parameters)
            policy_file = parameters.pickle_file_name_grid_down()
            # with open(policy_file, 'wb') as f:
            #     pickle.dump(memo, f)
            # print(f"Policy saved to {policy_file}")
            with open(policy_file, 'rb') as f:
                memo = pickle.load(f)
            plot_policy_boxes(memo, parameters)
            #plot_policy_states(memo, next_state, params)
            plt.savefig(f"figures/policy_thresholds/{city}_{parameters.STRUCTURE}_{parameters.N_BATT}_{parameters.N_SOLAR}threshold_GRIDDOWN.png")    
            