import numpy as np
import matplotlib.pyplot as plt
import pickle
from plotting import *
from dynamics import *
from parameters import Parameters

def term_cost(state, parameters: Parameters):
    return 0

memo = dict()
def solve(stage: int, state: float, parameters: Parameters):
    if (stage, state) in memo:
        return memo[(stage, state)]
    else:
        result = _solve(stage, state, parameters)
        memo[(stage, state)] = result
        return result

def _solve(stage: int, state: float, parameters: Parameters):
    if stage == parameters.MAX_STAGE:
        return [(), term_cost(state, parameters)]
    irr, load = get_expected_irr_and_load(stage, parameters)
    controls_to_costs = dict()
    for next in parameters.state_space:
        control = control_from_state(state,next,parameters) # compute control needed to get to given state
        if control is None: # control is inadmissible
            continue
        irr_range, _ = get_irr_and_load_range(stage, parameters)
        min_sol = irr_range[0] * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
        max_sol = irr_range[1] * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
        if control > min_sol: # state transition is non deterministic
            worst_next = next_state(state, min_sol, parameters)
            best_next = next_state(state, max_sol, parameters)

            # all reachable next states (irrespective of chosen control)
            all_next = [state for state in parameters.state_space if worst_next <= state <= best_next]
            n_real = len(all_next)
            if n_real == 0:
                all_next = [max(state for state in parameters.state_space if state < worst_next)]
                n_real = 1
            # all reachable next states with given control
            possible_next = [state for state in parameters.state_space if worst_next <= state <= min(next,best_next)]
            possible_next.extend([next]*(n_real-len(possible_next)))

            cost_sum = 0
            for p_next in possible_next:
                next_controls, next_cost = solve(stage+1, p_next, parameters)
                cost = arbitrage_cost(stage, control, load, irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY, parameters) + next_cost
                cost_sum += cost
            
            controls_to_costs[(control,) + next_controls] = cost_sum / n_real
        else:        
            next_controls, next_cost = solve(stage+1, next, parameters)
            cost = arbitrage_cost(stage, control, load, irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY, parameters) + next_cost
            controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])

if __name__ == "__main__":
    parameters = Parameters(N_SOLAR=2, STRUCTURE='B', MAX_STAGE=24*1)
    start_states = np.linspace(0, parameters.N_BATT*parameters.BATT_CAP, parameters.N_STATE_DISC)
    raws = []
    costs = dict()
    memo.clear()
    for state in start_states:
        out = solve(stage=0, state=state, parameters=parameters)
        costs[state] = out[1]
    plot_cost_function(memo, parameters)
    policy = extract_policy(memo, parameters)
    plot_policy_states(policy, next_state, parameters)
    filename = parameters.pickle_file_name()
    with open(filename, 'wb') as f:
        pickle.dump(policy, f)
    
    print(policy[(0,0)])
    plt.show()

