import numpy as np
import matplotlib.pyplot as plt
import pickle
from plotting import *
from dynamics import *
import parameters

def term_cost(state):
    return 0

memo = dict()
def solve(stage: int, state: float,parameters):
    if (stage, state) in memo:
        return memo[(stage, state)]
    else:
        result = _solve(stage, state,parameters=parameters)
        memo[(stage, state)] = result
        return result

def _solve(stage: int, state: float,parameters):
    if stage == parameters.MAX_STAGE:
        return [(), term_cost(state)]
    irr, load = get_expected_irr_and_load(stage, parameters)
    controls_to_costs = dict()
    for next in parameters.state_space:
        control = control_from_state(state,next,parameters=parameters) # compute control needed to get to given state
        if control is None: # control is inadmissible
            continue

        next_controls, next_cost = solve(stage+1, next,parameters)
        solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
        cost = carbon_arbitrage_cost(stage, control, load, solar, parameters) + next_cost
        controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])

if __name__ == "__main__":
    params = parameters.Parameters(N_BATT=5,N_SOLAR=20,CITY='Phoenix',STRUCTURE='B')  # Create a parameters instance
    start_states = np.linspace(0, params.N_BATT * params.BATT_CAP, params.N_STATE_DISC)
    costs = dict()
    memo.clear()
    for state in start_states:
        out = solve(stage=0, state=state, parameters=params)
        costs[state] = out[1]
        
    policy = extract_policy(memo,params)        
    plot_cost_function(memo,params)
    plot_policy_states(policy, next_state,params)
    plot_state_cost(0,memo,params)
    
    plt.show()
    # 
    # 
    # for state in start_states:
    #     plot_state_cost(state, memo, params)    


