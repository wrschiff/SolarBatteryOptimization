import numpy as np
from scipy.stats import uniform
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
    PROB_FAIL = parameters.PROB_FAIL
    if stage == parameters.MAX_STAGE:
        return [(), term_cost(state, parameters)]
    #Stuff for distribution
    Plmean =get_load_means(stage, parameters)
    dist = uniform(loc=0.8, scale=1.2 - 0.8)
    scaled_low = Plmean * 0.8
    scaled_high = Plmean * 1.2
    lo,hi = min(scaled_low, scaled_high), max(scaled_low, scaled_high)
    dist = uniform(loc=lo, scale=hi - lo)
    # End of distribution stuff
    irr, load = get_expected_irr_and_load(stage, parameters)
    controls_to_costs = dict()
    for next in parameters.state_space:
        if lo < next < hi:  # if next state is in the range of the distribution
            jp = (1-dist.cdf(next))*PROB_FAIL #joint prob threshold
            if jp > 0.05: #prob of having less than load and prob of grid fail
                #continue
                next_controls, next_cost = solve(stage + 1, next, parameters)
                cost = float('inf') + next_cost
                controls_to_costs[(control,) + next_controls] = cost
            else:
                control = control_from_state(state, next, parameters)
                if control is None:
                    continue
                next_controls, next_cost = solve(stage + 1, next, parameters)
                cost = arbitrage_cost(stage, control, load, irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY, parameters) + next_cost
                controls_to_costs[(control,) + next_controls] = cost          
        else:
            control = control_from_state(state, next, parameters)
            if control is None:
                continue
            next_controls, next_cost = solve(stage + 1, next, parameters)
            cost = arbitrage_cost(stage, control, load, irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY, parameters) + next_cost
            controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])

        
if __name__ == "__main__":
    params = Parameters(N_BATT=5,N_SOLAR=20,CITY='Seattle',STRUCTURE='B',MAX_STAGE=24)  # Create a parameters instance
    start_states = np.linspace(0, params.N_BATT * params.BATT_CAP, params.N_STATE_DISC)
    costs = dict()
    memo.clear()
    for state in start_states:
        out = solve(stage=0, state=state, parameters=params)
        costs[state] = out[1]
    
    policy = extract_policy(memo,params)
    print('Min ctg for state 0:', costs[0])        
    plot_cost_function(memo,params)
    plot_policy_states(policy, next_state,params)
    plot_state_cost(0,memo,params)
    plt.show()

    # If want to ignore state
    # cost_matrix = np.zeros((params.MAX_STAGE + 1, len(start_states)))

    # for stage in range(params.MAX_STAGE + 1):
    #     for i, state in enumerate(start_states):
    #         out = solve(stage=stage, state=state, parameters=params)
    #         cost_matrix[stage, i] = out[1]

    # plt.figure(figsize=(10, 6))
    # plt.imshow(cost_matrix, aspect='auto', origin='lower',
    #            extent=[0, params.MAX_STAGE, start_states[0], start_states[-1]])
    # plt.colorbar(label='Cost')
    # plt.xlabel('Stage')
    # plt.ylabel('State')
    # plt.title('Cost Colormap: Stage vs State')
    # plt.show()