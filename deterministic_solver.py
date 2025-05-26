import numpy as np
import matplotlib.pyplot as plt
import pickle
from plotting import *
from dynamics import *
from parameters import *

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
    if stage == MAX_STAGE:
        return [(), term_cost(state)]
    irr, load = get_expected_irr_and_load(stage, CITY)
    controls_to_costs = dict()
    for next in state_space:
        control = control_from_state(state,next) # compute control needed to get to given state
        if control is None: # control is inadmissible
            continue

        next_controls, next_cost = solve(stage+1, next)
        cost = arbitrage_cost(stage, control, load, irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY) + next_cost
        controls_to_costs[(control,) + next_controls] = cost
    return min(controls_to_costs.items(), key=lambda x: x[1])

if __name__ == "__main__":
    start_states = np.linspace(0, N_BATT*BATT_CAP, N_STATE_DISC)
    raws = []
    costs = dict()
    memo.clear()
    for state in start_states:
        out = solve(stage=0, state=state)
        costs[state] = out[1]
    plot_cost_function(memo)
    policy = extract_policy(memo)
    plot_policy_states(policy,next_state)
    filename = 'policy/' + CITY + '_' + STRUCTURE + '_' + str(N_BATT) + '_' + str(N_SOLAR) +  '_policy.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(policy, f)
    plt.show()

