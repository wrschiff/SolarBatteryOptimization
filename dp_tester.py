from dynamics import *

import parameters
import matplotlib.pyplot as plt
def policy_control(stage, state, policy):
    controls = [control for control, _ in policy.items() if control[0] == stage]
    states = [state for _, state in controls]
    min_state = min(states, key=lambda x: abs(x - state))
    return policy[(stage, min_state)]

def test_policy(num_sim, length, policy, parameters):
    """
    Return a dictionary where with items
    tuple(state,stage) -> average cost
    """
    big_states = []
    big_costs = []
    for _ in range(num_sim):
        costs = [0]
        i_state = 0
        states = [i_state]
        iter_range = range(0, length)
        for i in iter_range:
            mod_i = i % 24
            # pick next stochastic variables
            irr, load = gen_irr_and_load(mod_i, parameters)
            solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY

            # pick control
            opt_control = policy_control(mod_i, i_state, policy)
            # update state
            cost = arbitrage_cost(mod_i, opt_control, load, solar, parameters)
            i_state = next_state(i_state, opt_control, parameters=parameters)
            states.append(i_state)
            costs.append(cost)

        big_states.append(states)
        big_costs.append(costs)
    return big_states, big_costs

