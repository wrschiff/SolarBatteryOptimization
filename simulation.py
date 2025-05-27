import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize_scalar
import NN_Linefit
import torch
import dynamics
from parameters import *

parameters = Parameters(MAX_STAGE=24*1)
BIN_SIZE = 0.1
NUM_BINS = int((parameters.N_BATT * parameters.BATT_CAP) // BIN_SIZE)

def dumb_cost(state):
    # change to a parabola with random x offset
    return 0

def simulate(stage, state, cost_func_arr, num_sim, pick_control=False, training=True):
    """
    Return a dictionary where with items
    tuple(state,stage) -> average cost
    """
    out_dicts = []
    for _ in range(num_sim):
        out_dict = {}
        costs = [0]
        total_cost = 0
        i_state = state
        states = [i_state]
        iter_range = range(stage, 24)
        for i in iter_range:
            # pick next stochastic variables
            irr,load = dynamics.get_expected_irr_and_load(i, parameters)
            solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY # irr = 0.6

            # pick control
            if pick_control:
                sol = minimize_scalar(lambda control: dynamics.arbitrage_cost(i, control, load, solar, parameters) +
                                    cost_func_arr[i+1](dynamics.next_state(i_state, control, parameters)),
                                    bounds = [max(-parameters.N_BATT*2,-state),min(2*parameters.N_BATT,5*parameters.N_BATT-i_state)])
                if training:
                    epsilon = max(0.05, 0.2 * (0.8 ** i))

                    lb = max(-parameters.N_BATT*2, -state)
                    ub = min(2*parameters.N_BATT, 5*parameters.N_BATT - i_state)

                    if np.random.rand() < epsilon:
                        opt_control = np.random.uniform(lb, ub)
                    else:
                        opt_control = sol.x
                else:
                    opt_control = sol.x
            else:
                opt_control = np.random.uniform(max(-parameters.N_BATT*2,-i_state),min(2*parameters.N_BATT,5*parameters.N_BATT-i_state))
            
            irr,load = dynamics.gen_irr_and_load(i, parameters)
            solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY
            # update state
            cost = dynamics.arbitrage_cost(i, opt_control, load, solar, parameters)
            i_state = dynamics.next_state(i_state, opt_control, parameters)
            states.append(i_state)
            costs.append(cost)
        cum_costs = np.cumsum(costs[::-1])[::-1]
        if not training and stage == 0:
            total_cost = sum(costs)
            print(f"Total cost for simulation: {total_cost:.2f}")
        tups = [(states[i],i) for i in iter_range]
        out_dict.update(dict(zip(tups,cum_costs)))
        out_dicts.append(out_dict)
    total_keys = [(state,stage) for state in np.linspace(0,parameters.N_BATT*5,NUM_BINS) for stage in range(24)]
    result_dict = {key: [] for key in total_keys}

    for out_dict in out_dicts:
        for key in out_dict.keys():
            # Find the appropriate bin
            bin_key = min([k for k in total_keys if k[1] == key[1] and k[0] >= key[0]])
            result_dict[bin_key].append(out_dict[key])
    result_dict = {k: np.mean(v) for k,v in result_dict.items() if len(v) > 0}
    return result_dict

if __name__ == "__main__":
    data = simulate(0, 0, [dumb_cost for i in range(parameters.MAX_STAGE)], 5000)

    models = NN_Linefit.backward_pass(data)
    for i in range(2):
        models_prev = models.copy()
        data = simulate(0, 0, models, 50, pick_control=True)
        models = NN_Linefit.backward_pass(data)
        # find the minimum point and generate a new model
        for j in range(parameters.MAX_STAGE):
            x_s = np.linspace(0, parameters.N_BATT * parameters.BATT_CAP, NUM_BINS)
            y_s = [models[j](x) for x in x_s]
            y_s_prev = [models_prev[j](x) for x in x_s]
            for k in range(len(x_s)):
                if y_s[k] < y_s_prev[k]:
                    y_s_prev[k] = y_s[k]
            models[j].fit(torch.tensor(x_s, dtype=torch.float32).view(-1, 1),
                          torch.tensor(y_s_prev, dtype=torch.float32).view(-1,1), epochs=60)
        
    data = simulate(0, 0, models, 1, pick_control=True, training=False)
    # for stage in set([key[1] for key in data.keys()]):
    #     x_values = [key[0] for key in data.keys() if key[1] == stage]
    #     y_values = [data[key] for key in data.keys() if key[1] == stage]
    #     color = "#%06x" % random.randint(0, 0xFFFFFF)
    #     plt.figure()
    #     plt.scatter(x_values, y_values, label=str(stage), color=color)
    #     plt.title("Stage " + str(stage))
    #     plt.ylabel("Cost")
    #     plt.xlabel("Energy Level")
    # plt.show()
