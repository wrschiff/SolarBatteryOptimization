import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize_scalar

N_BATT = 5
BATT_CAP = 5
N_SOLAR = 10
AREA_SOLAR = 2
ETA = 0.949
SOL_EFFICIENCY = 0.15*0.82
STRUCTURE = 'A'
CITY = 'Seattle'
BIN_SIZE = 0.1
NUM_BINS = int((N_BATT * BATT_CAP) // BIN_SIZE)

def dumb_cost(state):
    # change to a parabola with random x offset
    return 0

def simulate(stage, state, cost_func, num_sim):
    """
    Return a dictionary where with items
    tuple(state,stage) -> average cost
    """
    out_dicts = []
    for _ in range(num_sim):
        out_dict = {}
        costs = [0]
        i_state = state
        states = [i_state]
        iter_range = range(stage, 24)
        for i in iter_range:
            # pick next stochastic variables
            irr,load = gen_irr_and_load(i, CITY)
            solar = irr * N_SOLAR * AREA_SOLAR * SOL_EFFICIENCY

            # pick control
            # sol = minimize_scalar(lambda control: arbitrage_cost(i, i_state, control, load, solar) +
            #                      cost_func(next_state(i, i_state, control, irr, load)),
            #                      bounds = [max(-N_BATT*2,-state),min(2*N_BATT,5*N_BATT-i_state)])
            # opt_control = sol.x
            opt_control = np.random.uniform(max(-N_BATT*2,-i_state),min(2*N_BATT,5*N_BATT-i_state))
            # update state
            cost = arbitrage_cost(i, opt_control, load, solar)
            i_state = next_state(i, i_state, opt_control, irr, load)
            states.append(i_state)
            costs.append(cost)
        cum_costs = np.cumsum(costs[::-1])[::-1]
        tups = [(states[i],i) for i in iter_range]
        out_dict.update(dict(zip(tups,cum_costs)))

        out_dicts.append(out_dict)
    total_keys = [(state,stage) for state in np.linspace(0,N_BATT*5,NUM_BINS) for stage in range(24)]
    result_dict = {key: [] for key in total_keys}

    for out_dict in out_dicts:
        for key in out_dict.keys():
            # Find the appropriate bin
            bin_key = min([k for k in total_keys if k[1] == key[1] and k[0] >= key[0]])
            result_dict[bin_key].append(out_dict[key])
    result_dict = {k: np.mean(v) for k,v in result_dict.items() if len(v) > 0}
    return result_dict

def gen_irr_and_load(stage, city):
    if city == "Phoenix":
        minVars = [0.8, 0.9, 0.85, 0.8]
        maxVars = [1.2, 1.1, 1.15, 1.2]
        means = [0, 0, 0, 0, 0.032, 0.178, 0.410, 0.632,
            0.812, 0.942, 1.016, 1.028, 0.974, 0.862, 0.698, 0.498, 0.285, 0.124,
            0.018, 0, 0, 0, 0]
    elif city == "Sacramento":
        minVars = [0.7, 0.8, 0.75, 0.7]
        maxVars = [1.3, 1.2, 1.25, 1.3]
        means = [0, 0, 0, 0, 0, 0.015, 0.142, 0.356, 0.556,
                 0.712, 0.825, 0.891, 0.902, 0.855, 0.756, 0.612, 0.436, 0.245, 0.098,
                 0.008, 0, 0, 0, 0]
    else:
        minVars = [0.6, 0.65, 0.6, 0.55]
        maxVars = [1.40, 1.35, 1.40, 1.45]
        means = [0, 0, 0, 0, 0, 0, 0.072, 0.224, 0.367, 0.498,
                 0.594, 0.654, 0.676, 0.644, 0.562, 0.442, 0.302, 0.158, 0.054, 0,
                 0, 0, 0, 0]
        
    consump = [0.52, 0.42, 0.38, 0.35, 0.32, 0.38, 0.62, 0.98, 0.85, 0.68, 0.62, 0.65,
        0.75, 0.68, 0.65, 0.72, 0.95, 1.42, 1.95, 1.65, 1.38, 1.15, 0.88, 0.65]
    consumpVarMin = 0.8
    consumpVarMax = 1.2
    
    zone = [stage < 8, stage < 12, stage < 26, 1].index(1)

    irr = np.random.uniform(minVars[zone], maxVars[zone]) * means[stage]
    load = np.random.uniform(consumpVarMin, consumpVarMax) * consump[stage]
    return irr, load
    
def next_state(stage, state, control, irr, load):
    return state + control * (1/ETA if control < 0 else ETA)

def arbitrage_cost(stage, control, load, solar):
    p_grid = load - solar + control
    [buy, sell] = buy_sell_rates(stage, STRUCTURE)
    rate = buy if p_grid > 0 else sell
    return p_grid * rate

def buy_sell_rates(stage, structure):
    if structure == 'A':
        return 0.15, 0.15
    if structure == 'B':
        arr = [[0.1,0.1],[0.15,0.15],[0.3,0.3],[0.1,0.1]]
    else:
        arr = [[0.12,0.06],[0.18,0.09],[0.35,0.07],[0.12,0.06]]
    
    zone = [stage < 8, stage < 16, stage < 20, 1]
    return arr[zone.index(1)]

if __name__ == "__main__":
    data = simulate(0, 0, dumb_cost, 1000)

    for stage in set([key[1] for key in data.keys()]):
        x_values = [key[0] for key in data.keys() if key[1] == stage]
        y_values = [data[key] for key in data.keys() if key[1] == stage]
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        plt.figure()
        plt.scatter(x_values, y_values, label=str(stage), color=color)
        plt.title("Stage " + str(stage))
        plt.ylabel("Cost")
        plt.xlabel("Energy Level")
    plt.show()
