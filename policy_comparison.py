import deterministic_solver
import parameters
import plotting
import pickle
import matplotlib.pyplot as plt
import os
import dp_tester
import numpy as np
import dynamics
LENGTH = 7
def calc_equip_cost(params):
    num_solar = params.N_SOLAR
    num_batt = params.N_BATT

    kw = 0.4 * num_solar
    cost_per_kw = 1200 + 800 + 200
    kw_cost = kw * cost_per_kw

    cap = 5* num_batt
    cost_per_cap = 500

    panel_cost = 480 * num_solar
    battery_cost = 2500 * num_batt

    return 2000 + kw_cost + cap * cost_per_cap + panel_cost + battery_cost
# Simulate optimal policies, makes plots and estimates avg costs
structures = ['A', 'B', 'C']
cities = ['Phoenix', 'Sacramento', 'Seattle']
params = parameters.Parameters()
batts = [5,5,1]
state_dict = {}
cost_dict = {}
# Simulate optimal
if os.path.exists('state_dict.pkl') and os.path.exists('cost_dict.pkl'):
    print("Pickle files already exist. Skipping.")
    with open('state_dict.pkl', 'rb') as f:
        state_dict = pickle.load(f)
    with open('cost_dict.pkl', 'rb') as f:
        cost_dict = pickle.load(f)
else:
    for i,struc in enumerate(structures):
        params.STRUCTURE = struc
        batt = batts[i]
        solar = 20
        for city in cities:
            params.CITY = city
            params.N_BATT = batt
            params.N_SOLAR = solar
            fn = params.pickle_file_name()
            print(f"Simulating policy for {params.CITY}, {params.STRUCTURE} structure with {params.N_BATT} batteries and {params.N_SOLAR} solar panels.")
            with open(fn, 'rb') as f:
                policy = pickle.load(f)
            state_dict[fn],cost_dict[fn] = dp_tester.test_policy(5, 24*LENGTH, policy, params)
    with open('state_dict.pkl', 'wb') as f:
        pickle.dump(state_dict, f)
    with open('cost_dict.pkl', 'wb') as f:
        pickle.dump(cost_dict, f)
if os.path.exists('no_pol_state.pkl') and os.path.exists('no_pol_cost.pkl'):
    print("No equipment pickle files already exist. Skipping.")
    with open('no_pol_state.pkl', 'rb') as f:
        no_pol_state = pickle.load(f)
    with open('no_pol_cost.pkl', 'rb') as f:
        no_pol_cost = pickle.load(f)
else:
    no_policy = {k: 0 for k in policy.keys()}
    no_pol_state = {}
    no_pol_cost = {}
    for i, struc in enumerate(structures):
        params.STRUCTURE = struc
        batt = 0
        solar = 0
        for city in cities:
            params.CITY = city
            params.N_BATT = batt
            params.N_SOLAR = solar
            fn = params.pickle_file_name()
            print(f"Simulating no equipment for {params.CITY}, {params.STRUCTURE}.")
            no_pol_state[fn], no_pol_cost[fn] = dp_tester.test_policy(5, 24*LENGTH, no_policy, params)
    with open('no_pol_state.pkl', 'wb') as f:
        pickle.dump(no_pol_state, f)
    with open('no_pol_cost.pkl', 'wb') as f:
        pickle.dump(no_pol_cost, f)

# Plot average policy states against average no policy states
for i,struc in enumerate(structures):
    for city in cities:
        fig_fn = f"{city}_{struc}_avg_policy_costs.png"
        fig_dir = 'figures/avg_policy_cost'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        if not os.path.exists(os.path.join(fig_dir, fig_fn)):
            print(f"Plotting average costs for {city}, {struc} structure")
            opt_costs = []
            no_costs = []
            for fn in cost_dict.keys():
                fn_city, fn_struc,_,_,_ = fn[9:].split('_')
                if fn_struc == struc and fn_city == city:
                    opt_costs.extend(cost_dict[fn])
                    params = parameters.Parameters(CITY=city, STRUCTURE=struc, N_BATT=0, N_SOLAR=0)
                    no_costs.extend(no_pol_cost[params.pickle_file_name()])
            opt_params = parameters.Parameters(CITY=city, STRUCTURE=struc, N_BATT=batts[i], N_SOLAR=20)
            equip_cost = calc_equip_cost(opt_params)
            avg_opt_state = np.mean(opt_costs,axis=0)
            avg_opt_state[0] += equip_cost
            avg_no_state = np.mean(no_costs,axis=0)

            cum_opt_costs = np.cumsum(avg_opt_state)
            cum_no_costs = np.cumsum(avg_no_state)
            plt.plot(cum_opt_costs, label='Optimal')
            plt.plot(cum_no_costs, label='No Equipment')
            plt.xlabel('Time Steps')
            plt.ylabel('Cumulative Cost')
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(os.path.join(fig_dir, fig_fn))
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")
# Simulate "speculator"
def build_speculator_policy(opt_policy, params):
    keys = opt_policy.keys()
    max_sell = max([dynamics.buy_sell_rates(stage, params.STRUCTURE)[1] for stage in range(24)])
    min_buy = min([dynamics.buy_sell_rates(stage, params.STRUCTURE)[0] for stage in range(24)])
    new_policy = dict()
    for k in keys:
        # buy at cheapest rate, sell at most expensive
        stage = k[0]
        state = k[1]
        if dynamics.buy_sell_rates(stage, params.STRUCTURE)[0] <= min_buy + 0.01: # small amount for floating point
            new_policy[k] = min(params.N_BATT * params.BATT_CAP - state, params.N_BATT * 2)
        elif dynamics.buy_sell_rates(stage, params.STRUCTURE)[1] >= max_sell - 0.01: # small amount for floating point
            new_policy[k] = max(-state, -params.N_BATT*2)
        else:
            new_policy[k] = 0
    return new_policy
if os.path.exists('spec_state_dict.pkl') and os.path.exists('spec_cost_dict.pkl'):
    with open('spec_state_dict.pkl', 'rb') as f:
        spec_state_dict = pickle.load(f)
    with open('spec_cost_dict.pkl', 'rb') as f:
        spec_cost_dict = pickle.load(f)
else:
    spec_state_dict = {}
    spec_cost_dict = {}
    for i,struc in enumerate(structures):
        params.STRUCTURE = struc
        batt = batts[i]
        solar = 20
        for city in cities:
            params.CITY = city
            params.N_BATT = batt
            params.N_SOLAR = solar
            fn = params.pickle_file_name()
            with open(fn, 'rb') as f:
                policy = pickle.load(f)
            speculator_policy = build_speculator_policy(policy, params)
            spec_state_dict[fn],spec_cost_dict[fn] = dp_tester.test_policy(5, 24*LENGTH, speculator_policy, params)
    with open('spec_state_dict.pkl', 'wb') as f:
        pickle.dump(spec_state_dict, f)
    with open('spec_cost_dict.pkl', 'wb') as f:
        pickle.dump(spec_cost_dict, f)

# Simulate "speculator"
def build_speculator_policy(opt_policy, params):
    keys = opt_policy.keys()
    max_sell = max([dynamics.buy_sell_rates(stage, params.STRUCTURE)[1] for stage in range(24)])
    min_buy = min([dynamics.buy_sell_rates(stage, params.STRUCTURE)[0] for stage in range(24)])
    new_policy = dict()
    for k in keys:
        # buy at cheapest rate, sell at most expensive
        stage = k[0]
        state = k[1]
        if dynamics.buy_sell_rates(stage, params.STRUCTURE)[0] <= min_buy + 0.01: # small amount for floating point
            new_policy[k] = min(params.N_BATT * params.BATT_CAP - state, params.N_BATT * 2)
        elif dynamics.buy_sell_rates(stage, params.STRUCTURE)[1] >= max_sell - 0.01: # small amount for floating point
            new_policy[k] = max(-state, -params.N_BATT*2)
        else:
            new_policy[k] = 0
    return new_policy
if os.path.exists('spec_state_dict.pkl') and os.path.exists('spec_cost_dict.pkl'):
    with open('spec_state_dict.pkl', 'rb') as f:
        spec_state_dict = pickle.load(f)
    with open('spec_cost_dict.pkl', 'rb') as f:
        spec_cost_dict = pickle.load(f)
else:
    spec_state_dict = {}
    spec_cost_dict = {}
    for i,struc in enumerate(structures):
        params.STRUCTURE = struc
        batt = batts[i]
        solar = 20
        for city in cities:
            params.CITY = city
            params.N_BATT = batt
            params.N_SOLAR = solar
            fn = params.pickle_file_name()
            with open(fn, 'rb') as f:
                policy = pickle.load(f)
            speculator_policy = build_speculator_policy(policy, params)
            spec_state_dict[fn],spec_cost_dict[fn] = dp_tester.test_policy(5, 24*LENGTH, speculator_policy, params)
    with open('spec_state_dict.pkl', 'wb') as f:
        pickle.dump(spec_state_dict, f)
    with open('spec_cost_dict.pkl', 'wb') as f:
        pickle.dump(spec_cost_dict, f)
# Simulate "self-sufficient"
def build_self_sufficient_policy(opt_policy, params):
    keys = opt_policy.keys()
    new_policy = dict()
    for k in keys:
        # buy at cheapest rate, sell at most expensive
        stage = k[0]
        state = k[1]
        irr_range, load_range = dynamics.get_irr_and_load_range(stage, params)
        load_max = load_range[1]
        irr_min = irr_range[0]
        solar_min = dynamics.solar_from_irr(irr_min, params)
        surplus = solar_min - load_max
        if surplus > 0:
            new_policy[k] = min(params.N_BATT * params.BATT_CAP - state, params.N_BATT * 2, surplus)
        else:
            new_policy[k] = 0
    return new_policy
if os.path.exists('self_sufficient_state_dict.pkl') and os.path.exists('self_sufficient_dict.pkl'):
    with open('self_sufficient_dict.pkl', 'rb') as f:
        self_sufficient_state_dict = pickle.load(f)
    with open('self_sufficient_dict.pkl', 'rb') as f:
        self_sufficient_cost_dict = pickle.load(f)
else:
    self_sufficient_state_dict = {}
    self_sufficient_cost_dict = {}
    for i,struc in enumerate(structures):
        params.STRUCTURE = struc
        batt = batts[i]
        solar = 20
        for city in cities:
            params.CITY = city
            params.N_BATT = batt
            params.N_SOLAR = solar
            fn = params.pickle_file_name()
            with open(fn, 'rb') as f:
                policy = pickle.load(f)
            self_sufficient_policy = build_self_sufficient_policy(policy, params)
            self_sufficient_state_dict[fn],self_sufficient_cost_dict[fn] = dp_tester.test_policy(5, 24*LENGTH, self_sufficient_policy, params)
    with open('self_sufficient_state_dict.pkl', 'wb') as f:
        pickle.dump(self_sufficient_state_dict, f)
    with open('self_sufficient_cost_dict.pkl', 'wb') as f:
        pickle.dump(self_sufficient_cost_dict, f)
# Plot states through time for speculator and optimal
for i, struc in enumerate(structures):
    for city in cities:
        fig_fn = f"{city}_{struc}_states_through_time.png"
        fig_dir = 'figures/states_through_time'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        if not os.path.exists(os.path.join(fig_dir, fig_fn)):
            print(f"Plotting states through time for {city}, {struc} structure")
            opt_states = []
            spec_states = []
            self_sufficient_states = []
            for fn in state_dict.keys():
                fn_city, fn_struc, _, _, _ = fn[9:].split('_')
                if fn_struc == struc and fn_city == city:
                    opt_states.extend(state_dict[fn])
                    spec_states.extend(spec_state_dict[fn])
                    self_sufficient_states.extend(self_sufficient_state_dict[fn])
            
            avg_opt_state = np.mean(opt_states, axis=0)
            avg_spec_state = np.mean(spec_states, axis=0)
            avg_self_sufficient_state = np.mean(self_sufficient_states, axis=0)

            plt.plot(avg_opt_state, label='Optimal Policy')
            plt.plot(avg_spec_state, label='Speculator Policy')
            plt.plot(avg_self_sufficient_state, label='Self-Sufficient Policy')
            plt.xlabel('Time Steps')
            plt.ylabel('State Value')
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(os.path.join(fig_dir, fig_fn))
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")
for i, struc in enumerate(structures):
    for city in cities:
        fig_fn = f"{city}_{struc}_cumulative_costs_through_time.png"
        fig_dir = 'figures/cumulative_costs_through_time'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        if not os.path.exists(os.path.join(fig_dir, fig_fn)):
            print(f"Plotting cumulative costs through time for {city}, {struc} structure")
            opt_costs = []
            spec_costs = []
            self_sufficient_costs = []
            for fn in cost_dict.keys():
                fn_city, fn_struc, _, _, _ = fn[9:].split('_')
                if fn_struc == struc and fn_city == city:
                    opt_costs.extend(cost_dict[fn])
                    spec_costs.extend(spec_cost_dict[fn])
                    self_sufficient_costs.extend(self_sufficient_cost_dict[fn])
            
            cum_opt_cost = np.cumsum(np.mean(opt_costs, axis=0))
            cum_spec_cost = np.cumsum(np.mean(spec_costs, axis=0))
            cum_self_sufficient_cost = np.cumsum(np.mean(self_sufficient_costs, axis=0))

            plt.plot(cum_opt_cost, label='Optimal Policy')
            plt.plot(cum_spec_cost, label='Speculator Policy')
            plt.plot(cum_self_sufficient_cost, label='Self-Sufficient Policy')
            plt.xlabel('Time Steps')
            plt.ylabel('Cumulative Cost Value')
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(os.path.join(fig_dir, fig_fn))
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")

