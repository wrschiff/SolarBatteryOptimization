import deterministic_solver
import no_grid_charge_solver
import parameters
import plotting
import pickle
import matplotlib.pyplot as plt
import os
import dp_tester

structures = ['A', 'B', 'C']
cities = ['Phoenix', 'Sacramento', 'Seattle']
# Optimal control structure for each city and structure type
params = parameters.Parameters(N_BATT=5, N_SOLAR=20)

# Find optimal control structures
for struc in structures:
    params.STRUCTURE = struc
    for city in cities:
        params.CITY = city
        fn = params.pickle_file_name()
        if os.path.exists(fn):
            print(f"Policy for {city} and {struc} structure already found at {fn}. Skipping.")
            continue
        print(f"Computing policy for {city} and {struc} structure.")
        # Have not found policy for this condition yet
        deterministic_solver.memo.clear()
        for start in params.state_space:
            deterministic_solver.solve(stage=0, state=start, parameters=params)
        policy = deterministic_solver.extract_policy(deterministic_solver.memo, params)
        
        with open(fn, 'wb') as f:
            pickle.dump(policy, f)
            print(f"Policy saved at {fn}.")

# Create and save figures
fig_dir = 'figures/control_structures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
for fn in os.listdir('policies'):
    if fn.endswith('.pkl'):
        city, struc, batt, solar, buy = fn.split('_')
        with open(os.path.join('policies', fn), 'rb') as f:
            policy = pickle.load(f)
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_policy_{batt}_{solar}.png")
        plotting.plot_policy_states(policy, deterministic_solver.next_state,parameters=params)
        plt.title(f"{city}, {struc} structure with {batt} batteries and {solar} solar panels")
        plt.savefig(fig_fn)
        plt.close('all')
        prev = policy

# Simulate optimal policies
for struc in structures:
    params.STRUCTURE = struc
    for city in cities:
        params.CITY = city
        fn = params.pickle_file_name()
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                policy = pickle.load(f)
            states,costs = dp_tester.test_policy(5, 24*365, policy, params)
            fig_dir = 'figures/year_sims'
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_fn = os.path.join(fig_dir, f"{city}_{struc}_state_{batt}_{solar}.png")
            fig_fn = os.path.join(fig_dir, f"{city}_{struc}_cum_cost_{batt}_{solar}.png")
            plotting.plot_tester_cum_costs(costs)
            plt.title(f"{city}, {struc} structure with {batt} batteries and {solar} solar panels")
            plt.savefig(fig_fn)
            plt.close('all')
        else:
            print(f"Policy for {city} and {struc} structure not found. Skipping.")

