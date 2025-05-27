import parameters
import numpy as np
import matplotlib.pyplot as plt
import no_grid_charge_solver
import pickle
import plotting
import os

structures = ['A', 'B', 'C']
cities = ['Phoenix', 'Sacramento', 'Seattle']

# Find optimal control structures
# Dict mapping pickle file name to avg day cost
avg_day_costs = {} # theoretical avg daily costs
try_over_write = False
for struc in structures:
    for city in cities:
        for batt in [0, 3, 5]:
            for solar in [0, 10, 20]:
                params = parameters.Parameters(STRUCTURE=struc, CITY=city, N_BATT=batt, N_SOLAR=solar, NO_GRID=True)
                fn = params.pickle_file_name()
                if os.path.exists(fn):
                    print(f"Policy for {city} and {struc} structure with {batt} batteries and {solar} solar panels already found at {fn}. Skipping.")
                    continue
                try_over_write = True
                print(f"Computing policy for {city} and {struc} structure with {batt} batteries and {solar} solar panels.")
                # Have not found policy for this condition yet
                no_grid_charge_solver.memo.clear()
                for start in params.state_space:
                    no_grid_charge_solver.solve(stage=0, state=start, parameters=params)
                policy = no_grid_charge_solver.extract_policy(no_grid_charge_solver.memo, params)

                avg_day_costs[fn] = plotting.get_day_cost(no_grid_charge_solver.memo, params)
                
                with open(fn, 'wb') as f:
                    pickle.dump(policy, f)
                    print(f"Policy saved at {fn} with average day cost: {avg_day_costs[fn]:.2f}.")
if try_over_write:
    if os.path.exists('avg_day_costs_nogrid_theory.pkl'):
        ans = input("avg_day_costs)nogrid_theory.pkl already exists. Overwrite? (y/n): ")
        if ans.lower() == 'y':
            with open('avg_day_nogrid_costs.pkl', 'wb') as f:
                pickle.dump(avg_day_costs, f)
        else:
            print("Not overwriting.")
    else:
        with open('avg_day_nogrid_costs.pkl', 'wb') as f:
            pickle.dump(avg_day_costs, f)

# Create and save arrow figures
fig_dir = 'figures/control_structures_no_grid/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
for fn in os.listdir('policies'):
    if fn.endswith('.pkl') and 'NOGRID' in fn:
        print(fn)
        city, struc, batt, solar, buy, _ = fn.split('_')
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_policy_{batt}_{solar}.png")
        if not os.path.exists(fig_fn):
            print(f"Creating figure for {city}, {struc} structure with {batt} batteries and {solar} solar panels. No grid.")
            with open(os.path.join('policies', fn), 'rb') as f:
                policy = pickle.load(f)
            plotting.plot_policy_states(policy, no_grid_charge_solver.next_state, parameters=params)
            plt.title(f"{city}, {struc} structure with {batt} batteries and {solar} solar panels. No grid charging.")
            plt.savefig(fig_fn)
            plt.close('all')
