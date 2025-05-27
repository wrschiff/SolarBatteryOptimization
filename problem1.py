import deterministic_solver
import parameters
import plotting
import pickle
import matplotlib.pyplot as plt
import os
import dp_tester
import numpy as np

structures = ['A', 'B', 'C']
cities = ['Phoenix', 'Sacramento', 'Seattle']
# Optimal control structure for each city and structure type
params = parameters.Parameters(N_BATT=5, N_SOLAR=20)

# Find optimal control structures
# Dict mapping pickle file name to avg day cost
avg_day_costs = {} # theoretical avg daily costs
try_over_write = False
for struc in structures:
    for city in cities:
        for batt in range(6):
            for solar in range(21):
                params = parameters.Parameters(STRUCTURE=struc, CITY=city, N_BATT=batt, N_SOLAR=solar)
                fn = params.pickle_file_name()
                if os.path.exists(fn):
                    print(f"Policy for {city} and {struc} structure with {batt} batteries and {solar} solar panels already found at {fn}. Skipping.")
                    continue
                try_over_write = True
                print(f"Computing policy for {city} and {struc} structure with {batt} batteries and {solar} solar panels.")
                # Have not found policy for this condition yet
                deterministic_solver.memo.clear()
                for start in params.state_space:
                    deterministic_solver.solve(stage=0, state=start, parameters=params)
                policy = deterministic_solver.extract_policy(deterministic_solver.memo, params)

                avg_day_costs[fn] = plotting.get_day_cost(deterministic_solver.memo, params)
                
                with open(fn, 'wb') as f:
                    pickle.dump(policy, f)
                    print(f"Policy saved at {fn} with average day cost: {avg_day_costs[fn]:.2f}.")
if try_over_write:
    if os.path.exists('avg_day_costs.pkl'):
        ans = input("avg_day_costs.pkl already exists. Overwrite? (y/n): ")
        if ans.lower() == 'y':
            with open('avg_day_costs.pkl', 'wb') as f:
                pickle.dump(avg_day_costs, f)
        else:
            print("Not overwriting.")
    else:
        with open('avg_day_costs.pkl', 'wb') as f:
            pickle.dump(avg_day_costs, f)
# Create and save figures (arrow plots)
fig_dir = 'figures/control_structures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
for fn in os.listdir('policies'):
    if fn.endswith('.pkl'):
        city, struc, batt, solar, buy = fn.split('_')
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_policy_{batt}_{solar}.png")
        if not os.path.exists(fig_fn):
            print(f"Creating figure for {city}, {struc} structure with {batt} batteries and {solar} solar panels")
            with open(os.path.join('policies', fn), 'rb') as f:
                policy = pickle.load(f)
            plotting.plot_policy_states(policy, deterministic_solver.next_state, parameters=params)
            plt.title(f"{city}, {struc} structure with {batt} batteries and {solar} solar panels")
            plt.savefig(fig_fn)
            plt.close('all')

# Simulate optimal policies, makes plots and estimates avg costs
daily_costs = {} # simulated avg daily costs
try_over_write = False
for struc in structures:
    params.STRUCTURE = struc
    for city in cities:
        params.CITY = city
        for batt in range(6):
            for solar in range(21):
                params.N_BATT = batt
                params.N_SOLAR = solar
                fn = params.pickle_file_name()
                fig_fn = os.path.join(fig_dir, f"{city}_{struc}_cum_cost_{batt}_{solar}.png")
                if os.path.exists(fig_fn):
                    print(f"Figure for {city}, {struc} structure with {batt} batteries and {solar} solar panels already exists. Skipping.")
                    continue
                print(f"Simulating policy for {params.CITY}, {params.STRUCTURE} structure with {params.N_BATT} batteries and {params.N_SOLAR} solar panels.")
                if not os.path.exists(fn):
                    print(f"Policy for {city} and {struc} structure with {batt} batteries and {solar} solar panels not found. Skipping.")
                    continue
                try_over_write = True
                with open(fn, 'rb') as f:
                    policy = pickle.load(f)
                states,costs = dp_tester.test_policy(5, 24*365, policy, params)
                avg_costs = [sum(cost_list)/365 for cost_list in costs]
                daily_costs[fn] = np.mean(avg_costs)
                fig_dir = 'figures/year_sims'
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                plotting.plot_tester_cum_costs(costs)
                plt.title(f"{city}, {struc} structure with {batt} batteries and {solar} solar panels")
                plt.savefig(fig_fn)
                plt.close('all')

if try_over_write:
    if os.path.exists('daily_costs.pkl'):
        ans = input("daily_costs.pkl already exists. Overwrite? (y/n): ")
        if ans.lower() == 'y':
            with open('daily_costs.pkl', 'wb') as f:
                pickle.dump(daily_costs, f)
        else:
            print("Not overwriting.")
# Create and save solar analysis figures from SIM
for struc in structures:
    for city in cities:
        fig_dir = 'figures/sim_solar_equipment_analysis'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_daily_cost_v_solar.png")
        print(f"Processing solar figure for {city}, {struc} structure")
        with open('daily_costs.pkl', 'rb') as f:
            daily_costs = pickle.load(f)
        if not os.path.exists(fig_fn):
            x_values = list(range(21))
            for batt in range(6):
                y_values = [daily_costs[f"policies/{city}_{struc}_{batt}_{solar}_policy.pkl"] for solar in x_values]
                plt.plot(x_values, y_values, label=f"{batt} batteries")
            plt.xlabel("Number of Solar Panels")
            plt.ylabel("Average Daily Cost")
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(fig_fn)
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")

# Create and save battery analysis figures from SIM
for struc in structures:
    for city in cities:
        fig_dir = 'figures/sim_battery_equipment_analysis'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_daily_cost_v_batteries.png")
        with open('daily_costs.pkl', 'rb') as f:
            daily_costs = pickle.load(f)
        print(f"Processing battery figure for {city}, {struc} structure")
        if not os.path.exists(fig_fn):
            x_values = list(range(6))
            for solar in range(21):
                y_values = [daily_costs[f"policies/{city}_{struc}_{batt}_{solar}_policy.pkl"] for batt in x_values]
                plt.plot(x_values, y_values, label=f"{solar} solar panels")
            plt.xlabel("Number of Batteries")
            plt.ylabel("Average Daily Cost")
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(fig_fn)
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")

# Create and save solar analysis figures from THEORY
for struc in structures:
    for city in cities:
        fig_dir = 'figures/proj_solar_equipment_analysis'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_daily_cost_v_solar.png")
        print(f"Processing solar figure for {city}, {struc} structure")
        with open('avg_day_costs.pkl', 'rb') as f:
            daily_costs = pickle.load(f)
        if not os.path.exists(fig_fn):
            x_values = list(range(21))
            for batt in range(6):
                y_values = [daily_costs[f"policies/{city}_{struc}_{batt}_{solar}_policy.pkl"] for solar in x_values]
                plt.plot(x_values, y_values, label=f"{batt} batteries")
            plt.xlabel("Number of Solar Panels")
            plt.ylabel("Average Daily Cost")
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(fig_fn)
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")

# Create and save battery analysis figures
for struc in structures:
    for city in cities:
        fig_dir = 'figures/proj_battery_equipment_analysis'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_fn = os.path.join(fig_dir, f"{city}_{struc}_daily_cost_v_batteries.png")
        with open('avg_day_costs.pkl', 'rb') as f:
            daily_costs = pickle.load(f)
        print(f"Processing battery figure for {city}, {struc} structure")
        if not os.path.exists(fig_fn):
            x_values = list(range(6))
            for solar in range(21):
                y_values = [daily_costs[f"policies/{city}_{struc}_{batt}_{solar}_policy.pkl"] for batt in x_values]
                plt.plot(x_values, y_values, label=f"{solar} solar panels")
            plt.xlabel("Number of Batteries")
            plt.ylabel("Average Daily Cost")
            plt.title(f"{city}, {struc} structure")
            plt.legend()
            plt.savefig(fig_fn)
            plt.close('all')
        else:
            print(f"Figure for {city}, {struc} structure already exists. Skipping.")
