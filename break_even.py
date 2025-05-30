import pickle
import numpy as np
import matplotlib.pyplot as plt
import dynamics
import parameters
import os
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

# calculate the expected daily cost for power in each of the three cities without any solar panels or batteries
expected = {}
for city in ['Phoenix', 'Sacramento', 'Seattle']:
    expected_city = {}
    for struc in ['A', 'B', 'C']:
        params = parameters.Parameters(CITY=city, STRUCTURE=struc, N_SOLAR=0, N_BATT=0)
        costs = []
        for i in range(24):
            _, load= dynamics.get_expected_irr_and_load(i,params)
            costs.append(dynamics.arbitrage_cost(i,0,load,0,params))
        expected[city, struc] = np.sum(costs)

with open('avg_day_costs.pkl', 'rb') as f:
    avg_day_costs = pickle.load(f)

break_even_times = {}
daily_savings = {}
for city in ['Phoenix', 'Sacramento', 'Seattle']:
    for struc in ['A', 'B', 'C']:
        for num_solar in range(21):
            for num_batt in range(6):
                params = parameters.Parameters(CITY=city, STRUCTURE=struc, N_SOLAR=num_solar, N_BATT=num_batt)
                avg_cost = avg_day_costs[params.pickle_file_name()]
                avg_cost_without_equipment = expected[city, struc]
                daily_savings[params.pickle_file_name()] = avg_cost_without_equipment - avg_cost

for city in ['Phoenix', 'Sacramento', 'Seattle']:
    for struct in ['A', 'B', 'C']:
        fig, ax = plt.subplots()
        x = np.arange(21)
        y = np.arange(6)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((6, 21))

        for i in range(6):
            for j in range(21):
                params = parameters.Parameters(CITY=city, STRUCTURE=struct, N_SOLAR=j, N_BATT=i)
                Z[i, j] = daily_savings[params.pickle_file_name()]

        cmap = plt.cm.get_cmap('Blues')
        cs = ax.pcolor(X, Y, Z, cmap=cmap, edgecolors='black', linewidths=0.1, vmin=-1, vmax=15)

        for i in range(6):
            for j in range(21):
                params = parameters.Parameters(CITY=city, STRUCTURE=struct, N_SOLAR=j, N_BATT=i)
                daily_savings_days = daily_savings[params.pickle_file_name()]
                if daily_savings_days < 0:
                    ax.text(j, i, '0', ha='center', va='center', size=8, color='black')
                else:
                    ax.text(j, i, f"{daily_savings_days:.2f}", ha='center', va='center', size=5, color='black')

        ax.set_xlabel('Number of Solar Panels')
        ax.set_ylabel('Number of Batteries')
        ax.set_title(f'Daily Savings for {city} {struct} Structure')
        ax.set_xticks(np.arange(21))
        ax.set_xticklabels([str(i) for i in range(21)])
        ax.set_yticks(np.arange(6))
        ax.set_yticklabels([str(i) for i in range(6)])
        fig.savefig(os.path.join('figures/daily_savings', f'{city}_{struct}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

