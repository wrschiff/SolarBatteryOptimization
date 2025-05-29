import numpy as np
import matplotlib.pyplot as plt
import pickle
from plotting import *
from dynamics import *
import parameters


if __name__ == "__main__":
    cities = ['Phoenix', 'Sacramento', 'Seattle']
    structures = ['A', 'B','C']
    for structure in structures:
        for city in cities:
            params = Parameters(CITY=city, STRUCTURE=structure, N_BATT=5, N_SOLAR=5, MAX_STAGE=24)  # Create a parameters instance
            policy = params.pickle_file_name_grid_down()
            with open( policy, 'rb') as f:
                memo = pickle.load(f)
            #policy = extract_policy(memo, params) 
            plot_policy_boxes(memo, params)
            #plot_policy_states(memo, next_state, params)
            plt.savefig(f"figures/policy_thresholds/{city}_{params.STRUCTURE}_{params.N_BATT}_{params.N_SOLAR}threshold_GRIDDOWN.png")
            #plt.show()
        