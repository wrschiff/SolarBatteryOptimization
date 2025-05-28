import numpy as np
import matplotlib.pyplot as plt
import pickle
from plotting import *
from dynamics import *
import parameters


if __name__ == "__main__":
    cities = ['Phoenix', 'Sacramento', 'Seattle']
    
    for i in range(len(cities)):
        city = cities[i]
        params = Parameters(CITY=city, STRUCTURE='B', N_BATT=5, N_SOLAR=20, MAX_STAGE=24)
        policy = params.pickle_file_name_carbon()
        with open( policy, 'rb') as f:
            memo = pickle.load(f)
        #policy = extract_policy(memo, params) 
        plot_policy_boxes(memo, params)
        #plot_policy_states(memo, next_state, params)
        plt.savefig(f"figures/policy_thresholds/{city}_{params.STRUCTURE}_{params.N_BATT}_{params.N_SOLAR}threshold_carbon.png")
        #plt.show()
    