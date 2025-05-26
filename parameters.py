import numpy as np
N_BATT = 5
BATT_CAP = 5
N_SOLAR = 5
AREA_SOLAR = 2
ETA = 0.949
SOL_EFFICIENCY = 0.15*0.82
STRUCTURE = 'A'
CITY = 'Phoenix'
MAX_STAGE = 24 * 7
N_STATE_DISC = 50
state_space = np.linspace(0,N_BATT*BATT_CAP,N_STATE_DISC)
if 0 not in state_space:
    state_space = np.insert(state_space, 0, 0)