import numpy as np
class Parameters:
    def __init__(self, N_BATT=5, BATT_CAP=5, N_SOLAR=0, AREA_SOLAR=2, ETA=0.949, SOL_EFFICIENCY=0.15*0.82, STRUCTURE='B', CITY='Seattle', MAX_STAGE=24*7, N_STATE_DISC=50, NO_GRID=False):
        self.N_BATT = N_BATT
        self.BATT_CAP = BATT_CAP
        self.N_SOLAR = N_SOLAR
        self.AREA_SOLAR = AREA_SOLAR
        self.ETA = ETA
        self.SOL_EFFICIENCY = SOL_EFFICIENCY
        self.STRUCTURE = STRUCTURE
        self.CITY = CITY
        self.MAX_STAGE = MAX_STAGE
        self.N_STATE_DISC = N_STATE_DISC
        self.NO_GRID = NO_GRID
        self.state_space = np.linspace(0,self.N_BATT*self.BATT_CAP,self.N_STATE_DISC)
        if 0 not in self.state_space:
            self.state_space = np.insert(self.state_space, 0, 0)
    def pickle_file_name(self):
        cap = '_NOGRID_' if self.NO_GRID else ''
        return 'policies/' + self.CITY + '_' + self.STRUCTURE + '_' + str(self.N_BATT) + '_' + str(self.N_SOLAR) + cap + '_policy.pkl'
