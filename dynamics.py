from parameters import *
import random
def get_expected_irr_and_load(stage, parameters):
    city = parameters.CITY
    stage = stage % 24
    if city == "Phoenix":
        means = [0, 0, 0, 0, 0, 0.032, 0.178, 0.410, 0.632,
            0.812, 0.942, 1.016, 1.028, 0.974, 0.862, 0.698, 0.498, 0.285, 0.124,
            0.018, 0, 0, 0, 0]
    elif city == "Sacramento":
        means = [0, 0, 0, 0, 0, 0.015, 0.142, 0.356, 0.556,
                 0.712, 0.825, 0.891, 0.902, 0.855, 0.756, 0.612, 0.436, 0.245, 0.098,
                 0.008, 0, 0, 0, 0]
    elif city == "Seattle":
        means = [0, 0, 0, 0, 0, 0, 0.072, 0.224, 0.367, 0.498,
                 0.594, 0.654, 0.676, 0.644, 0.562, 0.442, 0.302, 0.158, 0.054, 0,
                 0, 0, 0, 0]
    else:
        raise ValueError("City not recognized. Please use 'Phoenix', 'Sacramento', or 'Seattle'.")
        
    consump = [0.52, 0.42, 0.38, 0.35, 0.32, 0.38, 0.62, 0.98, 0.85, 0.68, 0.62, 0.65,
        0.75, 0.68, 0.65, 0.72, 0.95, 1.42, 1.95, 1.65, 1.38, 1.15, 0.88, 0.65]

    irr = means[stage]
    load = consump[stage]
    return irr, load
def get_irr_and_load_range(stage, parameters):
    city = parameters.CITY
    stage = stage % 24
    if city == "Phoenix":
        minVars = [0.8, 0.9, 0.85, 0.8]
        maxVars = [1.2, 1.1, 1.15, 1.2]
        means = [0, 0, 0, 0, 0, 0.032, 0.178, 0.410, 0.632,
            0.812, 0.942, 1.016, 1.028, 0.974, 0.862, 0.698, 0.498, 0.285, 0.124,
            0.018, 0, 0, 0, 0]
    elif city == "Sacramento":
        minVars = [0.7, 0.8, 0.75, 0.7]
        maxVars = [1.3, 1.2, 1.25, 1.3]
        means = [0, 0, 0, 0, 0, 0.015, 0.142, 0.356, 0.556,
                 0.712, 0.825, 0.891, 0.902, 0.855, 0.756, 0.612, 0.436, 0.245, 0.098,
                 0.008, 0, 0, 0, 0]
    elif city == "Seattle":
        minVars = [0.6, 0.65, 0.6, 0.55]
        maxVars = [1.40, 1.35, 1.40, 1.45]
        means = [0, 0, 0, 0, 0, 0, 0.072, 0.224, 0.367, 0.498,
                 0.594, 0.654, 0.676, 0.644, 0.562, 0.442, 0.302, 0.158, 0.054, 0,
                 0, 0, 0, 0]
    else:
        raise ValueError("City not recognized. Please use 'Phoenix', 'Sacramento', or 'Seattle'.")
        
    consump = [0.52, 0.42, 0.38, 0.35, 0.32, 0.38, 0.62, 0.98, 0.85, 0.68, 0.62, 0.65,
        0.75, 0.68, 0.65, 0.72, 0.95, 1.42, 1.95, 1.65, 1.38, 1.15, 0.88, 0.65]
    consumpVarMin = 0.8
    consumpVarMax = 1.2
    
    zone = [stage < 8, stage < 12, stage < 26, 1].index(1)

    return [minVars[zone] * means[stage], maxVars[zone] * means[stage]] , [consumpVarMin * consump[stage], consumpVarMax * consump[stage]] #irr, load
def gen_irr_and_load(stage, city):
    irr , load = get_irr_and_load_range(stage, city)
    return random.uniform(irr[0], irr[1]), random.uniform(load[0], load[1])
def next_state(state: float, control, parameters: Parameters):
    ETA = parameters.ETA
    return state + control * (1/ETA if control < 0 else ETA)
def control_from_state(current:float, next: float, parameters: Parameters):
    N_BATT = parameters.N_BATT
    ETA = parameters.ETA
    needed = (next - current) * (ETA if current > next else 1/ETA)
    if needed < max(-N_BATT*2,-current) or needed > min(2*N_BATT,5*N_BATT-current):
        return None
    return needed
def arbitrage_cost(stage, control, load, solar, parameters: Parameters):
    STRUCTURE = parameters.STRUCTURE
    stage = stage % 24
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

def get_carbon_emission(stage, parameters):
    city = parameters.CITY
    stage = stage % 24
    if city == "Phoenix":
        emission =  [420, 410, 400, 390, 400, 410, 430, 410, 350,
                 250, 180, 150, 140, 145, 160, 210, 290, 380, 460,
                 470, 450, 440, 430, 425]
    elif city == "Sacramento":
        emission = [340, 330, 320, 310, 320, 330, 350, 340, 310,
                 280, 260, 240, 230, 235, 250, 280, 320, 370, 410,
                 400, 380, 370, 360, 350]
    elif city == "Seattle":
        emission =  [120, 110, 100, 100, 100, 110, 120, 130, 140,
                 135, 130, 125, 120, 120, 125, 130, 140, 150, 160,
                 150, 140, 130, 125, 120]
    else:
        raise ValueError("City not recognized. Please use 'Phoenix', 'Sacramento', or 'Seattle'.")
    return emission[stage] 

def carbon_arbitrage_cost(stage, control, load, solar, parameters: Parameters):
    stage = stage % 24
    p_grid = load - solar + control
    CI = get_carbon_emission(stage,parameters)
    return p_grid * CI if p_grid > 0 else 0
