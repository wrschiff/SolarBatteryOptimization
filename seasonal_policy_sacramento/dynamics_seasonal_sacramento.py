from parameters_seasonal import *
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np


from generate_dynamics_dbs import (
    sell_df,
    irr_upper_df,
    irr_lower_df,
    irr_mean_df,
    load_df,
    get_pge_buy_price,
)
 
def stage_to_calendar(stage, base_year=2025):
    #Convert an hour index since Jan-1 to (month, week_of_year, hour_of_day)
    dt = datetime(base_year, 1, 1) + timedelta(hours=stage)
    return dt.month, dt.isocalendar().week, dt.hour # (1-12, 1-52, 0-23)

def get_expected_irr_and_load(stage, parameters):
    #city = "Sacramento"
    month, week, hour = stage_to_calendar(stage, parameters.BASE_YEAR)
    irr = irr_mean_df.loc[week, hour]
    load = load_df.loc[hour, str(month)]
    return irr, load

def get_irr_and_load_range(stage, parameters):
    #city = parameters.CITY
    month, week, hour = stage_to_calendar(stage, parameters.BASE_YEAR)
    irr = irr_mean_df.loc[week, hour]
    load = load_df.loc[hour, str(month)]
    minVars = irr_lower_df.loc[week, hour]
    maxVars = irr_upper_df.loc[week, hour]
    consumpVarMin = 0.8
    consumpVarMax = 1.2
    return [minVars * irr, maxVars * irr] , [consumpVarMin * load, consumpVarMax * load] #irr, load

def gen_irr_and_load(stage, city):
    irr , load = get_irr_and_load_range(stage, city)
    return random.uniform(irr[0], irr[1]), random.uniform(load[0], load[1])

def next_state(state: float, control, parameters):
    ETA = parameters.ETA
    return state + control * (1/ETA if control < 0 else ETA)

def control_from_state(current:float, next: float, parameters):
    N_BATT = parameters.N_BATT
    ETA = parameters.ETA
    needed = (next - current) * (ETA if current > next else 1/ETA)
    if needed < max(-N_BATT*2,-current) or needed > min(2*N_BATT,5*N_BATT-current):
        return None
    return needed

def arbitrage_cost(stage, control, load, solar, parameters):
    #stage = stage % 24
    p_grid = load - solar + control
    
    [buy, sell] = buy_sell_rates(stage, parameters.BASE_YEAR)
    rate = buy if p_grid > 0 else sell
    return p_grid * rate

def buy_sell_rates(stage, base_year):
    stage_in_year = stage % 8760
    buy_rate = get_pge_buy_price(stage_in_year)
    month, _, hour = stage_to_calendar(stage_in_year, base_year)
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_name = month_names[month-1]
    
    # Access sell rate from sell_df (24 hours × 12 months)
    sell_rate = sell_df.loc[hour, month_name]
    return [buy_rate,sell_rate]

"""
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


def get_load_means(stage, parameters):
    city = parameters.CITY
    stage = stage % 24    
    consump = [0.52, 0.42, 0.38, 0.35, 0.32, 0.38, 0.62, 0.98, 0.85, 0.68, 0.62, 0.65,
        0.75, 0.68, 0.65, 0.72, 0.95, 1.42, 1.95, 1.65, 1.38, 1.15, 0.88, 0.65]
    return consump[stage] if city in ["Phoenix", "Sacramento", "Seattle"] else ValueError("City not recognized. Please use 'Phoenix', 'Sacramento', or 'Seattle'.")
   
def get_grid_down_energy_threshold(stage, parameters: Parameters):
    stage = stage % 24
    _, load_range = get_irr_and_load_range(stage, parameters)
    
    # (upper - thres) / (upper - lower) * p(Grid Down) = 0.05 
    thres = load_range[1] - (load_range[1] - load_range[0]) * 0.05 / parameters.GRID_DOWN_PROB[stage]
    if abs(thres - load_range[0]) < 1e-8:
        thres = 0
    return thres
"""