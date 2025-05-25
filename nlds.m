function [cntr, obj] = nlds(scenario, var,structure)

t = scenario.getTime() ;


N_batt = 5;
N_sp = 20;
area_pan = 2; 
area_tot = N_sp*area_pan;
n_charge = 0.95;
n_discharge = 0.95;
pcd = var.pcd;
energy = var.energy;

p_sol = scenario.data(2)*area_tot*0.15*0.82;

% p_grid

p_grid = (scenario.data(1) - p_sol + pcd(t));
[buy,sell] = grid_arbi_rates(t,structure);

pwr_cost = p_grid*buy;
% Energy constraints
energy_max_level = energy(t) <= N_batt*5;
energy_min_level = energy(t) >= 0;
% control constraints
pcd_cnt_pos = pcd(t) <= N_batt*2;
pcd_cnt_neg = pcd(t) >= -N_batt*2;

%pcd_const = pcd(t) == pcd_plus - pcd_minus;
if t == 1
    energy_level = energy(t) == pcd(t);
else
    energy_level = energy(t) == energy(t-1) + pcd(t);
    
end

obj = pwr_cost ;
cntr = [energy_min_level, ...
        energy_max_level, ...
        energy_level,...
        pcd_cnt_pos,...
        pcd_cnt_neg];
end