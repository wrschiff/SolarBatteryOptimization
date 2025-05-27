import numpy as np
import matplotlib.pyplot as plt
from dynamics import *
from plotting import *
import parameters

memo = dict()


def multiobjective_solver(stage, state, params, w1, min_econ, max_econ, min_emis, max_emis):
    if (stage, state) in memo:
        return memo[(stage, state)]

    if stage == params.MAX_STAGE:
        return [(), 0.0, 0.0]

    irr, load = get_expected_irr_and_load(stage, params)
    controls_to_costs = dict()

    for next_state in params.state_space:
        control = control_from_state(state, next_state, parameters=params)
        if control is None:
            continue

        next_controls, econ_next, emis_next = multiobjective_solver(stage + 1, next_state, params, w1,
                                                                    min_econ, max_econ, min_emis, max_emis)

        solar = irr * params.N_SOLAR * params.AREA_SOLAR * params.SOL_EFFICIENCY

        econ = arbitrage_cost(stage, control, load, solar, params)
        emis = carbon_arbitrage_cost(stage, control, load, solar, params)

        norm_econ = (econ - min_econ) / (max_econ - min_econ + 1e-8)
        norm_emis = (emis - min_emis) / (max_emis - min_emis + 1e-8)

        combined_cost = w1 * norm_econ + (1 - w1) * norm_emis
        total_cost = combined_cost + (w1 * econ_next + (1 - w1) * emis_next)

        controls_to_costs[(control,) + next_controls] = (total_cost, econ + econ_next, emis + emis_next)

    best_controls, (final_cost, total_econ, total_emis) = min(controls_to_costs.items(), key=lambda x: x[1][0])
    memo[(stage, state)] = (best_controls, total_econ, total_emis)
    return memo[(stage, state)]


def estimate_cost_ranges(params):
    irr, load = get_expected_irr_and_load(0, params)
    solar = irr * params.N_SOLAR * params.AREA_SOLAR * params.SOL_EFFICIENCY

    econ_vals = []
    emis_vals = []
    for s1 in params.state_space:
        for s2 in params.state_space:
            u = control_from_state(s1, s2, parameters=params)
            if u is not None:
                econ_vals.append(arbitrage_cost(0, u, load, solar, params))
                emis_vals.append(carbon_arbitrage_cost(0, u, load, solar, params))

    return min(econ_vals), max(econ_vals), min(emis_vals), max(emis_vals)


if __name__ == "__main__":
    params = parameters.Parameters(N_BATT=5, N_SOLAR=20, CITY="Phoenix", STRUCTURE="B")
    start_states = np.linspace(0, params.N_BATT * params.BATT_CAP, params.N_STATE_DISC)

    w1 = 0.1 # eco
    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)

    memo.clear()
    costs = dict()

    for s in start_states:
        out = multiobjective_solver(0, s, params, w1, min_econ, max_econ, min_emis, max_emis)
        costs[s] = w1 * out[1] + (1 - w1) * out[2]

    policy = extract_policy(memo, params)
    plot_cost_function(memo, params)
    plot_policy_states(policy, next_state, params)
    plot_state_cost(0, memo, params)
    plt.show()
