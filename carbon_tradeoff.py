import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from dynamics import *
from plotting import *


params = Parameters(
    N_BATT=5,
    N_SOLAR=5,
    CITY="Seattle",
    STRUCTURE="B"
)
initial_state = params.state_space[0]


def build(num_steps=11):

    weights = np.linspace(0, 1, num_steps)
    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)

    results = []
    for w1 in weights:
        memo.clear()
        _, ecost, emis = multiobjective_solver(
            stage=0,
            state=initial_state,
            params=params,
            w1=w1,
            min_econ=min_econ, max_econ=max_econ,
            min_emis=min_emis, max_emis=max_emis
        )
        results.append((w1, ecost, emis))

    return results

def plot_pareto(pareto):
    weights, ecosts, emissions = zip(*pareto)
    plt.figure(figsize=(8,5))
    plt.scatter(emissions, ecosts, s=50, edgecolor='black')
    for w, c, e in zip(weights, ecosts, emissions):
        plt.text(e, c, f"{w:.2f}", fontsize=8, ha='right', va='bottom')
    plt.xlabel('Total Emissions (g CO₂)')
    plt.ylabel('Total Economic Cost ($)')
    plt.title('Economic Cost vs. Emissions for ' + params.CITY +  ' with ' + params.STRUCTURE + ' structure')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_extremes(params, initial_state, min_econ, max_econ, min_emis, max_emis):

    # Pure economy (w1=1)
    memo.clear()
    _, cost_econ, emis_econ = multiobjective_solver(
        stage=0,
        state=initial_state,
        params=params,
        w1=1.0,
        min_econ=min_econ, max_econ=max_econ,
        min_emis=min_emis, max_emis=max_emis
    )

    # Pure emissions
    memo.clear()
    _, cost_emis, emis_emis = multiobjective_solver(
        stage=0,
        state=initial_state,
        params=params,
        w1=0.0,
        min_econ=min_econ, max_econ=max_econ,
        min_emis=min_emis, max_emis=max_emis
    )

    delta_cost = cost_emis - cost_econ
    delta_emis = emis_econ - emis_emis
    cost_change = (delta_cost / abs(cost_econ) * 100) if cost_econ != 0 else float('inf')
    emis_change = (delta_emis / emis_econ * 100) if emis_econ != 0 else float('inf')
    threshold = delta_cost / delta_emis if delta_emis != 0 else float('inf')

    print("Pure economy (w1=1.0): Cost = ${:.2f}, Emis = {:.0f} g CO₂".format(cost_econ, emis_econ))
    print("Pure emission (w1=0.0): Cost = ${:.2f}, Emis = {:.0f} g CO₂".format(cost_emis, emis_emis))
    print(f"Emissions increase by {delta_emis:.0f} g (+{emis_change:.1f}%) when focusing purely on economy.")
    print(f"Cost increase by ${delta_cost:.2f} (+{cost_change:.1f}%) when focusing purely on emissions.")
    print(f"Threshold trade-off: ${threshold:.4f} per g CO₂ avoided")

def compute_deltas(pareto):
 
    deltas = []
    for i in range(len(pareto)-1):
        w0, c0, e0 = pareto[i]
        w1, c1, e1 = pareto[i+1]
        dC = c1 - c0
        dE = e1 - e0
        slope = dC / dE if dE != 0 else float('inf')
        deltas.append((w0, w1, dC, dE, slope))
    return deltas

def plot_knee(pareto):

    deltas = compute_deltas(pareto)
    mids = [(w0 + w1)/2 for w0, w1, *_ in deltas]
    slopes = [abs(s) for *_, s in deltas]

    plt.figure(figsize=(6,4))
    plt.plot(mids, slopes, 'o-')
    plt.xlabel('Weight $w_1$')
    plt.ylabel('Marginal $/g CO₂')
    plt.title('Marginal Cost vs. Weight for ' + params.CITY +  ' with ' + params.STRUCTURE + ' structure')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tradeoff(pareto):

    weights, costs, emissions = zip(*pareto)
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    ax1.scatter(weights, costs, s=60, marker='o', color='tab:blue', label='Cost ($)')
    for w, c in zip(weights, costs):
        ax1.annotate(f"${c:.2f}", xy=(w, c), xytext=(3, 3), textcoords='offset points', color='tab:blue', fontsize=7)

    ax2.scatter(weights, emissions, s=60, marker='s', color='tab:orange', label='Emissions (g CO₂)')
    for w, e in zip(weights, emissions):
        ax2.annotate(f"{e:.0f}g", xy=(w, e), xytext=(3, -10), textcoords='offset points', color='tab:orange', fontsize=7)

    ax1.set_xlabel('Weight $w_1$')
    ax1.set_ylabel('Total Economic Cost ($)', color='tab:blue')
    ax2.set_ylabel('Total Emissions (g CO₂)', color='tab:orange')
    plt.title('Trade-off: Cost & Emissions vs. Weight for ' + params.CITY +  ' with ' + params.STRUCTURE + ' structure')
    fig.tight_layout()
    plt.show()


def plot_time_series(w1=0.10):

    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)

    memo.clear()
    controls, total_econ, total_emis = multiobjective_solver(
        0,                
        initial_state,
        params,          
        w1,     
        min_econ, max_econ,
        min_emis, max_emis
    )

    soc = initial_state
    econ_vals = []
    emis_vals = []
    for k, u in enumerate(controls):
        irr, load = get_expected_irr_and_load(k, params)
        solar    = irr * params.N_SOLAR * params.AREA_SOLAR * params.SOL_EFFICIENCY
        econ_k   = arbitrage_cost(k, u, load, solar, params)
        emis_k   = carbon_arbitrage_cost(k, u, load, solar, params)
        econ_vals.append(econ_k)
        emis_vals.append(emis_k)
        soc = next_state(soc, u, params)

    hours = np.arange(len(controls))
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    ax1.scatter(hours, econ_vals, marker='o', label='Economic Cost ($)')
    ax2.scatter(hours, emis_vals, marker='s', label='Emissions (g CO₂)')

    ax1.set_xlabel('Stage (hour)')
    ax1.set_ylabel('Economic Cost ($)', color='tab:blue')
    ax2.set_ylabel('Emissions (g CO₂)', color='tab:blue')


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Time-Series at w1={w1:.2f}: Cost & Emissions per Stage for {params.CITY} {params.STRUCTURE}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_tradeoff_curve(num_steps=100):

    initial_state = params.state_space[0]

    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)

    weights = np.linspace(0.0, 1.0, num_steps, endpoint=True)
    tradeoff = []

    for w1 in weights:
        memo.clear()

        controls, total_econ, total_emis = multiobjective_solver(
            0,
            initial_state,
            params,
            w1,
            min_econ, max_econ,
            min_emis, max_emis
        )

        tradeoff.append((w1, total_econ, total_emis))

    ws, econ_vals, emis_vals = zip(*tradeoff)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.scatter(ws, econ_vals, marker='o', label='Economic Cost ($)')
    ax2.scatter(ws, emis_vals, marker='s', label='Emissions (g CO₂)')

    ax1.set_xlabel('Weight $w_1$ (economic vs emissions)')
    ax1.set_ylabel('Total Economic Cost ($)')
    ax2.set_ylabel('Total Emissions (g CO₂)')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'Trade-off Curve for {params.CITY} {params.STRUCTURE}')
    plt.tight_layout()
    plt.show()

memo = {}


def multiobjective_solver(stage, state, params, w1, min_econ, max_econ, min_emis, max_emis):

    norm_cost, controls, total_econ, total_emis = _dp_recursive(
        stage, state, params, w1, min_econ, max_econ, min_emis, max_emis
    )
    return controls, total_econ, total_emis


def _dp_recursive(stage, state, params, w1, min_econ, max_econ, min_emis, max_emis):

    key = (stage, state, round(w1, 6))
    if key in memo:
        return memo[key]

    if stage == params.MAX_STAGE:
        result = (0.0, (), 0.0, 0.0)
        memo[key] = result
        return result
    
    irr, load = get_expected_irr_and_load(stage, params)
    solar = irr * params.N_SOLAR * params.AREA_SOLAR * params.SOL_EFFICIENCY

    best_norm_cost = float('inf')
    best_controls = ()
    best_raw_econ = 0.0
    best_raw_emis = 0.0

    for next_s in params.state_space:
        u = control_from_state(state, next_s, parameters=params)
        if u is None:
            continue

        (future_norm, future_ctls, future_econ, future_emis) = _dp_recursive(
            stage + 1,
            next_s,
            params,
            w1,
            min_econ, max_econ,
            min_emis, max_emis
        )

        econ_raw = arbitrage_cost(stage, u, load, solar, params)
        emis_raw = carbon_arbitrage_cost(stage, u, load, solar, params)

        norm_econ = (econ_raw - min_econ) / (max_econ - min_econ + 1e-8)
        norm_emis = (emis_raw - min_emis) / (max_emis - min_emis + 1e-8)

        total_norm = w1 * norm_econ + (1 - w1) * norm_emis + future_norm

        if total_norm < best_norm_cost:
            best_norm_cost = total_norm
            best_controls = (u,) + future_ctls
            best_raw_econ = econ_raw + future_econ
            best_raw_emis = emis_raw + future_emis

    result = (best_norm_cost, best_controls, best_raw_econ, best_raw_emis)
    memo[key] = result
    return result


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

    return (min(econ_vals), max(econ_vals), min(emis_vals), max(emis_vals))

if __name__ == "__main__":
    pareto = build(num_steps=101)
    # plot_pareto(pareto)
    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)
    compare_extremes(params, initial_state, min_econ, max_econ, min_emis, max_emis)

    deltas = compute_deltas(pareto)

    """
    print("\nAdjacent weight trade-offs:")
    print("w0→w1   ΔCost($)   ΔEmis(g)   $/gCO₂")
    for w0, w1, dC, dE, sl in deltas:
        print(f"{w0:.2f}→{w1:.2f}   {dC:8.2f}   {dE:8.0f}   {sl:8.4f}")
    """
    # plot_knee(pareto)
    # plot_tradeoff(pareto)
    # plot_time_series(w1=0.10)
    build_tradeoff_curve(num_steps=50)
