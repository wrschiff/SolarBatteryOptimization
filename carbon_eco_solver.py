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
    emis_change = (delta_emis / emis_emis * 100) if emis_emis != 0 else float('inf')

    print("Pure economy (w1=1.0): Cost = ${:.2f}, Emis = {:.0f} g CO₂".format(cost_econ, emis_econ))
    print("Pure emission (w1=0.0): Cost = ${:.2f}, Emis = {:.0f} g CO₂".format(cost_emis, emis_emis))
    print(f"Emissions increase by {delta_emis:.0f} g (+{emis_change:.1f}%) when focusing purely on economy.")
    print(f"Cost increase by ${delta_cost:.2f} (+{cost_change:.1f}%) when focusing purely on emissions.")

    return emis_emis, emis_econ, cost_emis, cost_econ

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


def plot_gap_comparison(emis_pair, econ_pair, params):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pair, labels, title in zip(
        axes,
        [emis_pair, econ_pair],
        [
            ["Pure Emission","Pure Economy"],
            ["Pure Economy","Pure Emission"]
        ],
        ["Emissions Gap", "Economy Cost Gap"]
    ):
        baseline, alternative = pair
        values = [baseline, alternative]
        bars = ax.bar(labels, values, color=["tab:green","tab:orange"], edgecolor="k")
        ax.set_title(f"{title}: {params.CITY}, {params.STRUCTURE}")
        ax.set_ylabel("Value")
    
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x()+bar.get_width()/2,
                v*1.02,
                f"{v:.0f}", ha='center', va='bottom'
            )
   
        delta = alternative - baseline
        pct = (delta / baseline * 100) if baseline != 0 else 0
        x_alt = bars[1].get_x()+bars[1].get_width()/2
        ax.vlines(x_alt, ymin=baseline, ymax=alternative, color='black', linewidth=1.5)
        y_mid = (baseline+alternative)/2
        ax.text(
            x_alt + 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),
            y_mid,
            f"Δ={delta:.0f}\n({pct:.1f}%)",
            ha='left', va='center', fontweight='bold'
        )

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

    ax1.scatter(ws, econ_vals, marker='o', label='Economic Cost ($)', color='tab:blue')
    ax2.scatter(ws, emis_vals, marker='s', label='Emissions (g CO₂)', color='tab:orange')

    ax1.set_xlabel('Weight $w_1$ (economic vs emissions)')
    ax1.set_ylabel('Total Economic Cost ($)')
    ax2.set_ylabel('Total Emissions (g CO₂)')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'Trade-off Curve for {params.CITY} {params.STRUCTURE}')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    memo = {}
    initial_state = params.state_space[0]
    min_econ, max_econ, min_emis, max_emis = estimate_cost_ranges(params)
    emis_emis, emis_econ, cost_econ, cost_emis = compare_extremes(params, initial_state, min_econ, max_econ, min_emis, max_emis)
    emis_pair = (emis_emis, emis_econ)
    econ_pair = (cost_econ, cost_emis)
    plot_gap_comparison(emis_pair, econ_pair, params)
    build_tradeoff_curve(num_steps=50)
