def policy_control(stage, state, policy):
    controls = [control for control, _ in policy.items() if control[0] == stage]
    states = [state for _, state in controls]
    min_state = min(states, key=lambda x: abs(x - state))
    return policy[(stage, min_state)]

def test_policy(num_sim, length, parameters):
    """
    Return a dictionary where with items
    tuple(state,stage) -> average cost
    """
    big_states = []
    big_costs = []
    for _ in range(num_sim):
        costs = [0]
        i_state = 0
        states = [i_state]
        iter_range = range(0, length)
        for i in iter_range:
            mod_i = i % 24
            # pick next stochastic variables
            irr, load = gen_irr_and_load(mod_i, parameters.CITY)
            solar = irr * parameters.N_SOLAR * parameters.AREA_SOLAR * parameters.SOL_EFFICIENCY

            # pick control
            opt_control = policy_control(mod_i, i_state, policy)

            # update state
            cost = arbitrage_cost(mod_i, opt_control, load, solar, parameters)
            i_state = next_state(mod_i, i_state, opt_control, irr, load)
            states.append(i_state)
            costs.append(cost)

        big_states.append(states)
        big_costs.append(costs)
    return big_states, big_costs

parameters = Parameters(N_BATT=5, N_SOLAR=10, AREA_SOLAR=2, CITY='Phoenix')
states, costs = test_policy(5, 365 * 24, parameters)

def plot_states(states):
    plt.figure()
    for state_list in states:
        plt.plot(state_list)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('States Over Time')

def plot_costs(costs):
    plt.figure()
    for cost_list in costs:
        plt.plot(cost_list)
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.title('Costs Over Time')

def plot_cum_costs(costs):
    plt.figure()
    for cost_list in costs:
        plt.plot(np.cumsum(cost_list))
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.title('Cumulative Costs Over Time')

plot_states(states)
plot_costs(costs)
plot_cum_costs(costs)
plt.show()

