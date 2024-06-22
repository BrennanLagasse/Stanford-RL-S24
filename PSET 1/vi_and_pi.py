### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

import copy

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.

    # TODO:

    # V(s) <- R(s,a) + \gamma*\sum_{s' \in S} P(s'|s,a)V(s')

    backup_val += R[state][action]

    for next_state in range(T.shape[0]):
        backup_val += gamma*T[state][action][next_state]*V[next_state]

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    # TODO:

    i = 0
    value_function_prev = copy.deepcopy(value_function)

    # Repeat until convergence
    while i == 0 or max(abs(value_function - value_function_prev)) > tol:
        value_function_prev = copy.deepcopy(value_function)

        for state in range(num_states):
            value_function[state] = bellman_backup(state, policy[state], R, T, gamma, value_function_prev)

        i += 1

    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    # TODO:

    for state in range(num_states):

        best_action, best_val = None, None
        
        for action in range(num_actions):

            val = bellman_backup(state, action, R, T, gamma, V_policy)

            if not best_val or val > best_val:
                best_action, best_val = action, val

        new_policy[state] = best_action

    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    # TODO:
    
    i = 0
    prev_policy = policy

    # Repeat until convergence of policies
    while i == 0 or max(abs(policy - prev_policy)) > 0:
        prev_policy = copy.deepcopy(policy)
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        policy = policy_improvement(policy, R, T, V_policy, gamma)

        i += 1

    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    # TODO:

    k = 0

    prev_value_function = copy.deepcopy(value_function)

    # Learn the value function
    while k == 0 or max(abs(value_function - prev_value_function)) > tol:
        prev_value_function = copy.deepcopy(value_function)
        
        for state in range(num_states):
            best_val = None

            for action in range(num_actions):
                
                val = bellman_backup(state, action, R, T, gamma, prev_value_function)

                if not best_val or val > best_val:
                    best_val = val

            value_function[state] = best_val

        k += 1

    # Induce the policy
    for state in range(num_states):
        best_val, best_action = None, None

        for action in range(num_actions):
            
            val = bellman_backup(state, action, R, T, gamma, value_function)

            if not best_val or val > best_val:
                best_action, best_val = action, val

        policy[state] = best_action

    return value_function, policy

def binary_search_policy(fun, R, T, tol):
    low, high = 0.0, 1.0

    while low < high:
        discount_factor = round((low + high) / 2, 2)

        if discount_factor == low: discount_factor = high

        V_pi, policy = fun(R, T, gamma=discount_factor, tol=tol)

        if policy[0] == 1:
            high = discount_factor - 0.01
        else:
            low = discount_factor

    return low


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    STRENGTHS = ['WEAK', 'MEDIUM', 'STRONG']

    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    s = 40

    print("\n" + "-" * s + "\nBeginning Policy Iteration\n" + "-" * s)

    for strength in STRENGTHS:
        env = RiverSwim(strength, SEED)
        R, T = env.get_model()
        V_pi, policy_pi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
        print(strength)
        print(f"\t{V_pi}")
        print(f"\t{[['L', 'R'][a] for a in policy_pi]}")

    print("\n" + "-" * s + "\nBeginning Value Iteration\n" + "-" * s)

    for strength in STRENGTHS:
        env = RiverSwim(strength, SEED)
        R, T = env.get_model()
        V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
        print(strength)
        print(f"\t{V_vi}")
        print(f"\t{[['L', 'R'][a] for a in policy_vi]}")

    print("\n" + "-" * s + "\nMax Decay for Not All Right (PI)\n" + "-" * s)

    for strength in STRENGTHS:
        env = RiverSwim(strength, SEED)
        R, T = env.get_model()
        discount_factor = binary_search_policy(policy_iteration, R, T, tol=1e-3)
        print(f"{strength}: {discount_factor}")

    print("\n" + "-" * s + "\nMax Decay for Not All Right (VI)\n" + "-" * s)

    for strength in STRENGTHS:
        env = RiverSwim(strength, SEED)
        R, T = env.get_model()
        discount_factor = binary_search_policy(value_iteration, R, T, tol=1e-3)
        print(f"{strength}: {discount_factor}")
