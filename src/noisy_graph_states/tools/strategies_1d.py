# -*- coding: utf-8 -*-
"""
A collection of measurement strategies and patterns on 1D cluster states.
"""

import math
import graphepp as gg
from itertools import permutations
from noisy_graph_states import State
from noisy_graph_states import Strategy
from noisy_graph_states.tools.patterns import pattern_to_sequence


def side_to_side(N):
    """This function creates a tuple of the Y measurements that need to be performed in an N-qubit 1D cluster
    to connect the two end of the cluster. The order of these measurements is side to side meaning from one end
    of the cluster to the other (left-to-right).

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    side_to_side : Strategy
        Strategy corresponding to the strategy of measuring side to side.
    """
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Create the corresponding sequence.
    sequence = tuple(("y", i) for i in range(1, N - 1))
    # Save the strategy (sequence and cluster).
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    return strategy


def reverse_side_to_side(N):
    """This function creates a tuple of the Y measurements that need to be performed in an N-qubit 1D cluster
    to connect the two end of the cluster. The order of these measurements is reverse side to side meaning from one end
    of the cluster to the other (right-to-left).

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    strategy : Strategy
        Strategy corresponding to the strategy of measuring reversed side to side.
    """
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Create the corresponding sequence.
    sequence = tuple(("y", i) for i in reversed(range(1, N - 1)))
    # Save the strategy (sequence and cluster).
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    return strategy


def every_second(N):
    """This function creates a tuple of the Y measurements that need to be performed in an N-qubit 1D cluster
    to connect the two end of the cluster. The order of these measurements is every second qubit.
    Such that, on the first loop one wants to measure every two qubits, thus 1, 3, 5, ...
    Then for round 1, you start measuring at 1 until N-2 every 2 --> range(1, N-1, 2)
    Next, you measure every second qubit of the ones that are left, so, 2, 6, 10...
    So for round 2, you start measuring at 2 until N-2 every 4 --> range(2, N-1, 4)
    Next, you measure every second qubit of the ones that are left, so, 4, 12, 20..
    So for round 3, you start measuring at 4 until N-2 every 8 --> range(2, N-1, 4)
    Thus, in general at each round you measure qubits in range(2**r, N-1, 2**(r+1)), where r starts at 0.
    After performing all the measurement r will reach the value such that range(2**r, N-1, 2**(r+1)) is empty.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    strategy : Strategy
        Strategy corresponding to the strategy of measuring every second qubit.
    """
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Create the corresponding sequence.
    sequence = tuple()
    r = 0
    while [j for j in range(2**r, N - 1, 2 ** (r + 1))]:
        for i in range(2**r, N - 1, 2 ** (r + 1)):
            sequence = (*sequence, ("y", i))
        r += 1
    # Save the strategy (sequence and cluster).
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    return strategy


def pairs(N):
    """This function creates a tuple of the Y measurements that need to be performed in an N-qubit 1D cluster
    to connect the two end of the cluster. The order of these measurements is every in pairs, starting from the outside
    to the inside. Such that first qubits 1 and N-2 are measured, then 3 and N-3, and so on.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    strategy : Strategy
        Strategy corresponding to the strategy of measuring in pairs.
    """
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Create the corresponding sequence.
    sequence = tuple()
    k = 1
    while k < math.ceil((N - 2) / 2):
        # If N is even, perform up until the second to last pair of measurements
        # If N is odd, perform up until the second to last measurement
        for i in [k, N - 1 - k]:
            sequence = (*sequence, ("y", i))
        k += 1
    if k == N - 1 - k:
        # If N is odd there is a single measurement left
        sequence = (*sequence, ("y", k))
    elif k + 1 == N - 1 - k:
        # If N is even there are two consecutive measurements left. They must be performed in different steps
        sequence = (*sequence, ("y", k))
        sequence = (*sequence, ("y", N - 1 - k))
    # Save the strategy (sequence and cluster).
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    return strategy


def all_y_strategies(N):
    """This function creates a list of all the possible tuples of the Y measurements that need to be performed in an
    N-qubit 1D cluster to connect the two end of the cluster.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    all_strategies : list of Strategy
        List of all the possible Strategy objects given a certain N.
    """
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Create an empty list to store the outcomes.
    strategies = []
    # Create a list of all possible permutations of the indices of the measured qubits.
    labels = list(permutations([*range(1, N - 1)]))
    for label in labels:
        # Create the corresponding sequence of a permutation.
        sequence = tuple()
        for i in label:
            sequence = (*sequence, ("y", i))
        # Save the strategy (sequence and cluster).
        strategies.append(Strategy(sequence=sequence, graph=linear_cluster))
    return strategies


def strategy_plus_z(strategy):
    """This function takes a strategy defined as a sequence of local Pauli measurements of the inner qubits of a 1D
    cluster such that the end nodes are connected, and transforms it into a strategy that also performs local Pauli
    measurements in the Z basis of the outer neighbours.

    Parameters
    ----------
    strategy : Strategy
        Order and bases of the measurements of the inner neighbours.

    Returns
    -------
    new_strategy : Strategy
        Order and bases of the measurements of the inner neighbours plus the local Pauli Z measurement of the two
        outer neighbours.

    """
    # Define the length of the chain with the target qubits, outer and inner neighbors.
    # The length of the initial strategy is only the inner neighbors.
    N = len(strategy.sequence) + 4
    # Create the corresponding 1D cluster state.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Convert the sequence (tuple) from the strategy into a list.
    list_sequence = list(list(i) for i in strategy.sequence)
    # Change the indices of the original strategy into the new ones. So we have to sum one to each inner neighbor.
    for measure in list_sequence:
        measure[1] += 1
    # Insert the local Pauli measurement in the Z basis of outer neighbors, 0 and N-1. We insert them in the beginning
    # of the strategy, as Z measurement can be place anywhere and do not affect the outcome.
    list_sequence.insert(0, ["z", N - 1])
    list_sequence.insert(0, ["z", 0])
    # Convert the list of the sequence into a tuple.
    tuple_sequence = tuple(tuple(i) for i in list_sequence)
    # Create a new strategy, with the new 1D cluster and the new sequence.
    new_strategy = Strategy(sequence=tuple_sequence, graph=linear_cluster)
    # Return the new strategy.
    return new_strategy


def _all_patterns_1d_pair(N):
    """This function gives a list of all possible measurement patterns of inner neighbours consisting of 'y' and 'x'
    measurements. It also includes the patterns of only 'y' and only 'x'.
    These measurement patterns manipulate a 1D cluster to a Bell pair between the end nodes.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    all_patterns : list
        List of all possible measurement patterns.
    """
    # Create an empty list to save the patterns.
    all_patterns = []
    # Establish the counter to 0.
    r = 0
    # Loop that ends when the counter is larger than the number of measured qubits.
    while r < N - 1:
        # Create a list of all possible permutations of the corresponding number of 'y' and 'x' measurements.
        patterns = permutations(["y"] * (N - 2 - r) + ["x"] * r)
        # Add the empty measurements for the target qubits.
        patterns = [["."] + list(p) + ["."] for p in patterns]
        # Save the patterns avoiding repetition.
        for i in patterns:
            if i not in all_patterns:
                all_patterns.append(i)
        r += 1
    return all_patterns


def all_strategies_1d_pair(N):
    """This function gives a list of all possible strategies of inner neighbours related to all the possible measurement
    patterns consisting of 'y' and 'x' measurements. It also includes the patterns of only 'y' and only 'x'.
    These strategies manipulate a 1D cluster to a Bell pair between the end nodes.
    The function checks internally that the strategy leads to the desired target state.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    good_strategies : list
        List of all possible strategies.
    """
    # Create the linear cluster, where the two end, labelled with 0 and N - 1, are the target qubits.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the initial state.
    state = State(graph=linear_cluster, maps=[])
    # Create the final Bell pair.
    bell_pair = gg.Graph(N=N, E=[(0, N - 1)])
    # Generate the target state.
    target_state = State(graph=bell_pair, maps=[])
    # Get all patterns.
    patterns = _all_patterns_1d_pair(N)
    # Transform each pattern to a sequence.
    sequences = [pattern_to_sequence(p, linear_cluster) for p in patterns]
    # Create a strategy for each sequence.
    strategies = [Strategy(sequence=s, graph=linear_cluster) for s in sequences]
    # Create an empty list to save the strategies that lead to the target state.
    good_strategies = []
    for strategy in strategies:
        # Apply each strategy and check if it leads to the target state or not.
        final_state = strategy(state)
        if final_state == target_state:
            good_strategies.append(strategy)
        else:
            print("bad strategy", strategy)
    return good_strategies
