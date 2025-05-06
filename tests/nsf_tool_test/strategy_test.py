"""This tests the strategy class.

They should behave the same as manual application.

Also tests the methods of the strategy class.
"""
import numpy as np
import pytest
import os
import noisy_graph_states.libs.graph as gt
import noisy_graph_states as nsf
import networkx as nx
from time import time
import shutil


def random_measurement_strategy(num_vertices: int):
    num_measurements = np.random.randint(1, num_vertices)
    measured_qubits = np.random.choice(
        np.arange(num_vertices, dtype=int), size=num_measurements, replace=False
    )
    strategy = tuple(
        (np.random.choice(["x", "y", "z"]), qubit_index)
        for qubit_index in measured_qubits
    )
    return strategy


def random_strategy(num_vertices: int):
    num_measurements = np.random.randint(1, num_vertices)
    measured_qubits = np.random.choice(
        np.arange(num_vertices, dtype=int), size=num_measurements, replace=False
    )
    strategy = tuple(
        (np.random.choice(["x", "y", "z", "lc"]), qubit_index)
        for qubit_index in measured_qubits
    )
    return strategy


def test_measurement_equivalence():
    # strategy application should be equivalent to calling individual functions in sequence
    for num_vertices in range(2, 12):
        for i in range(5):
            input_graph = gt.random_graph(num_vertices, p=0.8)
            input_state = nsf.State(graph=input_graph, maps=[])
            for qubit_index in range(num_vertices):
                coeff = np.random.random(4)
                coeff = coeff / np.sum(coeff)
                input_state = nsf.pauli_noise(
                    state=input_state, indices=[qubit_index], coefficients=coeff
                )
            instructions = random_measurement_strategy(num_vertices)
            strategy = nsf.Strategy(graph=input_graph, sequence=instructions)
            # strategy way
            strategy_way = strategy(input_state)
            # known way
            state = input_state
            for basis, qubit_index in instructions:
                if basis == "x":
                    state = nsf.x_measurement(state=state, index=qubit_index)
                elif basis == "y":
                    state = nsf.y_measurement(state=state, index=qubit_index)
                elif basis == "z":
                    state = nsf.z_measurement(state=state, index=qubit_index)
                else:
                    raise ValueError(f"measurement with {basis=} is not supported.")
            known_way = state
            assert strategy_way == known_way


def test_equivalence():
    # as previous test but also include local complementation "lc" instructions
    for num_vertices in range(2, 12):
        for i in range(5):
            input_graph = gt.random_graph(num_vertices, p=0.8)
            input_state = nsf.State(graph=input_graph, maps=[])
            for qubit_index in range(num_vertices):
                coeff = np.random.random(4)
                coeff = coeff / np.sum(coeff)
                input_state = nsf.pauli_noise(
                    state=input_state, indices=[qubit_index], coefficients=coeff
                )
            instructions = random_strategy(num_vertices)
            strategy = nsf.Strategy(graph=input_graph, sequence=instructions)
            # strategy way
            strategy_way = strategy(input_state)
            # known way
            state = input_state
            for instruction, qubit_index in instructions:
                if instruction == "x":
                    state = nsf.x_measurement(state=state, index=qubit_index)
                elif instruction == "y":
                    state = nsf.y_measurement(state=state, index=qubit_index)
                elif instruction == "z":
                    state = nsf.z_measurement(state=state, index=qubit_index)
                elif instruction == "lc":
                    state = nsf.local_complementation(state=state, index=qubit_index)
                else:
                    raise ValueError(f"Instruction {instruction=} is not supported.")
            known_way = state
            assert strategy_way == known_way


def test_speedup():
    # caching should make subsequent applications faster
    # not sure how to RELIABLY test this

    # this is the side-by-side strategy
    N = 32
    linear_cluster = nx.Graph([(i, i + 1) for i in range(N - 1)])
    sequence = tuple(("y", i) for i in range(1, N - 1))
    strategy = nsf.Strategy(graph=linear_cluster, sequence=sequence)

    state = nsf.State(graph=linear_cluster, maps=[])
    state = nsf.pauli_noise(
        state=state, indices=range(N), coefficients=[0.97, 0.01, 0.01, 0.01]
    )

    start_time = time()
    output_state = strategy(state)
    first_time = time() - start_time
    # following applications also on different states should be noticeably faster

    state = nsf.State(graph=linear_cluster, maps=[])
    state = nsf.pauli_noise(
        state=state, indices=range(N), coefficients=[0.985, 0.005, 0.005, 0.005]
    )
    start_time = time()
    output_state = strategy(state)
    second_time = time() - start_time
    # this should assert it being noticeably faster without having too narrow requirements
    assert second_time < (first_time * 3 / 4)

    # now with time dependent noise
    def error_parameter_from_time_interval(time_interval, dephasing_time):
        return (1 - np.exp(-time_interval / dephasing_time)) / 2

    distance_between_stations = 20e3
    communication_speed = 2e8
    dephasing_time = 100e-3
    # side to side protocol with all parties acting asap
    communication_interval = distance_between_stations / communication_speed
    times = [2 * (N - 2) * communication_interval] + [
        x * communication_interval for x in range(1, N)
    ]
    p = 0.99
    state = nsf.State(graph=linear_cluster, maps=[])
    state = nsf.pauli_noise(
        state=state,
        indices=range(N),
        coefficients=[p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4],
    )
    for i, time_interval in enumerate(times):
        state = nsf.z_noise(
            state=state,
            indices=[i],
            epsilon=error_parameter_from_time_interval(time_interval, dephasing_time),
        )
    start_time = time()
    output_state = strategy(state)
    third_time = time() - start_time
    # this should assert it being noticeably faster without having too narrow requirements
    assert third_time < (first_time * 3 / 4)


def test_populate_cache():
    # same as above test_speedup, but with populate cache instead of first application
    # this is the side-by-side strategy
    N = 32
    linear_cluster = nx.Graph([(i, i + 1) for i in range(N - 1)])
    sequence = tuple(("y", i) for i in range(1, N - 1))
    strategy = nsf.Strategy(graph=linear_cluster, sequence=sequence)

    start_time = time()
    strategy.populate_cache()
    first_time = time() - start_time
    # following applications also on different states should be noticeably faster

    state = nsf.State(graph=linear_cluster, maps=[])
    state = nsf.pauli_noise(
        state=state, indices=range(N), coefficients=[0.985, 0.005, 0.005, 0.005]
    )
    start_time = time()
    output_state = strategy(state)
    second_time = time() - start_time
    # this should assert it being noticeably faster without having too narrow requirements
    assert second_time < (first_time * 3 / 4)

    # now with time dependent noise
    def error_parameter_from_time_interval(time_interval, dephasing_time):
        return (1 - np.exp(-time_interval / dephasing_time)) / 2

    distance_between_stations = 20e3
    communication_speed = 2e8
    dephasing_time = 100e-3
    # side to side protocol with all parties acting asap
    communication_interval = distance_between_stations / communication_speed
    times = [2 * (N - 2) * communication_interval] + [
        x * communication_interval for x in range(1, N)
    ]
    p = 0.99
    state = nsf.State(graph=linear_cluster, maps=[])
    state = nsf.pauli_noise(
        state=state,
        indices=range(N),
        coefficients=[p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4],
    )
    for i, time_interval in enumerate(times):
        state = nsf.z_noise(
            state=state,
            indices=[i],
            epsilon=error_parameter_from_time_interval(time_interval, dephasing_time),
        )
    start_time = time()
    output_state = strategy(state)
    third_time = time() - start_time
    # this should assert it being noticeably faster without having too narrow requirements
    assert third_time < (first_time * 3 / 4)


@pytest.fixture
def clean_default_cache():
    if os.path.exists(nsf.DEFAULT_CACHE_DIR):
        shutil.rmtree(nsf.DEFAULT_CACHE_DIR)
    yield
    if os.path.exists(nsf.DEFAULT_CACHE_DIR):
        shutil.rmtree(nsf.DEFAULT_CACHE_DIR)


def test_save_load_default(clean_default_cache):
    for N in range(2, 12):
        graph = gt.random_graph(num_vertices=N)
        sequence = random_strategy(num_vertices=N)
        strat = nsf.Strategy(graph, sequence)
        strat.populate_cache()
        strat.save()

        input_state = nsf.State(graph=graph, maps=[])
        for qubit_index in range(N):
            coeff = np.random.random(4)
            coeff = coeff / np.sum(coeff)
            input_state = nsf.pauli_noise(
                state=input_state, indices=[qubit_index], coefficients=coeff
            )
        output_state = strat(input_state)

        strat_2 = nsf.Strategy(graph, sequence)
        assert len(strat_2._graph_sequence) == len(strat._graph_sequence)
        for graph_a, graph_b in zip(strat_2._graph_sequence, strat._graph_sequence):
            assert nx.utils.graphs_equal(graph_a, graph_b)
        assert strat_2._transform_noise_cache == strat._transform_noise_cache
        output_2 = strat_2(input_state)
        assert output_2 == output_state

        strat_3 = nsf.Strategy(graph, sequence, autoload=False)
        assert strat_3._transform_noise_cache != strat._transform_noise_cache
        strat_3.load()
        assert len(strat_3._graph_sequence) == len(strat._graph_sequence)
        for graph_a, graph_b in zip(strat_3._graph_sequence, strat._graph_sequence):
            assert nx.utils.graphs_equal(graph_a, graph_b)
        assert strat_3._transform_noise_cache == strat._transform_noise_cache
        output_3 = strat_3(input_state)
        assert output_3 == output_state


def test_save_load_custom(tmp_path):
    for N in range(2, 12):
        test_file_name = f"N{N}_test_file.pickle"
        graph = gt.random_graph(num_vertices=N)
        sequence = random_strategy(num_vertices=N)
        strat = nsf.Strategy(graph, sequence)
        strat.populate_cache()
        strat.save(os.path.join(tmp_path, test_file_name))

        input_state = nsf.State(graph=graph, maps=[])
        for qubit_index in range(N):
            coeff = np.random.random(4)
            coeff = coeff / np.sum(coeff)
            input_state = nsf.pauli_noise(
                state=input_state, indices=[qubit_index], coefficients=coeff
            )
        output_state = strat(input_state)

        strat_3 = nsf.Strategy(graph, sequence, autoload=False)
        with pytest.raises(FileNotFoundError):
            strat_3.load()
        assert strat_3._transform_noise_cache != strat._transform_noise_cache
        strat_3.load(os.path.join(tmp_path, test_file_name))
        assert len(strat_3._graph_sequence) == len(strat._graph_sequence)
        for graph_a, graph_b in zip(strat_3._graph_sequence, strat._graph_sequence):
            assert nx.utils.graphs_equal(graph_a, graph_b)
        assert strat_3._transform_noise_cache == strat._transform_noise_cache
        output_3 = strat_3(input_state)
        assert output_3 == output_state
