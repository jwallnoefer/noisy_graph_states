import numpy as np
import graphepp as gg
from random import choices
from itertools import permutations
from collections import defaultdict
import noisy_graph_states.libs.graph as gt
from noisy_graph_states.tools.patterns import sequence_to_pattern
from noisy_graph_states.tools.patterns import pattern_to_sequence
from noisy_graph_states.tools.patterns import pattern_to_all_sequences


def _order_to_pattern(sequence, graph):
    """An alternative conversion function for only y measurements"""
    tracking = defaultdict(int)
    output = ["."] * graph.N
    for index in sequence:
        neighbours = gt.neighbourhood(graph, index)
        for neighbour in neighbours:
            if tracking[neighbour] == 0:
                tracking[neighbour] = 1
            else:
                tracking[neighbour] = 0
        if tracking[index] == 1:
            output[index] = "x"
        else:
            output[index] = "y"
        graph = gt.measure_y(graph, index)
    return output


def _random_graph(num_vertices, p=0.5):
    """Generate random gg.Graph.

    Graph with `num_vertices` vertices. Each edge exists with probability `p`.
    """
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if np.random.random() < p:
                edges += [(i, j)]
    return gg.Graph(N=num_vertices, E=edges)


def test_y_equivalence():
    for N in range(3, 9):
        start_graph = gg.Graph(N, [(i, i + 1) for i in range(N - 1)])
        for order in permutations(range(1, N - 1)):
            seq = [("y", idx) for idx in order]
            pattern = sequence_to_pattern(seq, start_graph)
            pattern_alternative = _order_to_pattern(order, start_graph)
            assert pattern == pattern_alternative


def test_pattern_reversible_lcs():
    for N in range(1, 12):
        start_graph = gg.Graph(N, [(i, i + 1) for i in range(N - 1)])
        for _ in range(10):
            input_pattern = choices(["x", "y", "z", "."], k=N)
            sequence = pattern_to_sequence(pattern=input_pattern, graph=start_graph)
            output_pattern = sequence_to_pattern(sequence, graph=start_graph)
            assert input_pattern == output_pattern


def test_pattern_reversible_general():
    for N in range(1, 12):
        start_graph = _random_graph(N)
        for _ in range(10):
            input_pattern = choices(["x", "y", "z", "."], k=N)
            sequence = pattern_to_sequence(pattern=input_pattern, graph=start_graph)
            output_pattern = sequence_to_pattern(sequence, graph=start_graph)
            assert input_pattern == output_pattern


def test_pattern_all_reversible_lcs():
    for N in range(1, 6):
        start_graph = gg.Graph(N, [(i, i + 1) for i in range(N - 1)])
        for _ in range(10):
            input_pattern = choices(["x", "y", "z", "."], k=N)
            sequences = pattern_to_all_sequences(
                pattern=input_pattern, graph=start_graph
            )
            for sequence in sequences:
                output_pattern = sequence_to_pattern(sequence, graph=start_graph)
                assert input_pattern == output_pattern


def test_pattern_all_reversible_general():
    for N in range(1, 6):
        start_graph = _random_graph(N)
        for _ in range(10):
            input_pattern = choices(["x", "y", "z", "."], k=N)
            sequences = pattern_to_all_sequences(
                pattern=input_pattern, graph=start_graph
            )
            for sequence in sequences:
                output_pattern = sequence_to_pattern(sequence, graph=start_graph)
                assert input_pattern == output_pattern
