import noisy_graph_states.libs.graph as gt
import networkx as nx
import noisy_graph_states.libs.matrix as mat
import numpy as np
from cmath import sqrt


def test_graph_state():
    graph = nx.Graph([(0, 1)])
    state = gt.graph_state(graph)
    test_state = [[1 / 2], [1 / 2], [1 / 2], [-1 / 2]]
    assert np.allclose(state, test_state)


def test_disconnect_vertex():
    start_graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    final_graph = gt.disconnect_vertex(start_graph, 0)
    target_graph = nx.create_empty_copy(start_graph)
    target_graph.add_edges_from([(1, 2), (1, 3), (2, 3)])
    nx.utils.graphs_equal(final_graph, target_graph)


def test_neighbourhood():
    start_graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    for i in range(4):
        neighbourhood = gt.neighbourhood(start_graph, i)
        target_neighbourhood = list(range(4))
        target_neighbourhood.remove(i)
        target_neighbourhood = tuple(target_neighbourhood)
        assert neighbourhood == target_neighbourhood


def test_update_graph_cnot():
    start_graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    target_graph = nx.Graph([(0, 1), (1, 2), (3, 4), (1, 4)])
    final_graph = gt.update_graph_cnot(start_graph, source=1, target=3)
    nx.utils.graphs_equal(final_graph, target_graph)


def test_complement_op():
    graph = nx.Graph([(0, 1)])
    complement_op = gt.complement_op(graph, 0)
    target_complement_op = 1 / sqrt(2) * (mat.I(2) - 1j * mat.X)
    target_complement_op = mat.tensor(
        target_complement_op, 1 / sqrt(2) * (mat.I(2) + 1j * mat.Z)
    )
    for i, j in zip(range(4), range(4)):
        assert complement_op[i][j] == target_complement_op[i][j]
