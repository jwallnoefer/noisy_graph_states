# -*- coding: utf-8 -*-
"""Auxiliary functions for graph states.

Includes both functions for handling graph states as density matrices
and some graph transformations that are not included directly in
graphepp as of version 0.4
"""

import numpy as np
from . import matrix as mat
from cmath import sqrt
import networkx as nx


def local_complementation(graph: nx.Graph, index: int):
    neighbours_complete_graph = nx.create_empty_copy(graph)
    neighbours_complete_graph.update(nx.complete_graph(neighbourhood(graph, index)))
    return nx.symmetric_difference(graph, neighbours_complete_graph)


def graph_state(graph):
    """Return the graph state in the computational basis.

    Parameters
    ----------
    graph : nx.Graph
        The graph describing the graph state.

    Returns
    -------
    np.ndarray
        A column-vector of the graph state given in the computational basis.
        shape = (2**N, 1)

    """
    aux = [mat.x0] * len(graph)
    psi = mat.tensor(*aux)
    for edge in graph.edges:
        psi = mat.Ucz(psi, *edge)
    return psi


# Graph, ket state and density matrix of a Bell pair
bipartite_graph = nx.Graph([(0, 1)])
bell_pair_ket = graph_state(bipartite_graph)
bell_pair_dm = bell_pair_ket @ mat.H(bell_pair_ket)

# Graph, ket state and density matrix of a 3-qubit GHZ state with the order leaf-root-leaf
ghz_3_graph = nx.Graph([(0, 1), (1, 2)])
ghz_3_ket = graph_state(ghz_3_graph)
ghz_3_dm = ghz_3_ket @ mat.H(ghz_3_ket)


def graph_from_adj_matrix(adj):
    """Construct a Graph object from adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        The adjacency matrix. A symmetric 2D array with shape (N, N)
        where N is the number of vertices. An entry equal to 1 indicates
        an edge, while 0 indicates the absence of an edge.

    Returns
    -------
    nx.Graph
        The graph corresponding to the adjacency matrix `adj`.

    """
    assert len(adj.shape) == 2
    assert adj.shape[0] == adj.shape[1]
    assert np.allclose(adj, adj.transpose())
    return nx.Graph(adj)


def disconnect_vertex(graph, index):
    """Disconnect a given vertex from the rest of the graph.

    This is done by removing all edges that are connecting
    to the specified `index`.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    index : int
        The `index`-th vertex is disconnected. Counting starts at 0.

    Returns
    -------
    Graph
        The updated graph.
    """
    new_graph = graph.copy()
    new_graph.remove_edges_from(graph.edges(index))
    return new_graph


def measure_z(graph: nx.Graph, index: int):
    return disconnect_vertex(graph, index)


def measure_y(graph: nx.Graph, index: int):
    graph = local_complementation(graph=graph, index=index)
    graph = measure_z(graph=graph, index=index)
    return graph


def measure_x(graph: nx.Graph, index: int, b0: int or None = None):
    if b0 is None:
        neighbours = neighbourhood(graph, index)
        try:
            b0 = neighbours[0]
        except IndexError:
            b0 = None

    else:
        if b0 not in neighbourhood(graph, index):
            raise ValueError(
                f"{b0=} is not in the neighbourhood of qubit {index} in graph {graph}."
            )
    if b0 is not None:
        graph = local_complementation(graph=graph, index=b0)
    graph = local_complementation(graph=graph, index=index)
    graph = measure_z(graph=graph, index=index)
    if b0 is not None:
        graph = local_complementation(graph=graph, index=b0)
    return graph


def neighbourhood(graph, index):
    """Return the neighbouring vertices of a vertex.

    Parameters
    ----------
    graph : nx.Graph
        The neighbours are defined according to this graph.
    index : int
        The `index`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    tuple[int]
        Contains all the neighbours of the `index`-th vertex.

    """
    return tuple(graph[index])


def update_graph_cnot(graph, source, target):
    """Returns the graph state after a CNOT between source and target.

    Parameters
    ----------
    graph : nx.Graph
        The CNOT is applied to this graph.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    graph : nx.Graph
        Graph after the CNOT is applied.

    """
    new_graph = graph.copy()
    neighbours_target = graph[target]
    shared_neighbours = nx.common_neighbors(graph, source, target)
    new_graph.add_edges_from([(source, i) for i in neighbours_target])
    new_graph.remove_edges_from([(source, i) for i in shared_neighbours])
    return new_graph


def complement_op(graph, index):
    """Return the operator of local complementation on the `n`-th vertex

    Parameters
    ----------
    graph : nx.Graph
        The graph describing the adjacencies needed to define the operator.
    index : int
        The index of the vertex around which local complementation is performed.

    Returns
    -------
    np.ndarray
        An operator that performs the local complementation when applied to
        a graph state in the computational basis.
        shape = (2**graph.N, 2**graph.N)

    """
    a = np.array([[1]])
    neighbors = graph[index]
    for i in range(len(graph)):
        if i == index:
            a = mat.tensor(a, 1 / sqrt(2) * (mat.I(2) - 1j * mat.X))
        elif i in neighbors:
            a = mat.tensor(a, 1 / sqrt(2) * (mat.I(2) + 1j * mat.Z))
        else:
            a = mat.tensor(a, mat.I(2))
    return a


def Ugraph(rho, graph):
    """Switch from the computational basis to a graph state basis.

    Transforms a density matrix given in the computational basis to a density
    matrix given in the graph state basis defined by `graph`.

    Parameters
    ----------
    rho : np.ndarray
        A density matrix given in the computational basis.
    graph : nx.Graph
        The graph defining the desired graph state basis.

    Returns
    -------
    np.ndarray
        The same density matrix in the graph state basis.

    """
    # again here the specific form of the graph state enters
    g_state = graph_state(graph)
    N = len(graph)
    my_tuple = ()
    for i in range(2**N):  # get all 2**N basis states
        operator = np.array([[1]])
        for n in range(N):  # build operator to generate state
            if i & (1 << ((N - 1) - n)):
                operator = mat.tensor(operator, mat.Z)
            else:
                operator = mat.tensor(operator, mat.I(2))
        my_tuple += (np.dot(operator, g_state),)
    U = mat.H(np.hstack(my_tuple))
    return np.dot(np.dot(U, rho), mat.H(U))


def Ungraph(rho, graph):
    """Switch from a graph state basis to the computational basis.

    Transforms a density matrix given in the graph state basis defined by
    `graph` to a density matrix given in the computational basis.

    Parameters
    ----------
    rho : np.ndarray
        A density matrix given in the graph state basis.
    graph : nx.Graph
        The graph defining the graph state basis, in which is given `rho`.

    Returns
    -------
    np.ndarray
        The same density matrix in the computational basis.


    """
    # again here the specific form of the graph state enters
    g_state = graph_state(graph)
    N = len(graph)
    my_tuple = ()
    for i in range(2**N):  # get all 16 basis states
        operator = np.array([[1]])
        for n in range(N):  # build operator to generate state
            if i & (1 << ((N - 1) - n)):
                operator = mat.tensor(operator, mat.Z)
            else:
                operator = mat.tensor(operator, mat.I(2))
        my_tuple += (np.dot(operator, g_state),)
    U = mat.H(np.hstack(my_tuple))
    return np.dot(np.dot(mat.H(U), rho), U)


def random_graph(num_vertices, p=0.5):
    """Generate random nx.Graph.

    Graph with `num_vertices` vertices. Each edge exists with probability `p`.
    """
    return nx.gnp_random_graph(n=num_vertices, p=p)
