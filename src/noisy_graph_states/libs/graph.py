# -*- coding: utf-8 -*-
"""Auxiliary functions for graph states.

Includes both functions for handling graph states as density matrices
and some graph transformations that are not included directly in
graphepp as of version 0.4
"""

import numpy as np
import graphepp as gg
from . import matrix as mat
from cmath import sqrt
from .. import noisy_graph_states as nsf


def graph_state(graph):
    """Return the graph state in the computational basis.

    Parameters
    ----------
    graph : gg.Graph
        The graph describing the graph state.

    Returns
    -------
    np.ndarray
        A column-vector of the graph state given in the computational basis.
        shape = (2**graph.N, 1)

    """
    aux = [mat.x0] * graph.N
    psi = mat.tensor(*aux)
    for edge in graph.E:
        psi = mat.Ucz(psi, *edge)
    return psi


# Graph, ket state and density matrix of a Bell pair
bipartite_graph = gg.Graph(2, [(0, 1)])
bell_pair_ket = graph_state(bipartite_graph)
bell_pair_dm = bell_pair_ket @ mat.H(bell_pair_ket)

# Graph, ket state and density matrix of a 3-qubit GHZ state with the order leaf-root-leaf
ghz_3_graph = gg.Graph(3, [(0, 1), (1, 2)])
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
    Graph
        The graph corresponding to the adjacency matrix `adj`.

    """
    assert len(adj.shape) == 2
    assert adj.shape[0] == adj.shape[1]
    assert np.allclose(adj, adj.transpose())
    N = adj.shape[0]
    E = []
    for i in range(N):
        for j in range(i + 1, N):  # iterates over upper diagonal
            if adj[i, j] == 1:
                E += [(i, j)]
    return gg.Graph(N, E)


def disconnect_vertex(graph, index):
    """Disconnect a given vertex from the rest of the graph.

    This is done by removing all edges that are connecting
    to the specified `index`.

    Parameters
    ----------
    graph : Graph
        The input graph.
    index : int
        The `index`-th vertex is disconnected. Counting starts at 0.

    Returns
    -------
    Graph
        The updated graph.
    """
    adj = np.copy(graph.adj)
    adj[:, index] = np.zeros_like(adj[:, index])
    adj[index, :] = np.zeros_like(adj[index, :])
    return graph_from_adj_matrix(adj)


def measure_z(graph: gg.Graph, index: int):
    return disconnect_vertex(graph, index)


def measure_y(graph: gg.Graph, index: int):
    graph = gg.local_complementation(n=index, graph=graph)
    graph = measure_z(graph=graph, index=index)
    return graph


def measure_x(graph: gg.Graph, index: int, b0: int or None = None):
    if b0 is None:
        neighbours = neighbourhood(graph=graph, index=index)
        try:
            b0 = neighbours[0]
        except IndexError:
            b0 = None

    else:
        if b0 not in neighbourhood(graph=graph, index=index):
            raise ValueError(
                f"{b0=} is not in the neighbourhood of qubit {index} in graph {graph}."
            )
    if b0 is not None:
        graph = gg.local_complementation(n=b0, graph=graph)
    graph = gg.local_complementation(n=index, graph=graph)
    graph = measure_z(graph=graph, index=index)
    if b0 is not None:
        graph = gg.local_complementation(n=b0, graph=graph)
    return graph


def neighbourhood(graph, index):
    """Return the neighbouring vertices of a vertex.

    Parameters
    ----------
    graph : gg.Graph
        The neighbours are defined according to this graph.
    index : int
        The `index`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    tuple[int]
        Contains all the neighbours of the `index`-th vertex.

    """
    return tuple(np.nonzero(graph.adj[index, :])[0])


def update_graph_cnot(graph, source, target):
    """Returns the graph state after a CNOT between source and target.

    Parameters
    ----------
    graph : gg.Graph
        The CNOT is applied to this graph.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    graph : gg.Graph
        Graph after the CNOT is applied.

    """
    neighbours_source = neighbourhood(graph, source)
    neighbours_target = neighbourhood(graph, target)
    new_neighbours_source = nsf.add_or_remove(neighbours_target, neighbours_source)
    adj = np.copy(graph.adj)
    # first remove all edges from source
    adj[:, source] = np.zeros_like(adj[:, source])
    adj[source, :] = np.zeros_like(adj[source, :])
    # then add them in again
    for index in new_neighbours_source:
        adj[source, index] = 1
        adj[index, source] = 1
    graph = graph_from_adj_matrix(adj)
    return graph


def complement_op(graph, index):
    """Return the operator of local complementation on the `n`-th vertex

    Parameters
    ----------
    graph : gg.Graph
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
    for i in range(graph.N):
        if i == index:
            a = mat.tensor(a, 1 / sqrt(2) * (mat.I(2) - 1j * mat.X))
        elif graph.adj[i, index]:
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
    graph : gg.Graph
        The graph defining the desired graph state basis.

    Returns
    -------
    np.ndarray
        The same density matrix in the graph state basis.

    """
    # again here the specific form of the graph state enters
    g_state = graph_state(graph)
    my_tuple = ()
    for i in range(2**graph.N):  # get all 2**N basis states
        operator = np.array([[1]])
        for n in range(graph.N):  # build operator to generate state
            if i & (1 << ((graph.N - 1) - n)):
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
    graph : gg.Graph
        The graph defining the graph state basis, in which is given `rho`.

    Returns
    -------
    np.ndarray
        The same density matrix in the computational basis.


    """
    # again here the specific form of the graph state enters
    g_state = graph_state(graph)
    my_tuple = ()
    for i in range(2**graph.N):  # get all 16 basis states
        operator = np.array([[1]])
        for n in range(graph.N):  # build operator to generate state
            if i & (1 << ((graph.N - 1) - n)):
                operator = mat.tensor(operator, mat.Z)
            else:
                operator = mat.tensor(operator, mat.I(2))
        my_tuple += (np.dot(operator, g_state),)
    U = mat.H(np.hstack(my_tuple))
    return np.dot(np.dot(mat.H(U), rho), U)


def random_graph(num_vertices, p=0.5):
    """Generate random gg.Graph.

    Graph with `num_vertices` vertices. Each edge exists with probability `p`.
    """
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if np.random.random() < p:
                edges += [(i, j)]
    return gg.Graph(N=num_vertices, E=edges)
