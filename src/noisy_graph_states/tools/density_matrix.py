# -*- coding: utf-8 -*-
"""An equivalent implementation using density matrices

It is a collection of functions that work with density matrices instead but
use the same syntax as in the main package.
Mostly useful for verification.
"""

import numpy as np
import graphepp as gg
from dataclasses import dataclass
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
from noisy_graph_states.libs.matrix import z0


@dataclass
class State(object):
    """A noisy graph state.

    Represents a quantum state in the density matrix
    formalism, where the `density_matrix` encloses all noises acting
    on the state. The corresponding to `graph` is also given.
    This has the same syntax that the State in noisy_graph_states,
    but using the density matrix instead of the noise maps.

    Parameters
    ----------
    graph : gg.Graph
        The graph of the underlying noiseless graph state.
    density_matrix : np.ndarray
        Noisy density matrix.

    Attributes
    ----------
    graph
    density_matrix

    """

    graph: gg.Graph
    density_matrix: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.graph == other.graph and np.allclose(
            self.density_matrix, other.density_matrix
        )


def x_noise(state, indices, epsilon):
    """Apply a Pauli-X channel on a single qubit.

    The effect on an input state rho is given by:
    (1 - epsilon) * rho + epsilon * X @ rho @ X

    Parameters
    ----------
    state : density_matrix.State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]

    Returns
    -------
    State : density_matrix.State
        The state after the noise channel has been applied.
    """
    rho = state.density_matrix
    for index in indices:
        rho = mat.xnoise(rho=rho, n=index, p=(1 - epsilon))
    return State(graph=state.graph, density_matrix=rho)


def y_noise(state, indices, epsilon):
    """Apply a Pauli-Y channel on a single qubit.

    The effect on an input state rho is given by:
    (1 - epsilon) * rho + epsilon * Y @ rho @ Y

    Parameters
    ----------
    state : density_matrix.State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]

    Returns
    -------
    State : density_matrix.State
        The state after the noise channel has been applied.
    """
    rho = state.density_matrix
    for index in indices:
        rho = mat.ynoise(rho=rho, n=index, p=(1 - epsilon))
    return State(graph=state.graph, density_matrix=rho)


def z_noise(state, indices, epsilon):
    """Apply a Pauli-Z channel on a single qubit.

    The effect on an input state rho is given by:
    (1 - epsilon) * rho + epsilon * Z @ rho @ Z

    Parameters
    ----------
    state : density_matrix.State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]

    Returns
    -------
    State : density_matrix.State
        The state after the noise channel has been applied.
    """
    rho = state.density_matrix
    for index in indices:
        rho = mat.znoise(rho=rho, n=index, p=(1 - epsilon))
    return State(graph=state.graph, density_matrix=rho)


def pauli_noise(state, indices, coefficients):
    """A Pauli-diagonal noise channel acts on a qubit.

    The effect on an input state rho is given by:
    p_0 * rho + p_1 * X @ rho @ X + p_2 * Y @ rho @ Y + p_3 * Z @ rho @ Z
    where
    p_0, p_1, p_2, p_3 = coefficients

    Parameters
    ----------
    state : density_matrix.State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    coefficients : list[scalar]
        The four coefficients of the noise channel, corresponding
        to the application Identity, X, Y and Z, respectively.
        Should sum to 1.

    Returns
    -------
    State : density_matrix.State
        The state after the noise channel has been applied.
    """
    p_0, p_1, p_2, p_3 = coefficients
    density_matrix = state.density_matrix
    for index in indices:
        identity = p_0 * density_matrix
        x_applied = p_1 * mat.xnoisy(density_matrix, index)
        y_applied = p_2 * mat.ynoisy(density_matrix, index)
        z_applied = p_3 * mat.znoisy(density_matrix, index)
        density_matrix = identity + x_applied + y_applied + z_applied
    return State(state.graph, density_matrix)


def local_complementation(state, index):
    """Performs a local complementation on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.

    Returns
    -------
    State : density_matrix.State
        The state after the manipulation has been applied.
    """
    graph = state.graph
    density_matrix = state.density_matrix
    # Update the density matrix
    complement_op = gt.complement_op(graph, index)
    new_density_matrix = complement_op @ density_matrix @ mat.H(complement_op)
    # Update the graph
    new_graph = gg.local_complementation(index, graph)
    return State(new_graph, new_density_matrix)


def z_measurement(state, index):
    """This function deletes all the edges of qubit index. It updates the state,
    which includes the graph and the density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    index : int
        Label of the qubit to be measured.

    Returns
    -------
    State : density_matrix.State
        The state after the manipulation has been applied.

    """
    graph = state.graph
    density_matrix = state.density_matrix
    # Update the density matrix
    projector = np.array([[1]])
    for i in range(graph.N):
        if i == index:
            projector = mat.tensor(projector, np.dot(z0, mat.H(z0)))
        else:
            projector = mat.tensor(projector, mat.I(2))
    new_density_matrix = projector @ density_matrix @ mat.H(projector)
    new_density_matrix = new_density_matrix / np.trace(
        new_density_matrix
    )  # Normalization
    # Update the graph
    new_graph = gt.disconnect_vertex(graph, index)
    return State(new_graph, new_density_matrix)


def y_measurement(state, index):
    """Performs a local Pauli Y measurement on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.

    Returns
    -------
    state : density_matrix.State
        The state after the manipulation has been applied.
    """
    state = local_complementation(state, index)
    state = z_measurement(state, index)
    return state


def x_measurement(state, index, b0=None):
    """Performs a local Pauli X measurement on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.
    b0 : int
        Index of the special neighbour of the X measurement. Counting starts at 0.
    Returns
    -------
    state : density_matrix.State
        The state after the manipulation has been applied.
    """
    neighbours = gt.neighbourhood(state.graph, index)
    if b0 is None:
        b0 = neighbours[0]
    else:
        if b0 in neighbours:
            b0 = b0
        else:
            raise ValueError(f"{b0=} is not in neighbourhood of {index=}: {neighbours}")
    state = local_complementation(state, b0)
    state = local_complementation(state, index)
    state = z_measurement(state, index)
    state = local_complementation(state, b0)
    return state


def cnot(state, source, target):
    """Performs a CNOT gate between source and target on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    State : density_matrix.State
        The state after the manipulation has been applied.
    """
    graph = state.graph
    density_matrix = state.density_matrix
    new_density_matrix = (
        mat.CNOT(source, target, graph.N)
        @ density_matrix
        @ mat.H(mat.CNOT(source, target, graph.N))
    )
    new_graph = gt.update_graph_cnot(graph, source, target)
    return State(new_graph, new_density_matrix)


def merge(state, source, target):
    """Performs a merging operation between source and target on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    state : density_matrix.State
        The state after the manipulation has been applied.
    """
    state = cnot(state, source, target)
    state = z_measurement(state, target)
    return state


def full_merge(state, source, target):
    """Performs a full-merging operation between source and target on a graph and its density matrix.

    Parameters
    ----------
    state : density_matrix.State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    state : density_matrix.State
        The state after the manipulation has been applied.
    """
    state = merge(state, source, target)
    state = y_measurement(state, source)
    return state


def fidelity(noiseless_ket, noisy_dm):
    """Compute the fidelity of a state in the density matrix formalism against the target state in ket formalism.
    Parameters
    ----------
    noiseless_ket : vector matrix
        The target state in the ket formalism
    noisy_dm : matrix
        The state in the density matrix formalism.

    Returns
    -------
    fidelity : int
        Fidelity, takes values [0, 1]
    """
    return mat.H(noiseless_ket) @ noisy_dm @ noiseless_ket
