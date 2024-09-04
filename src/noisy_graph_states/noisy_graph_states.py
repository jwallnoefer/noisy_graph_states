from collections import defaultdict
import numpy as np
import graphepp as gg
from noisy_graph_states.libs import graph as gt
from dataclasses import dataclass
import noisy_graph_states.libs.matrix as mat
from copy import deepcopy
from functools import lru_cache, cached_property


@dataclass
class Map(object):
    """A noise map using the NSF convention.

    Can be used to represent Pauli-diagonal noise.
    Y and X noise are represented as correlated Z noises,
    which only makes sense in the context of graph states.


    Parameters
    ----------
    weights : list
        weights associated with this noise pattern. (1 - sum(weights)) is the weight of the identity.
    noises : list[tuple[int]]
        noises are represented by tuples of indices, indicating correlated Z-operators being applied to those qubits.

    Attributes
    ----------
    weights
    noises

    """

    weights: list
    noises: list[tuple[int]]

    def __call__(self, other):
        """Apply the Map to a State object.

        Effectively, this just adds the Map to the list of maps.

        Parameters
        ----------
        other : State

        Returns
        -------
        State
            The State after the application of the map.
        """
        if not isinstance(other, State):
            raise TypeError("Map can only be applied to State objects.")
        return State(graph=other.graph, maps=other.maps + [self])

    def __eq__(self, other):
        """Check if two maps are equal.

        Since there are ambiguities in how the noises
        can be specified, the standard form description
        is used to decide equality. See the `to_standard_form`
        method for a more detailed explanation.

        Parameters
        ----------
        other : Map

        Returns
        -------
        bool

        """
        first = (
            self.as_standard_form()
        )  # not using to_standard_form to avoid side effects
        second = other.as_standard_form()
        # expand to same shape, this catches floating point issues with weights close to 0
        all_noises = deepcopy(first.noises)
        for noise in second.noises:
            if noise not in first.noises:
                all_noises += [noise]
        all_noises = sorted(all_noises)
        first_new_weights = []
        second_new_weights = []
        for noise in all_noises:
            if noise in first.noises:
                first_new_weights += [first.weights[first.noises.index(noise)]]
            else:
                first_new_weights += [0]
            if noise in second.noises:
                second_new_weights += [second.weights[second.noises.index(noise)]]
            else:
                second_new_weights += [0]
        return np.allclose(
            first_new_weights, second_new_weights, atol=1e-14, rtol=1e-12
        )

    def as_standard_form(self):
        """Return an equivalent map that is in standard form.

        See the to_standard_form method for an explanation of the standard form.

        Returns
        -------
        Map
            an equivalent map that is in standard form
        """
        new_map = deepcopy(self)
        new_map.to_standard_form()
        return new_map

    def to_standard_form(self):
        """Transform the map to standard form in place.

        The standard form is useful because it makes the format more predictable for other functions.
        The standard form does this:
        * each individual noise is sorted
        * noises are sorted
        * duplicate noises are combined (their weights are added)
        * empty noises (corresponding to identity operator) are dropped
        * noises with weight 0 are dropped

        Returns
        -------
        None

        """
        new_noises = []
        new_weights = []
        for weight, noise in zip(self.weights, self.noises):
            noise = sorted(noise)
            if weight != 0:
                try:
                    index = new_noises.index(noise)
                except ValueError:  # if noise is not yet in list
                    if noise:  # drop identity noise []
                        new_noises += [noise]
                        new_weights += [weight]
                    continue
                new_weights[index] += weight
        try:
            new_noises, new_weights = zip(
                *sorted(
                    ((n, w) for n, w in zip(new_noises, new_weights)),
                    key=lambda x: x[0],
                )
            )
        except ValueError as e:
            if (
                new_noises == [] and new_weights == []
            ):  # the second zip raises ValueError if list is empty
                pass
            else:
                raise e
        self.noises = list(new_noises)
        self.weights = list(new_weights)


@dataclass
class State(object):
    """A noisy graph state.

    Represents a quantum state that is the
    perfect graph state corresponding to `graph`
    with the noise `maps` applied to it.

    Parameters
    ----------
    graph : gg.Graph
        The graph of the underlying graph state.
    maps : list[Map]
        Noise maps that are acting on the noiseless graph state.

    Attributes
    ----------
    graph
    maps

    """

    graph: gg.Graph
    maps: list[Map]  # of maps

    def __eq__(self, other):
        """States are equal if the graphs are equal and the maps are equivalent.

        Do note that there is some ambiguity in the
        representation of states in this description
        when it comes to graphs that are essentially
        the same in terms of entanglement but have
        additional disconnected vertices. Furthermore,
        it does not take into account other things like
        the local Clifford equivalence of graph states
        under local complementation.

        Parameters
        ----------
        other : State

        Returns
        -------
        bool
        """
        if not isinstance(other, State):
            return NotImplemented
        return self.graph == other.graph and compile_maps(*self.maps) == compile_maps(
            *other.maps
        )


@dataclass(frozen=True)
class Strategy(object):
    """A representation of a measurement strategy on a noisy graph state.

    This only supports transformations that consist solely of
    measurements. (no merging/connecting operations)
    One can use this class instead of individual functions to make use
    of the built-in caching mechanisms - i.e. repeated applications
    of the same strategy on different input states will be much faster.

    Parameters
    ----------
    graph : Graph
        The graph state at the beginning of the strategy.
    sequence : tuple[tuple[str, int]]
        A measurement order corresponding to combined operators, which both
        project on an eigenstate and perform the local Cliffords that return
        the state to a graph state.
        Individual instructions must be of form ("x" or "y" or "z", qubit_index).
        Optionally, x-measurements may specify the index of the special
        neighbour b0 as a third entry in the tuple ("x", qubit_index, b0).

    Attributes
    ----------
    graph
    sequence

    """

    graph: gg.Graph
    sequence: tuple[tuple[str, int]]

    @cached_property
    def _graph_sequence(self):
        current_graph = self.graph
        graph_sequence = [current_graph]
        for instruction in self.sequence:
            instruction_type = instruction[0]
            qubit_index = instruction[1]
            if instruction_type == "z":
                current_graph = gt.measure_z(graph=current_graph, index=qubit_index)
            elif instruction_type == "y":
                current_graph = gt.measure_y(graph=current_graph, index=qubit_index)
            elif instruction_type == "x":
                try:
                    b0 = instruction[2]
                except IndexError:
                    b0 = None
                current_graph = gt.measure_x(
                    graph=current_graph, index=qubit_index, b0=b0
                )
            else:
                raise ValueError(
                    f"{instruction[0]} is not an accepted measurement type."
                )
            graph_sequence += [current_graph]
        return tuple(graph_sequence)

    def __call__(self, other):
        """Apply the strategy to an initial state.

        Parameters
        ----------
        other : State

        Returns
        -------
        State
            The state after all measurements defined by the strategy were performed.
        """
        if not isinstance(other, State):
            raise TypeError("Strategy can only be applied to State objects.")
        if not other.graph == self.graph:
            raise ValueError(
                f"State Graph {other.graph} is not compatible with Strategy Graph {self.graph}."
            )
        transformed_maps = [
            self._transform_map(noise_map=noise_map) for noise_map in other.maps
        ]
        output_graph = self._graph_sequence[-1]
        return State(graph=output_graph, maps=transformed_maps)

    def populate_cache(self):
        """Calculate all noise patterns for local initial noise.

        This is a helper method that populates the cache of the _transform_noise method with
        patterns corresponding to local Pauli-diagonal initial noise.
        Generally, this will not be necessary, as the needed entries will be created anyway
        as needed, but this allows optionally to pre-calculate some of this before the first
        application.

        Returns
        -------
        None
        """
        for qubit_index in range(self.graph.N):
            z_pattern = (qubit_index,)
            x_pattern = gt.neighbourhood(graph=self.graph, index=qubit_index)
            y_pattern = tuple(sorted(x_pattern + z_pattern))
            self._transform_noise(noise=z_pattern)
            self._transform_noise(noise=y_pattern)
            self._transform_noise(noise=x_pattern)

    @lru_cache(maxsize=None)
    def _transform_noise(self, noise: tuple[int]):
        current_noise = noise
        for instruction, current_graph in zip(self.sequence, self._graph_sequence):
            instruction_type = instruction[0]
            qubit_index = instruction[1]
            if instruction_type == "z":
                current_noise = _z_measure_noise(noise=current_noise, index=qubit_index)
            elif instruction_type == "y":
                current_noise = _y_measure_noise(
                    noise=current_noise,
                    index=qubit_index,
                    neighbours=gt.neighbourhood(graph=current_graph, index=qubit_index),
                )
            elif instruction_type == "x":
                try:
                    b0 = instruction[2]
                except IndexError:
                    b0 = None
                if b0 is None:
                    neighbours = gt.neighbourhood(
                        graph=current_graph, index=qubit_index
                    )
                    try:
                        b0 = neighbours[0]
                    except IndexError:
                        b0 = None
                neighbours_sequence = _x_neighbours_sequence(
                    graph=current_graph, index=qubit_index, b0=b0
                )
                current_noise = _x_measure_noise(
                    noise=current_noise,
                    index=qubit_index,
                    b0=b0,
                    neighbours_sequence=neighbours_sequence,
                )
            else:
                raise ValueError(
                    f"{instruction[0]} is not an accepted measurement type."
                )
        return tuple(sorted(current_noise))

    def _transform_map(self, noise_map: Map):
        new_noises = [self._transform_noise(noise) for noise in noise_map.noises]
        return Map(weights=noise_map.weights, noises=new_noises)

    def get_weight_vector_expression(self):
        """Get an expression describing how initial local Pauli noise transforms under the strategy.

        This can be used to build an analytic expression that is equivalent to the strategy, but is more easily
        optimized.

        Returns
        -------
        dict:
            keys : tuple[int]
                final noise patterns
            values: list[str]
                which initial noises contribute to the noise pattern, in the format [xyz]_[0-9]+
                example: ["x_0", "y_3", "y_13", "z_8"]

        """
        expression = defaultdict(list)
        for qubit_index in range(self.graph.N):
            z_pattern = (qubit_index,)
            x_pattern = gt.neighbourhood(graph=self.graph, index=qubit_index)
            y_pattern = tuple(sorted(x_pattern + z_pattern))
            x_outcome = self._transform_noise(noise=x_pattern)
            expression[x_outcome] += [f"x_{qubit_index}"]
            y_outcome = self._transform_noise(noise=y_pattern)
            expression[y_outcome] += [f"y_{qubit_index}"]
            z_outcome = self._transform_noise(noise=z_pattern)
            expression[z_outcome] += [f"z_{qubit_index}"]
        return expression


def add_or_remove(index_list, noise):
    """Add noise operators if they are not there. Remove the noise operators if they are there.

    Parameters
    ----------
    index_list : Iterable
        List of the indices of the affected qubits, counting starts at 0.
    noise : tuple[int]
        Initial noise operators
    Returns
    -------
    tuple
        Final noise operators
    """
    new_noise = list(noise)
    for index in index_list:
        if index in new_noise:
            new_noise.remove(index)
        else:
            new_noise.append(index)
    return tuple(new_noise)


def x_noise(state, indices, epsilon):
    """Apply Pauli-X noise channel with error probability `epsilon` on a qubit.

    The equivalent effect (in density matrix notation) on an input state rho is:
    (1 - epsilon) * rho + epsilon * X @ rho @ X

    Parameters
    ----------
    state : State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]
    Returns
    -------
    State
        The state after the noise channel has been applied.
    """
    for index in indices:
        neighbours = gt.neighbourhood(graph=state.graph, index=index)
        noise_map = Map(weights=[epsilon], noises=[neighbours])
        state = noise_map(state)
    return state


def y_noise(state, indices, epsilon):
    """Apply Pauli-Y noise channel with error probability `epsilon` on a qubit.

    The equivalent effect (in density matrix notation) on an input state rho is:
    (1 - epsilon) * rho + epsilon * Y @ rho @ Y

    Parameters
    ----------
    state : State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]
    Returns
    -------
    State
        The state after the noise channel has been applied.
    """
    for index in indices:
        neighbours = gt.neighbourhood(graph=state.graph, index=index)
        noise_map = Map(weights=[epsilon], noises=[(index,) + neighbours])
        state = noise_map(state)
    return state


def z_noise(state, indices, epsilon):
    """Apply Pauli-Z noise channel with error probability `epsilon` on a qubit.

    The equivalent effect (in density matrix notation) on an input state rho is:
    (1 - epsilon) * rho + epsilon * Z @ rho @ Z

    Parameters
    ----------
    state : State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    epsilon : scalar
        The weight of the noise channel; should be in the interval [0, 1]
    Returns
    -------
    State
        The state after the noise channel has been applied.
    """
    for index in indices:
        noise_map = Map(weights=[epsilon], noises=[(index,)])
        state = noise_map(state)
    return state


def pauli_noise(state, indices, coefficients):
    """A Pauli-diagonal noise channel acts on a qubit.

    The equivalent effect (in density matrix notation) on an input state rho is given by:
    p_0 * rho + p_1 * X @ rho @ X + p_2 * Y @ rho @ Y + p_3 * Z @ rho @ Z
    where
    p_0, p_1, p_2, p_3 = coefficients

    Parameters
    ----------
    state : State
        The state on which the noise acts.
    indices : list
        List of the indices of the affected qubits, counting starts at 0.
    coefficients : list[scalar]
        The four coefficients of the noise channel, corresponding
        to the application Identity, X, Y and Z, respectively.
        Should sum to 1.

    Returns
    -------
    State
        The state after the noise channel has been applied.
    """
    p_0, p_1, p_2, p_3 = coefficients
    for index in indices:
        neighbours = gt.neighbourhood(graph=state.graph, index=index)
        x_pattern = neighbours
        y_pattern = tuple(sorted((index,) + neighbours))
        z_pattern = (index,)
        noise_map = Map(
            weights=[p_1, p_2, p_3], noises=[x_pattern, y_pattern, z_pattern]
        )
        state = noise_map(state)
    return state


@lru_cache(maxsize=16384)
def _complement_noise(noise, index, neighbours):
    """Update a single set of noises under local complementation.

    Update rule is:
    if there is Z-noise on the vertex, add correlated Z-noise on the neighbours

    This exists as a dedicated function, so we have a way to
    do specify transformations on noises without having to go via
    a State or the weights from a Map.
    The neighbours are expected to be calculated beforehand, to
    make the inputs easily hashable.

    Parameters
    ----------
    noise : tuple[int]
    index : int
    neighbours : tuple[int]

    Returns
    -------
    tuple[int]
        The updated list of vertices.

    """
    if index in noise:
        return tuple(add_or_remove(neighbours, noise))
    else:
        return noise


def _complement_map(noise_map, index, neighbours):
    """Update the noise map under local complementation.

    This already expects that you have calculated the neighbours of the index-th qubit,
    so a Graph object is no longer needed.


    Parameters
    ----------
    noise_map : Map
    index : int
    neighbours: tuple[int]

    Returns
    -------
    Map

    """
    new_noises = [
        _complement_noise(noise, index, neighbours) for noise in noise_map.noises
    ]
    return Map(weights=noise_map.weights, noises=new_noises)


def local_complementation(state, index):
    """Performs a local complementation on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    new_graph = gg.local_complementation(n=index, graph=state.graph)
    neighbours = gt.neighbourhood(state.graph, index)
    new_maps = [
        _complement_map(noise_map, index=index, neighbours=neighbours)
        for noise_map in state.maps
    ]
    return State(graph=new_graph, maps=new_maps)


def _z_measure_noise(noise: tuple[int], index: int):
    """Update the noise pattern under a Pauli-Z measurement.

    Parameters
    ----------
    noise : tuple[int]
        noise pattern.
    index : int
        the index-th qubit is measured in Z.

    Returns
    -------
    tuple[int]
        updated noise pattern.
    """
    if index in noise:
        new_noise = list(noise)
        new_noise.remove(index)
        return tuple(new_noise)
    else:
        return noise


def z_measurement(state, index):
    """Performs a local Pauli Z measurement on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    new_graph = gt.measure_z(state.graph, index)
    new_maps = []
    for map in state.maps:
        new_noises = [
            _z_measure_noise(noise=noise, index=index) for noise in map.noises
        ]
        new_maps += [Map(map.weights, new_noises)]
    return State(graph=new_graph, maps=new_maps)


def _y_measure_noise(noise: tuple[int], index: int, neighbours: tuple[int]):
    """Update the noise pattern under a Pauli-Y measurement.

    Parameters
    ----------
    noise : tuple[int]
        noise pattern.
    index : int
        the index-th qubit is measured in Y.
    neighbours : tuple[int]
        The neighbours of the index-th qubit.

    Returns
    -------
    tuple[int]
        updated noise pattern.
    """
    noise = _complement_noise(noise=noise, index=index, neighbours=neighbours)
    noise = _z_measure_noise(noise=noise, index=index)
    return noise


def y_measurement(state, index):
    """Performs a local Pauli Y measurement on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    new_graph = gt.measure_y(graph=state.graph, index=index)
    new_maps = []
    neighbours = gt.neighbourhood(graph=state.graph, index=index)
    for noise_map in state.maps:
        new_noises = [
            _y_measure_noise(noise=noise, index=index, neighbours=neighbours)
            for noise in noise_map.noises
        ]
        new_maps += [Map(weights=noise_map.weights, noises=new_noises)]
    return State(graph=new_graph, maps=new_maps)


def _x_measure_noise(
    noise: tuple[int],
    index: int,
    b0: int or None,
    neighbours_sequence: list[tuple[int]],
):
    """Update the noise pattern under a Pauli-X measurement.

    Parameters
    ----------
    noise : tuple[int]
        noise pattern.
    index : int
        the index-th qubit is measured in X.
    b0 : int or None
        the special neighbor b0, None is only allowed if the
        index-th vertex is isolated.
    neighbours_sequence : list[tuple[int]]
        A sequence of neighbours at the various steps that need to
        be followed for an X-measurement. _x_neighbours_sequence
        can be used to obtain them in the expected format.

    Returns
    -------
    tuple[int]
        updated noise pattern.
    """
    if b0 is not None:
        noise = _complement_noise(
            noise=noise, index=b0, neighbours=neighbours_sequence[0]
        )
        noise = _complement_noise(
            noise=noise, index=index, neighbours=neighbours_sequence[1]
        )
        noise = _z_measure_noise(noise=noise, index=index)
        noise = _complement_noise(
            noise=noise, index=b0, neighbours=neighbours_sequence[2]
        )
    else:
        noise = _complement_noise(
            noise=noise, index=index, neighbours=neighbours_sequence[0]
        )
        noise = _z_measure_noise(noise=noise, index=index)
    return noise


def _x_neighbours_sequence(graph: gg.Graph, index: int, b0: int or None):
    """Return a sequence of neighbours in the required format for _x_measure_noise.

    Parameters
    ----------
    graph : Graph
        the starting graph before the measurement
    index : int
        The index-th qubit that will be measured in X.
    b0 : int or None
        This neighbour of the index-th qubit is chosen as the special neighbour b0.
        None is here only allowed if the vertex is isolated.

    Returns
    -------
    list[tuple[int]]
    """
    if b0 is None:
        if gt.neighbourhood(graph, index=index):  # i.e. is not empty
            raise ValueError(
                f"b0=None is only allowed if the vertex {index=} is isolated, but it is not in {graph=}."
            )
    elif b0 not in gt.neighbourhood(graph, index=index):
        raise ValueError(
            f"{b0=} is not in the neighbourhood of qubit {index} in graph {graph}."
        )
    neighbour_sequence = []
    if b0 is not None:
        neighbour_sequence += [gt.neighbourhood(graph=graph, index=b0)]
        graph = gg.local_complementation(n=b0, graph=graph)
    neighbour_sequence += [gt.neighbourhood(graph=graph, index=index)]
    graph = gg.local_complementation(n=index, graph=graph)
    graph = gt.measure_z(graph=graph, index=index)
    if b0 is not None:
        neighbour_sequence += [gt.neighbourhood(graph=graph, index=b0)]
        # graph = gg.local_complementation(n=b0, graph=graph)
    return neighbour_sequence


def x_measurement(state, index, b0=None):
    """Performs a local Pauli X measurement on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    index : int
        The `index`-th vertex is where the local complementation is applied. Counting starts at 0.
    b0 : int
        Index of the special neighbour of the X measurement. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    neighbours = gt.neighbourhood(state.graph, index)
    if b0 is None:
        try:
            b0 = neighbours[0]
        except IndexError:
            b0 = None
    else:
        if b0 not in neighbours:
            raise ValueError(f"{b0=} is not in neighbourhood of {index=}: {neighbours}")
    new_graph = gt.measure_x(graph=state.graph, index=index, b0=b0)
    neighbour_sequence = _x_neighbours_sequence(graph=state.graph, index=index, b0=b0)
    new_maps = []
    for noise_map in state.maps:
        new_noises = [
            _x_measure_noise(
                noise=noise, index=index, b0=b0, neighbours_sequence=neighbour_sequence
            )
            for noise in noise_map.noises
        ]
        new_maps += [Map(weights=noise_map.weights, noises=new_noises)]
    return State(graph=new_graph, maps=new_maps)


def cnot(state, source, target):
    """Performs a CNOT gate between source and target on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    graph = gt.update_graph_cnot(state.graph, source, target)
    new_maps = []
    for map in state.maps:
        new_noises = []
        for noise in map.noises:
            if target in noise:
                new_noises += [add_or_remove([source], noise)]
            else:
                new_noises += [noise]
        new_maps += [Map(map.weights, new_noises)]
    return State(graph=graph, maps=new_maps)


def merge(state, source, target):
    """Performs a merging operation between source and target on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    state = cnot(state, source, target)
    state = z_measurement(state=state, index=target)
    return state


def full_merge(state, source, target):
    """Performs a full-merging operation between source and target on a graph and its noise.

    Parameters
    ----------
    state : State
        The state on which the manipulation is performed.
    source : int
        The `source`-th vertex is considered. Counting starts at 0.
    target : int
        The `target`-th vertex is considered. Counting starts at 0.

    Returns
    -------
    State
        The state after the manipulation has been applied.
    """
    state = merge(state, source, target)
    state = y_measurement(state=state, index=source)
    return state


def compile_maps(*args):
    """Combine multiple Maps into a single equivalent Map.

    Use with care, because this can impact efficiency. If arbitrary maps are combined, exponentially many entries
    may be needed to describe the final map.

    Parameters
    ----------
    args : Map
        Any number of Map objects that will be combined.

    Returns
    -------
    Map
        A single Map object that is equivalent to the application of all the Map objects provided in `args`.
    """
    res = Map(weights=[], noises=[])
    for map in args:
        # this whole iteration seems a bit inefficient, maybe accounting for duplicates directly
        # instead of using Map.to_standard_form may be faster?
        new_weights = []
        new_noises = []
        res_identity_weight = 1 - sum(res.weights)
        map_identity_weight = 1 - sum(map.weights)
        new_weights += [map_identity_weight * w for w in res.weights]
        new_noises += res.noises
        new_weights += [res_identity_weight * w for w in map.weights]
        new_noises += map.noises
        # and now all the cross terms
        for weight, noise in zip(res.weights, res.noises):
            new_weights += [weight * w for w in map.weights]
            new_noises += [add_or_remove(index_list=n, noise=noise) for n in map.noises]
        res.weights = new_weights
        res.noises = new_noises
        res.to_standard_form()
    return res


def reduce_maps(state, target_indices):
    """Compute the reduced noise maps of all the state on a subset of qubits.

    Parameters
    ----------
    state : State
        The state on whose maps will be reduced.
    target_indices : list[int]
        Indices of the target qubits. Counting starts at 0.

    Returns
    -------
    list[Map]
        Reduced maps
    """
    maps = state.maps
    reduced_maps = []
    for map in maps:
        reduced_noises = []
        for noise in map.noises:
            new_noise = list(noise)
            for index in noise:  # iterate over immutable original noise
                if index not in target_indices:
                    new_noise.remove(index)
            reduced_noises += [tuple(new_noise)]
        reduced_maps += [Map(map.weights, reduced_noises)]
    return reduced_maps


def apply_nsf_maps_to_dm(maps, density_matrix, target_indices):
    """Apply maps to a density matrix representing a subset of qubits.

    Parameters
    ----------
    maps : list[Map]
        All noise maps that should be applied.
    density_matrix : np.ndarray
        Input state to which the noise maps will be applied.
        Must have shape (2**N, 2**N) with N=len(target_indices) and
        the qubits are assumed to be in the same order as specified
        in `target_indices`.
    target_indices : list
        Indices of subset of qubits that is considered. Counting starts at 0.

    Returns
    -------
    np.ndarray
        Density matrix after the application of the maps.
    """
    map = compile_maps(*maps)
    weights = [1 - np.sum(map.weights)] + map.weights
    operators = [mat.I(len(density_matrix))]
    for noise in map.noises:
        operator = np.array([[1]])
        for target_index in target_indices:
            if target_index in noise:
                operator = mat.tensor(operator, mat.Z)
            else:
                operator = mat.tensor(operator, mat.I(2))
        operators += [operator]
    noisy_density_matrix = np.sum(
        [
            weight * operator @ density_matrix @ mat.H(operator)
            for weight, operator in zip(weights, operators)
        ],
        axis=0,
    )
    return noisy_density_matrix


def noisy_bp_dm(state, target_indices):
    """Compute the noisy density matrix of a targeted Bell pair.

    Parameters
    ----------
    state : State
        The state in the NSF of the noisy graph state containing a Bell pair.
    target_indices : list[int]
        Indices of the target qubits of the Bell pair. Counting starts at 0.

    Returns
    -------
    np.ndarray
        Noisy density matrix of the target Bell pair.

    Raises
    ------
    ValueError
        If the state of the `target_indices` cannot be reduced to a Bell pair.
        This is the case if either there is no edge between the
        `target_indices`, or there are other vertices connected to them.
    """
    adjacency_matrix = state.graph.adj
    if not len(target_indices) == 2:
        raise ValueError(
            f"Expected 2 target indices for Bell pair, got {len(target_indices)}."
        )
    if not adjacency_matrix[target_indices[0], target_indices[1]]:
        raise ValueError(
            f"Cannot be reduced to Bell pair. {state.graph} has no edge between "
            + f"{target_indices[0]} and {target_indices[1]}."
        )
    for target_index in target_indices:
        if np.sum(adjacency_matrix[target_index]) != 1:
            raise ValueError(
                f"Cannot be reduced to Bell pair. {state.graph} has excess edges "
                + f"connecting to vertex {target_index}."
            )
    reduced_maps = reduce_maps(state, target_indices)
    noisy_density_matrix = apply_nsf_maps_to_dm(
        reduced_maps, gt.bell_pair_dm, target_indices
    )
    return noisy_density_matrix


def noisy_3_ghz_dm(state, target_root, target_leafs):
    """Compute the noisy density matrix of a targeted 3-qubit GHZ state.

    This assumes the root-and-leafs representation of the graph state
    variant of the GHZ state. (which is LC-equivalent to the fully connected graph)

    Parameters
    ----------
    state : State
        The state in the NSF of the noisy graph state containing a 3-qubit GHZ state.
    target_root : int
        Index of the root qubit of the targeted 3-qubit GHZ state. Counting starts at 0.
    target_leafs : list[int]
        Indices of the target leaf qubits of the 3-qubit GHZ state. Counting starts at 0.

    Returns
    -------
    np.ndarray
        Noisy density matrix of the target 3-qubit GHZ state.


    Raises
    ------
    ValueError
        If the state of the `target_root` and `target_leafs` cannot
        represent a GHZ state.
        This is the case if there are edges missing in the graph
        or there are other extra connections to vertices outside
        the subset.

    """
    target_indices = [target_leafs[0]] + [target_root] + [target_leafs[1]]
    target_edges = [
        sorted([target_root, target_leafs[0]]),
        sorted([target_root, target_leafs[1]]),
    ]
    # check if graph is compatible with extracting a GHZ
    if tuple(target_edges[0]) not in state.graph.E:
        raise ValueError(
            f"Incompatible graph {state.graph} does not contain an edge {target_edges[0]}."
        )
    if tuple(target_edges[1]) not in state.graph.E:
        raise ValueError(
            f"Incompatible graph {state.graph} does not contain an edge {target_edges[1]}."
        )
    for edge in state.graph.E:
        if target_root in edge:
            if not (target_edges[0] == sorted(edge) or target_edges[1] == sorted(edge)):
                raise ValueError(
                    f"Incompatible graph {state.graph} contains extraneous edge {edge}."
                )
        if target_leafs[0] in edge:
            if not target_edges[0] == sorted(edge):
                raise ValueError(
                    f"Incompatible graph {state.graph} contains extraneous edge {edge}."
                )
        if target_leafs[1] in edge:
            if not target_edges[1] == sorted(edge):
                raise ValueError(
                    f"Incompatible graph {state.graph} contains extraneous edge {edge}."
                )
    reduced_maps = reduce_maps(state, target_indices)
    # Note that the order of the noiseless density matrix is leaf-root-leaf
    noisy_density_matrix = apply_nsf_maps_to_dm(
        reduced_maps, gt.ghz_3_dm, target_indices
    )
    return noisy_density_matrix
