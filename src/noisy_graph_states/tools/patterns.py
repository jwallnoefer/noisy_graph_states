# -*- coding: utf-8 -*-
"""
A collection of useful functions to translate between measurement patterns and strategies.
"""

import graphepp as gg
from itertools import permutations
from collections import defaultdict
import noisy_graph_states.libs.graph as gt


# Before using the function in this file. We define the concepts of sequence and pattern.
# sequence: ordered list of tuples ("z", index), ("y", index) or ("x", index, b0) where b0 is optional and automatically
# determined if not given.
# pattern: list of strings with "x" "y" "z" or "." (no measurement)


def _default_projector_matching():
    return {"x": "x", "y": "y", "z": "z"}


def sequence_to_pattern(sequence: list[tuple], graph: gg.Graph):
    """Find measurement pattern corresponding to graph measurements.

    Every sequence of measurements in the graph state picture (i.e.,
    with explicitly undoing the byproduct operators at every step)
    corresponds to a measurement pattern of Pauli measurements
    that postpones all corrections operations to the very end.
    There are multiple sequences that lead to the same
    pattern.

    | This is explained in the appendix of
    | Maria Flors Mor-Ruiz, Julius Wallnöfer, and Wolfgang Dür.
    | "Imperfect quantum networks with tailored resource states."
    | arXiv preprint arXiv:2403.19778 (2024).

    Parameters
    ----------
    sequence : list[tuple]
        A series of graph measurements that are performed in order.
        Format: ("x" or "y" or "z", qubit_index) for x measurements
        optionally one can specify the index of the special neighbour
        like so ("x", qubit_index, b0).
    graph : gg.Graph
        The start graph to which the sequence of measurements is applied.


    Returns
    -------
    list[str]
        The measurement pattern for postponed correction operations from
        the lowest qubit index to the highest qubit index.
        "x", "y" or "z" for a Pauli measurement. "." for no measurement.
    """
    matchings = defaultdict(_default_projector_matching)
    pattern = ["."] * graph.N
    for instruction in sequence:
        instruction_type = instruction[0]
        qubit_index = instruction[1]
        pattern[qubit_index] = matchings[qubit_index][instruction_type]
        if instruction_type == "z":
            # no change in projector matchings required
            graph = gt.measure_z(graph=graph, index=qubit_index)
        elif instruction_type == "y":
            neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
            for neighbour in neighbours:
                match = matchings[neighbour]
                tmp = match["x"]
                match["x"] = match["y"]
                match["y"] = tmp
            graph = gt.measure_y(graph=graph, index=qubit_index)
        elif instruction_type == "x":
            try:
                b0 = instruction[2]
            except IndexError:
                b0 = None
            if b0 is None:
                neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
                try:
                    b0 = neighbours[0]
                except IndexError:
                    b0 = None
                if b0 is not None:
                    match = matchings[b0]
                    tmp = match["x"]
                    match["x"] = match["z"]
                    match["z"] = tmp
            graph = gt.measure_x(graph=graph, index=qubit_index, b0=b0)
    return pattern


def pattern_to_sequence(pattern: list[str], graph: gg.Graph):
    """Find a sequence of graph measurements corresponding to a measurement pattern.

    Since multiple sequences can lead to the same pattern, this
    just returns one particular one (the one in which the graph
    transformations are from the lowest qubit index to the
    highest qubit index in order).
    Use `pattern_to_all_sequences` instead to get a list of all.

    | This is explained in the appendix of
    | Maria Flors Mor-Ruiz, Julius Wallnöfer, and Wolfgang Dür.
    | "Imperfect quantum networks with tailored resource states."
    | arXiv preprint arXiv:2403.19778 (2024).

    Parameters
    ----------
    pattern : list[str]
        The measurement pattern from the lowest qubit index to the highest qubit index.
        "x", "y" or "z" for a Pauli measurement. "." for no measurement.
        length must match `graph.N`
    graph : gg.Graph
        The start graph to which the measurements are applied.

    Returns
    -------
    list[tuple]
        A sequence of graph measurements.
        Format: ("x" or "y" or "z", qubit_index)
    """
    assert len(pattern) == graph.N
    matchings = defaultdict(_default_projector_matching)
    sequence = []
    for qubit_index, proj in enumerate(pattern):
        if proj == ".":
            continue
        effective_instruction = matchings[qubit_index][proj]
        sequence.append((effective_instruction, qubit_index))
        if effective_instruction == "z":
            # no change in projector matchings required
            graph = gt.measure_z(graph=graph, index=qubit_index)
        elif effective_instruction == "y":
            neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
            for neighbour in neighbours:
                match = matchings[neighbour]
                for k, v in match.items():
                    if v == "x":
                        match[k] = "y"
                    elif v == "y":
                        match[k] = "x"
            graph = gt.measure_y(graph=graph, index=qubit_index)
        elif effective_instruction == "x":
            neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
            try:
                b0 = neighbours[0]
            except IndexError:
                b0 = None
            if b0 is not None:
                match = matchings[b0]
                for k, v in match.items():
                    if v == "x":
                        match[k] = "z"
                    elif v == "z":
                        match[k] = "x"
            graph = gt.measure_x(graph=graph, index=qubit_index, b0=b0)
    return tuple(sequence)


def pattern_to_all_sequences(pattern: list[str], graph: gg.Graph):
    """Find a sequence of graph measurements corresponding to a measurement pattern.

    Multiple sequences can lead to the same pattern, this function
    finds all of them, except those that would arise from specifying
    the special neighbour b0 for x measurements explicitly.
    Use `pattern_to_sequence` instead if you just need a single
    sequence as representation of the pattern.

    | This is explained in the appendix of
    | Maria Flors Mor-Ruiz, Julius Wallnöfer, and Wolfgang Dür.
    | "Imperfect quantum networks with tailored resource states."
    | arXiv preprint arXiv:2403.19778 (2024).

    Parameters
    ----------
    pattern : list[str]
        The measurement pattern from the lowest qubit index to the highest qubit index.
        "x", "y" or "z" for a Pauli measurement. "." for no measurement.
        length must match `graph.N`
    graph : gg.Graph
        The start graph to which the measurements are applied.

    Returns
    -------
    list[tuple]
        A sequence of graph measurements.
        Format: ("x" or "y" or "z", qubit_index)
    """
    assert len(pattern) == graph.N
    relevant_measurements = [
        (idx, proj) for idx, proj in enumerate(pattern) if proj != "."
    ]
    start_graph = graph
    sequences = []
    for chosen_order in permutations(relevant_measurements):
        graph = start_graph
        matchings = defaultdict(_default_projector_matching)
        sequence = []
        for qubit_index, proj in chosen_order:
            if proj == ".":
                continue
            effective_instruction = matchings[qubit_index][proj]
            sequence.append((effective_instruction, qubit_index))
            if effective_instruction == "z":
                # no change in projector matchings required
                graph = gt.measure_z(graph=graph, index=qubit_index)
            elif effective_instruction == "y":
                neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
                for neighbour in neighbours:
                    match = matchings[neighbour]
                    for k, v in match.items():
                        if v == "x":
                            match[k] = "y"
                        elif v == "y":
                            match[k] = "x"
                graph = gt.measure_y(graph=graph, index=qubit_index)
            elif effective_instruction == "x":
                neighbours = gt.neighbourhood(graph=graph, index=qubit_index)
                try:
                    b0 = neighbours[0]
                except IndexError:
                    b0 = None
                if b0 is not None:
                    match = matchings[b0]
                    for k, v in match.items():
                        if v == "x":
                            match[k] = "z"
                        elif v == "z":
                            match[k] = "x"
                graph = gt.measure_x(graph=graph, index=qubit_index, b0=b0)
        sequences.append(sequence)
    return sequences
