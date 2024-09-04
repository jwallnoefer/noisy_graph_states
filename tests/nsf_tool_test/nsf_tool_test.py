import graphepp as gg
import numpy as np
import pytest

from noisy_graph_states.libs.graph import bell_pair_dm
from noisy_graph_states.libs.graph import ghz_3_dm
import noisy_graph_states.libs.matrix as mat
import noisy_graph_states
from noisy_graph_states import State as NSFState
from noisy_graph_states import Map, compile_maps
from itertools import permutations
from functools import lru_cache
import random


@lru_cache
def _get_all_possible_combinations(max_number_vertices):
    select_from = [i for i in range(max_number_vertices)]
    possible_combinations = [
        list(permutations(select_from, i)) for i in range(1, max_number_vertices + 1)
    ]
    possible_combinations = [
        list(element)
        for combination_list in possible_combinations
        for element in combination_list
    ]
    return possible_combinations


def _get_random_weights(length):
    random_weights = np.random.random(length + 1)
    random_weights = random_weights / np.sum(
        random_weights
    )  # normalize to make it an actual channel
    random_weights = list(
        random_weights[1:]
    )  # because the identity weight is not stored
    return random_weights


def _get_random_map(max_number_vertices, length):
    # this is just for test purposes, it is not an equal distribution of all possible maps or anything like that
    # it also deliberately does not give nicely sorted outputs and may include duplicates
    possible_combinations = _get_all_possible_combinations(max_number_vertices)
    random_noises = [random.choice(possible_combinations) for i in range(length)]
    random_map = Map(weights=_get_random_weights(length), noises=random_noises)
    return random_map


def test_map_to_standard_form():
    # we define the standard form as follows
    # * each noise is sorted
    # * noises are sorted
    # * no duplicate noises
    # * no empty noises
    # * no noises with weight 0
    # identity map should obviously work
    test_map = Map(weights=[], noises=[])
    test_map.to_standard_form()
    assert test_map.weights == []
    assert test_map.noises == []
    # individual noise is given in wrong order
    weights = _get_random_weights(1)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[1, 0]])
    test_map.to_standard_form()
    assert test_map.weights == weights  # weight does not change
    assert test_map.noises == [[0, 1]]
    weights = _get_random_weights(2)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[1, 0], [2, 0]])
    test_map.to_standard_form()
    assert test_map.weights == weights  # weights do not change
    assert test_map.noises == [[0, 1], [0, 2]]
    # noises are given in wrong order
    weights = _get_random_weights(2)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[0, 1], [0]])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[1],
        weights[0],
    ]  # weights get reordered in same manner as noises
    assert test_map.noises == [[0], [0, 1]]
    weights = _get_random_weights(2)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[0, 2], [0, 1]])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[1],
        weights[0],
    ]  # weights get reordered in same manner as noises
    assert test_map.noises == [[0, 1], [0, 2]]
    # a duplicate is given
    weights = _get_random_weights(2)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[0], [0]])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[0] + weights[1]
    ]  # weights of duplicates get added
    assert test_map.noises == [[0]]
    weights = _get_random_weights(4)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[0, 1, 2, 4], [2], [0, 1, 2, 4], [3]])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[0] + weights[2],
        weights[1],
        weights[3],
    ]  # weights of duplicates get added
    assert test_map.noises == [[0, 1, 2, 4], [2], [3]]
    # an empty noise is given
    weights = _get_random_weights(2)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[0], []])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[0]
    ]  # weight of empty noise is dropped, others stay the same
    assert test_map.noises == [[0]]
    weights = _get_random_weights(3)
    test_weights = list(weights)
    test_map = Map(weights=test_weights, noises=[[], [0], []])
    test_map.to_standard_form()
    assert test_map.weights == [
        weights[1]
    ]  # weight of empty noise is dropped, others stay the same
    assert test_map.noises == [[0]]
    # a noise with weight 0 is given
    test_weight = np.random.random()
    test_map = Map(weights=[0, test_weight, 0], noises=[[0], [1], [0, 1]])
    test_map.to_standard_form()
    assert test_map.weights == [test_weight]
    assert test_map.noises == [[1]]
    # test the 5 criterions on random maps
    for num_vertices in range(1, 8):
        for noises_length in range(20):
            for i in range(10):
                test_map = _get_random_map(
                    max_number_vertices=num_vertices, length=noises_length
                )
                test_map.to_standard_form()
                # each noise is sorted
                for noise in test_map.noises:
                    assert sorted(noise) == noise
                # noises themselves are sorted
                assert sorted(test_map.noises) == test_map.noises
                # no duplicate noises
                for idx, noise in enumerate(test_map.noises):
                    assert noise not in test_map.noises[idx + 1 :]
                # no empty noises
                assert [] not in test_map.noises
                # no zero weight noises (this should never happen anyway because of how the random maps are generated)
                assert 0 not in test_map.weights


def test_map_equality():
    #  every ambiguity that is also removed by the standard form should not matter for equality either
    #  there is one additional requirement for the equality check though:
    #  floating point weights that are close to 0 should give sensible results
    # start with: any map should be equal to itself and to its standard form
    identity_map = Map(weights=[], noises=[])
    assert identity_map == identity_map
    assert identity_map == identity_map.as_standard_form()
    map_with_zero_weights = Map(
        weights=[0, 0.2, 0], noises=[[0, 1], [1, 3, 4, 5, 6], [3]]
    )
    assert map_with_zero_weights == map_with_zero_weights
    assert map_with_zero_weights == map_with_zero_weights.as_standard_form()
    assert map_with_zero_weights != identity_map
    map_with_empty_noises = Map(weights=[0.1, 0.2, 0.3], noises=[[0, 1], [], []])
    assert map_with_empty_noises == map_with_empty_noises
    assert map_with_empty_noises == map_with_empty_noises.as_standard_form()
    assert map_with_empty_noises != identity_map
    map_with_zero_and_empty = Map(weights=[0.1, 0.2, 0], noises=[[0, 1], [], []])
    assert map_with_zero_and_empty == map_with_zero_and_empty
    assert map_with_zero_and_empty == map_with_zero_and_empty.as_standard_form()
    assert map_with_zero_and_empty != identity_map
    for num_vertices in range(1, 8):
        for noises_length in range(20):
            for i in range(10):
                test_map = _get_random_map(
                    max_number_vertices=num_vertices, length=noises_length
                )
                assert test_map == test_map
                assert test_map == test_map.as_standard_form()
    # now the ambiguity tests (these could be made a bit more general, though)
    # noise specification should not matter
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2], noises=[[1, 0], [4, 1, 3, 6, 5]])
    assert map1 == map2
    # order of noises should not matter
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.2, 0.1], noises=[[1, 3, 4, 5, 6], [0, 1]])
    assert map1 == map2
    # specification with and without duplicates should be equivalent
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.05, 0.2, 0.05], noises=[[0, 1], [1, 3, 4, 5, 6], [0, 1]])
    assert map1 == map2
    # adding empty noises should change nothing
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.3, 0.1, 0.2, 0.4], noises=[[], [0, 1], [1, 3, 4, 5, 6], []])
    assert map1 == map2
    # adding noises with weight 0 should change nothing
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(
        weights=[0.1, 0.2, 0, 0], noises=[[0, 1], [1, 3, 4, 5, 6], [2, 4, 5], [3, 2, 1]]
    )
    assert map1 == map2
    # same maps with different weights should be unequal
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.3, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    assert map1 != map2
    # maps with same weights but different noises be unequal
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2], noises=[[0, 2], [1, 2, 5, 6]])
    assert map1 != map2
    # but adding a non-empty noise with non-zero weight that is not a duplicate, should matter
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2, 0.3], noises=[[0, 1], [1, 3, 4, 5, 6], [0]])
    assert map1 != map2
    # common floating point problems should not break the equality:
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2 + 1e-15], noises=[[0, 1], [1, 3, 4, 5, 6]])
    assert map1 == map2
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2, 1e-15], noises=[[0, 1], [1, 3, 4, 5, 6], [2]])
    assert map1 == map2
    # but big enough differences should still show up, of course
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2 + 1e-10], noises=[[0, 1], [1, 3, 4, 5, 6]])
    assert map1 != map2
    map1 = Map(weights=[0.1, 0.2], noises=[[0, 1], [1, 3, 4, 5, 6]])
    map2 = Map(weights=[0.1, 0.2, 1e-10], noises=[[0, 1], [1, 3, 4, 5, 6], [2]])
    assert map1 != map2


def test_state_equality():
    # different sets of maps could still lead to the same state, which should be accounted for
    test_graph = gg.Graph(4, [(0, 1), (0, 2), (0, 3)])  # 4-qubit GHZ graph
    epsilon = 1e-4
    equivalent_F = (1 - epsilon) ** 3 + epsilon**3
    wnoise0_map = Map(
        [(1 - equivalent_F) / 3] * 3, [[1, 2, 3], [0, 1, 2, 3], [0]]
    )  # white noise on qubit 0
    wnoise_state = NSFState(graph=test_graph, maps=[wnoise0_map])
    xnoise0_map = Map([epsilon], [[1, 2, 3]])
    ynoise0_map = Map([epsilon], [[0, 1, 2, 3]])
    znoise0_map = Map([epsilon], [[0]])
    individual_state = NSFState(
        graph=test_graph, maps=[xnoise0_map, ynoise0_map, znoise0_map]
    )
    assert wnoise_state == individual_state
    # but should say unequal for wrong weights
    wnoise0_map = Map(
        [epsilon] * 3, [[1, 2, 3], [0, 1, 2, 3], [0]]
    )  # white noise on qubit 0
    wnoise_state = NSFState(graph=test_graph, maps=[wnoise0_map])
    individual_state = individual_state
    assert wnoise_state != individual_state
    # different graphs should lead to unequal, even if the physical meaning is similar
    other_test_graph = gg.Graph(
        5, [(0, 1), (0, 2), (0, 3)]
    )  # adds one disconnected qubit
    wnoise0_map = Map(
        [epsilon] * 3, [[1, 2, 3], [0, 1, 2, 3], [0]]
    )  # white noise on qubit 0
    state1 = NSFState(graph=test_graph, maps=[wnoise0_map])
    state2 = NSFState(graph=other_test_graph, maps=[wnoise0_map])
    assert state1 != state2


def test_compile_maps():
    # for single maps in standard form, it should just bring them to standard form and nothing more
    for num_vertices in range(1, 8):
        for noises_length in range(20):
            for i in range(10):
                test_map = _get_random_map(num_vertices, noises_length)
                assert test_map == compile_maps(test_map)
    # check if combination of two maps follows established rules
    # same one-length noises recombine
    for num_vertices in range(1, 8):
        for i in range(10):
            map1 = _get_random_map(num_vertices, length=1)
            map2 = Map(_get_random_weights(length=1), noises=map1.noises)
            w1 = map1.weights[0]
            w2 = map2.weights[0]
            expected_w = (1 - w2) * w1 + (1 - w1) * w2
            expected_map = Map([expected_w], noises=map1.noises)
            assert compile_maps(map1, map2) == expected_map
    # different noises combine as expected
    map1 = Map(_get_random_weights(length=1), noises=[[0, 1]])
    map2 = Map(_get_random_weights(length=1), noises=[[2, 3]])
    w1 = map1.weights[0]
    w2 = map2.weights[0]
    expected_map = Map(
        weights=[(1 - w2) * w1, (1 - w1) * w2, w1 * w2],
        noises=[[0, 1], [2, 3], [0, 1, 2, 3]],
    )
    assert compile_maps(map1, map2) == expected_map
    map1 = Map(_get_random_weights(length=1), noises=[[0, 1]])
    map2 = Map(_get_random_weights(length=1), noises=[[1, 3]])
    w1 = map1.weights[0]
    w2 = map2.weights[0]
    expected_map = Map(
        weights=[(1 - w2) * w1, (1 - w1) * w2, w1 * w2], noises=[[0, 1], [1, 3], [0, 3]]
    )
    assert compile_maps(map1, map2) == expected_map
    # test commutativity
    for num_vertices in range(1, 8):
        max_noise_length = 20
        for i in range(20):
            map1 = _get_random_map(
                num_vertices, length=random.randint(0, max_noise_length)
            )
            map2 = _get_random_map(
                num_vertices, length=random.randint(0, max_noise_length)
            )
            assert compile_maps(map1, map2) == compile_maps(map2, map1)
    # test associativity
    for num_vertices in range(1, 8):
        max_noise_length = 20
        for i in range(20):
            map1 = _get_random_map(
                num_vertices, length=random.randint(0, max_noise_length)
            )
            map2 = _get_random_map(
                num_vertices, length=random.randint(0, max_noise_length)
            )
            map3 = _get_random_map(
                num_vertices, length=random.randint(0, max_noise_length)
            )
            assert (
                compile_maps(map1, map2, map3)
                == compile_maps(map1, compile_maps(map2, map3))
                == compile_maps(compile_maps(map1, map2), map3)
            )


def test_x_noise():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    final_state = noisy_graph_states.x_noise(start_state, [1], 1)
    target_maps = [Map(weights=[1], noises=[[0, 2, 3]])]
    target_state = NSFState(start_graph, target_maps)
    assert final_state == target_state


def test_y_noise():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    final_state = noisy_graph_states.y_noise(start_state, [1], 1)
    target_maps = [Map(weights=[1], noises=[[0, 1, 2, 3]])]
    target_state = NSFState(start_graph, target_maps)
    assert final_state == target_state


def test_z_noise():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    final_state = noisy_graph_states.z_noise(start_state, [1], 1)
    target_maps = [Map(weights=[1], noises=[[1]])]
    target_state = NSFState(start_graph, target_maps)
    assert final_state == target_state


def test_pauli_noise():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    final_state = noisy_graph_states.pauli_noise(
        start_state, [1], (0, 1 / 3, 1 / 2, 1 / 6)
    )
    target_maps = [
        Map(weights=[1 / 3, 1 / 2, 1 / 6], noises=[[0, 2, 3], [0, 1, 2, 3], [1]])
    ]
    target_state = NSFState(start_graph, target_maps)
    assert final_state == target_state


def test_local_complementation():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [1], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [1], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [1], 1)
    final_state_x = noisy_graph_states.local_complementation(start_state_x, 1)
    final_state_y = noisy_graph_states.local_complementation(start_state_y, 1)
    final_state_z = noisy_graph_states.local_complementation(start_state_z, 1)
    target_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3), (0, 2), (0, 3), (2, 3)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [0, 2, 3], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [1], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [0, 1, 2, 3], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_z_measurement():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [1], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [1], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [1], 1)
    final_state_x = noisy_graph_states.z_measurement(start_state_x, 1)
    final_state_y = noisy_graph_states.z_measurement(start_state_y, 1)
    final_state_z = noisy_graph_states.z_measurement(start_state_z, 1)
    target_graph = gg.Graph(4, [])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [0, 2, 3], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [0, 2, 3], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_y_measurement():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [1], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [1], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [1], 1)
    final_state_x = noisy_graph_states.y_measurement(start_state_x, 1)
    final_state_y = noisy_graph_states.y_measurement(start_state_y, 1)
    final_state_z = noisy_graph_states.y_measurement(start_state_z, 1)
    target_graph = gg.Graph(4, [(0, 2), (2, 3), (0, 3)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [0, 2, 3], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [0, 2, 3], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_x_measurement():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (1, 3)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [1], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [1], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [1], 1)
    final_state_x = noisy_graph_states.x_measurement(start_state_x, 1)
    final_state_y = noisy_graph_states.x_measurement(start_state_y, 1)
    final_state_z = noisy_graph_states.x_measurement(start_state_z, 1)
    target_graph = gg.Graph(4, [(0, 2), (0, 3)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [0], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [0], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_cnot():
    start_graph = gg.Graph(5, [(0, 1), (1, 2), (3, 4)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [3], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [3], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [3], 1)
    final_state_x = noisy_graph_states.cnot(start_state_x, source=1, target=3)
    final_state_y = noisy_graph_states.cnot(start_state_y, source=1, target=3)
    final_state_z = noisy_graph_states.cnot(start_state_z, source=1, target=3)
    target_graph = gg.Graph(5, [(0, 1), (1, 2), (3, 4), (1, 4)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [4], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [1, 3, 4], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [1, 3], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_merge():
    start_graph = gg.Graph(5, [(0, 1), (1, 2), (3, 4)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [3], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [3], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [3], 1)
    final_state_x = noisy_graph_states.merge(start_state_x, source=1, target=3)
    final_state_y = noisy_graph_states.merge(start_state_y, source=1, target=3)
    final_state_z = noisy_graph_states.merge(start_state_z, source=1, target=3)
    target_graph = gg.Graph(5, [(0, 1), (1, 2), (1, 4)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [4], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [1, 4], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [1], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_full_merge():
    start_graph = gg.Graph(5, [(0, 1), (1, 2), (3, 4)])
    start_state = NSFState(start_graph, [])
    start_state_x = noisy_graph_states.x_noise(start_state, [3], 1)
    start_state_y = noisy_graph_states.y_noise(start_state, [3], 1)
    start_state_z = noisy_graph_states.z_noise(start_state, [3], 1)
    final_state_x = noisy_graph_states.full_merge(start_state_x, source=1, target=3)
    final_state_y = noisy_graph_states.full_merge(start_state_y, source=1, target=3)
    final_state_z = noisy_graph_states.full_merge(start_state_z, source=1, target=3)
    target_graph = gg.Graph(5, [(0, 2), (0, 4), (2, 4)])
    target_state = NSFState(target_graph, [])
    target_state_x = noisy_graph_states.z_noise(target_state, [4], 1)
    target_state_y = noisy_graph_states.z_noise(target_state, [0, 2], 1)
    target_state_z = noisy_graph_states.z_noise(target_state, [0, 2, 4], 1)
    assert final_state_x == target_state_x
    assert final_state_y == target_state_y
    assert final_state_z == target_state_z


def test_reduce_maps():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (2, 3)])
    start_state = NSFState(start_graph, [])
    p = 0.75
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, [0, 1, 2, 3], depolarizing_weights
    )
    final_state = noisy_graph_states.z_measurement(start_state, 2)
    reduced_maps = noisy_graph_states.reduce_maps(final_state, [0, 1])
    reduced_map = noisy_graph_states.compile_maps(*reduced_maps)
    target_map_1 = Map(weights=depolarizing_weights[1:], noises=[[0], [1], [0, 1]])
    target_map_2 = Map(weights=[2 * depolarizing_weights[1]], noises=[[1]])
    target_map = noisy_graph_states.compile_maps(
        *[target_map_1, target_map_1, target_map_2]
    )
    assert reduced_map == target_map

    # the following example led to the discovery of a bug in reduce_maps
    example_map = noisy_graph_states.Map(
        weights=[0.0025, 0.0025, 0.0025], noises=[(1, 10), (0, 1, 10), (0,)]
    )
    example_state = NSFState(gg.Graph(N=100, E=()), maps=[example_map])
    reduced_map = noisy_graph_states.reduce_maps(
        example_state, target_indices=[12, 67]
    )[0]
    known_result = noisy_graph_states.Map([], [])
    assert reduced_map == known_result

    # just to make sure, another example with explicitly cutting a Bell pair out of a bigger 1d cluster
    N = 8
    start_graph = gg.Graph(N, E=[(i, i + 1) for i in range(N - 1)])
    start_state = NSFState(start_graph, [])
    p = 0.75
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, list(range(N)), depolarizing_weights
    )
    final_state = noisy_graph_states.z_measurement(
        noisy_graph_states.z_measurement(start_state, 2), 5
    )
    reduced_maps = noisy_graph_states.reduce_maps(final_state, [3, 4])
    reduced_map = noisy_graph_states.compile_maps(*reduced_maps)
    target_map_1 = Map(weights=depolarizing_weights[1:], noises=[[3], [4], [3, 4]])
    target_map_2 = Map(weights=[2 * depolarizing_weights[1]], noises=[[3]])
    target_map_3 = Map(weights=[2 * depolarizing_weights[1]], noises=[[4]])
    target_map = noisy_graph_states.compile_maps(
        *[target_map_1, target_map_1, target_map_2, target_map_3]
    )
    assert reduced_map == target_map


def test_apply_nsf_maps_to_dm():
    density_matrix = bell_pair_dm
    start_graph = gg.Graph(2, [(0, 1)])
    start_state = NSFState(start_graph, [])
    start_state = noisy_graph_states.x_noise(start_state, [0], 1)
    final_density_matrix = noisy_graph_states.apply_nsf_maps_to_dm(
        start_state.maps, density_matrix, [0, 1]
    )
    target_noise = mat.tensor(mat.X, mat.I(2))
    target_density_matrix = target_noise @ density_matrix @ mat.H(target_noise)
    assert np.allclose(final_density_matrix, target_density_matrix)


def test_noisy_bp_dm():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (2, 3)])
    start_state = NSFState(start_graph, [])
    p = 0.75
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, [0, 1, 2, 3], depolarizing_weights
    )
    final_state = noisy_graph_states.z_measurement(start_state, 2)
    final_dm = noisy_graph_states.noisy_bp_dm(final_state, [0, 1])
    target_map_1 = Map(weights=depolarizing_weights[1:], noises=[[0], [1], [0, 1]])
    target_map_2 = Map(weights=[2 * depolarizing_weights[1]], noises=[[1]])
    target_maps = [target_map_1, target_map_1, target_map_2]
    target_dm = noisy_graph_states.apply_nsf_maps_to_dm(
        target_maps, bell_pair_dm, [0, 1]
    )
    # print(final_dm, target_dm)
    assert np.allclose(final_dm, target_dm)


def test_invalid_bp_dm():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (2, 3)])
    start_state = NSFState(start_graph, [])
    p = 0.99
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, [0, 1, 2, 3], depolarizing_weights
    )
    with pytest.raises(ValueError):  # on edge not present
        noisy_graph_states.noisy_bp_dm(start_state, [0, 3])
    with pytest.raises(ValueError):  # on not isolated
        noisy_graph_states.noisy_bp_dm(start_state, [0, 1])
    start_graph = gg.Graph(4, [(1, 2)])
    start_state = NSFState(start_graph, [])
    p = 0.99
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, [0, 1, 2, 3], depolarizing_weights
    )
    with pytest.raises(ValueError):  # especially on completely disconnected
        noisy_graph_states.noisy_bp_dm(start_state, [0, 3])


def test_noisy_3_ghz_dm():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (2, 3)])
    start_state = NSFState(start_graph, [])
    p = 0.75
    depolarizing_weights = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    start_state = noisy_graph_states.pauli_noise(
        start_state, [0, 1, 2, 3], depolarizing_weights
    )
    final_state = noisy_graph_states.z_measurement(start_state, 3)
    final_dm = noisy_graph_states.noisy_3_ghz_dm(final_state, 1, [0, 2])
    map_leaf_0 = Map(weights=depolarizing_weights[1:], noises=[(0,), (1,), (0, 1)])
    map_leaf_2 = Map(weights=depolarizing_weights[1:], noises=[(1,), (2,), (1, 2)])
    map_root = Map(weights=depolarizing_weights[1:], noises=[(1,), (0, 2), (0, 1, 2)])
    map_measured = Map(weights=[2 * depolarizing_weights[1]], noises=[(2,)])
    target_maps = [map_leaf_0, map_leaf_2, map_root, map_measured]
    target_dm = noisy_graph_states.apply_nsf_maps_to_dm(
        target_maps, ghz_3_dm, [0, 1, 2]
    )
    assert np.allclose(final_dm, target_dm)
