import numpy as np
import graphepp as gg
import noisy_graph_states
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
from noisy_graph_states import State as NSFState
import noisy_graph_states.tools.density_matrix as dm
from noisy_graph_states.libs.graph import bell_pair_dm
from noisy_graph_states.tools.density_matrix import State as DMState


def test_x_noise():
    start_graph = gg.Graph(2, [(0, 1)])
    nsf_state = NSFState(start_graph, [])
    nsf_state = noisy_graph_states.x_noise(nsf_state, [0, 1], 1)
    nsf_dm_state = noisy_graph_states.noisy_bp_dm(nsf_state, [0, 1])
    dm_state = DMState(start_graph, bell_pair_dm)
    dm_state = dm.x_noise(dm_state, [0, 1], 1)
    assert np.allclose(nsf_dm_state, dm_state.density_matrix)


def test_y_noise():
    start_graph = gg.Graph(2, [(0, 1)])
    nsf_state = NSFState(start_graph, [])
    nsf_state = noisy_graph_states.y_noise(nsf_state, [1], 1)
    nsf_dm_state = noisy_graph_states.noisy_bp_dm(nsf_state, [0, 1])
    dm_state = DMState(start_graph, bell_pair_dm)
    dm_state = dm.y_noise(dm_state, [1], 1)
    assert np.allclose(nsf_dm_state, dm_state.density_matrix)


def test_z_noise():
    start_graph = gg.Graph(2, [(0, 1)])
    nsf_state = NSFState(start_graph, [])
    nsf_state = noisy_graph_states.z_noise(nsf_state, [1], 1)
    nsf_dm_state = noisy_graph_states.noisy_bp_dm(nsf_state, [0, 1])
    dm_state = DMState(start_graph, bell_pair_dm)
    dm_state = dm.z_noise(dm_state, [1], 1)
    # print(nsf_dm_state)
    # print(dm_state.density_matrix)
    assert np.allclose(nsf_dm_state, dm_state.density_matrix)


def test_pauli_noise():
    start_graph = gg.Graph(2, [(0, 1)])
    nsf_state = NSFState(start_graph, [])
    p = 0.75
    coefficients = [(1 + 3 * p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    nsf_state = noisy_graph_states.pauli_noise(nsf_state, [1], coefficients)
    nsf_dm_state = noisy_graph_states.noisy_bp_dm(nsf_state, [0, 1])
    dm_state = DMState(start_graph, bell_pair_dm)
    dm_state = dm.pauli_noise(dm_state, [1], coefficients)
    assert np.allclose(nsf_dm_state, dm_state.density_matrix)


def test_local_complementation():
    start_graph = gg.Graph(3, [(0, 1), (1, 2), (0, 2)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [1], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [1], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [1], 1)
    nsf_state_x = noisy_graph_states.local_complementation(nsf_state_x, 1)
    nsf_state_y = noisy_graph_states.local_complementation(nsf_state_y, 1)
    nsf_state_z = noisy_graph_states.local_complementation(nsf_state_z, 1)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [1], 1)
    dm_state_y = dm.y_noise(dm_state, [1], 1)
    dm_state_z = dm.z_noise(dm_state, [1], 1)
    dm_state_x = dm.local_complementation(dm_state_x, 1)
    dm_state_y = dm.local_complementation(dm_state_y, 1)
    dm_state_z = dm.local_complementation(dm_state_z, 1)

    assert np.allclose(nsf_dm_state_x, dm_state_x.density_matrix)
    assert np.allclose(nsf_dm_state_y, dm_state_y.density_matrix)
    assert np.allclose(nsf_dm_state_z, dm_state_z.density_matrix)


def test_z_measurement():
    start_graph = gg.Graph(4, [(0, 1), (1, 2), (2, 3)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [3], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [3], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [3], 1)
    nsf_state_x = noisy_graph_states.z_measurement(nsf_state_x, 3)
    nsf_state_y = noisy_graph_states.z_measurement(nsf_state_y, 3)
    nsf_state_z = noisy_graph_states.z_measurement(nsf_state_z, 3)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [3], 1)
    dm_state_y = dm.y_noise(dm_state, [3], 1)
    dm_state_z = dm.z_noise(dm_state, [3], 1)
    dm_state_x = dm.z_measurement(dm_state_x, 3)
    dm_state_y = dm.z_measurement(dm_state_y, 3)
    dm_state_z = dm.z_measurement(dm_state_z, 3)
    dm_state_x = mat.ptrace(dm_state_x.density_matrix, [3])
    dm_state_y = mat.ptrace(dm_state_y.density_matrix, [3])
    dm_state_z = mat.ptrace(dm_state_z.density_matrix, [3])

    assert np.allclose(nsf_dm_state_x, dm_state_x)
    assert np.allclose(nsf_dm_state_y, dm_state_y)
    assert np.allclose(nsf_dm_state_z, dm_state_z)


def test_y_measurement():
    start_graph = gg.Graph(4, [(0, 1), (1, 3), (3, 2)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [3], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [3], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [3], 1)
    nsf_state_x = noisy_graph_states.y_measurement(nsf_state_x, 3)
    nsf_state_y = noisy_graph_states.y_measurement(nsf_state_y, 3)
    nsf_state_z = noisy_graph_states.y_measurement(nsf_state_z, 3)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [3], 1)
    dm_state_y = dm.y_noise(dm_state, [3], 1)
    dm_state_z = dm.z_noise(dm_state, [3], 1)
    dm_state_x = dm.y_measurement(dm_state_x, 3)
    dm_state_y = dm.y_measurement(dm_state_y, 3)
    dm_state_z = dm.y_measurement(dm_state_z, 3)
    dm_state_x = mat.ptrace(dm_state_x.density_matrix, [3])
    dm_state_y = mat.ptrace(dm_state_y.density_matrix, [3])
    dm_state_z = mat.ptrace(dm_state_z.density_matrix, [3])

    assert np.allclose(nsf_dm_state_x, dm_state_x)
    assert np.allclose(nsf_dm_state_y, dm_state_y)
    assert np.allclose(nsf_dm_state_z, dm_state_z)


def test_x_measurement():
    start_graph = gg.Graph(4, [(0, 3), (1, 3), (3, 2)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [3], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [3], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [3], 1)
    nsf_state_x = noisy_graph_states.x_measurement(nsf_state_x, 3, 1)
    nsf_state_y = noisy_graph_states.x_measurement(nsf_state_y, 3, 1)
    nsf_state_z = noisy_graph_states.x_measurement(nsf_state_z, 3, 1)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [3], 1)
    dm_state_y = dm.y_noise(dm_state, [3], 1)
    dm_state_z = dm.z_noise(dm_state, [3], 1)
    dm_state_x = dm.x_measurement(dm_state_x, 3, 1)
    dm_state_y = dm.x_measurement(dm_state_y, 3, 1)
    dm_state_z = dm.x_measurement(dm_state_z, 3, 1)
    dm_state_x = mat.ptrace(dm_state_x.density_matrix, [3])
    dm_state_y = mat.ptrace(dm_state_y.density_matrix, [3])
    dm_state_z = mat.ptrace(dm_state_z.density_matrix, [3])

    assert np.allclose(nsf_dm_state_x, dm_state_x)
    assert np.allclose(nsf_dm_state_y, dm_state_y)
    assert np.allclose(nsf_dm_state_z, dm_state_z)


def test_cnot():
    start_graph = gg.Graph(3, [(0, 1)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [0], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [0], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [0], 1)
    nsf_state_x = noisy_graph_states.cnot(nsf_state_x, 2, 0)
    nsf_state_y = noisy_graph_states.cnot(nsf_state_y, 2, 0)
    nsf_state_z = noisy_graph_states.cnot(nsf_state_z, 2, 0)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [0], 1)
    dm_state_y = dm.y_noise(dm_state, [0], 1)
    dm_state_z = dm.z_noise(dm_state, [0], 1)
    dm_state_x = dm.cnot(dm_state_x, 2, 0)
    dm_state_y = dm.cnot(dm_state_y, 2, 0)
    dm_state_z = dm.cnot(dm_state_z, 2, 0)

    assert np.allclose(nsf_dm_state_x, dm_state_x.density_matrix)
    assert np.allclose(nsf_dm_state_y, dm_state_y.density_matrix)
    assert np.allclose(nsf_dm_state_z, dm_state_z.density_matrix)


def test_merge():
    start_graph = gg.Graph(4, [(0, 1), (3, 2)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [3], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [3], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [3], 1)
    nsf_state_x = noisy_graph_states.merge(nsf_state_x, 1, 3)
    nsf_state_y = noisy_graph_states.merge(nsf_state_y, 1, 3)
    nsf_state_z = noisy_graph_states.merge(nsf_state_z, 1, 3)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [3], 1)
    dm_state_y = dm.y_noise(dm_state, [3], 1)
    dm_state_z = dm.z_noise(dm_state, [3], 1)
    dm_state_x = dm.merge(dm_state_x, 1, 3)
    dm_state_y = dm.merge(dm_state_y, 1, 3)
    dm_state_z = dm.merge(dm_state_z, 1, 3)
    dm_state_x = mat.ptrace(dm_state_x.density_matrix, [3])
    dm_state_y = mat.ptrace(dm_state_y.density_matrix, [3])
    dm_state_z = mat.ptrace(dm_state_z.density_matrix, [3])

    assert np.allclose(nsf_dm_state_x, dm_state_x)
    assert np.allclose(nsf_dm_state_y, dm_state_y)
    assert np.allclose(nsf_dm_state_z, dm_state_z)


def test_full_merge():
    start_graph = gg.Graph(5, [(0, 1), (1, 4), (3, 2)])
    nsf_state = NSFState(start_graph, [])
    nsf_state_x = noisy_graph_states.x_noise(nsf_state, [3], 1)
    nsf_state_y = noisy_graph_states.y_noise(nsf_state, [3], 1)
    nsf_state_z = noisy_graph_states.z_noise(nsf_state, [3], 1)
    nsf_state_x = noisy_graph_states.full_merge(nsf_state_x, 4, 3)
    nsf_state_y = noisy_graph_states.full_merge(nsf_state_y, 4, 3)
    nsf_state_z = noisy_graph_states.full_merge(nsf_state_z, 4, 3)
    nsf_dm_state_x = noisy_graph_states.noisy_3_ghz_dm(nsf_state_x, 1, [0, 2])
    nsf_dm_state_y = noisy_graph_states.noisy_3_ghz_dm(nsf_state_y, 1, [0, 2])
    nsf_dm_state_z = noisy_graph_states.noisy_3_ghz_dm(nsf_state_z, 1, [0, 2])

    start_dm = gt.graph_state(start_graph) @ mat.H(gt.graph_state(start_graph))
    dm_state = DMState(start_graph, start_dm)
    dm_state_x = dm.x_noise(dm_state, [3], 1)
    dm_state_y = dm.y_noise(dm_state, [3], 1)
    dm_state_z = dm.z_noise(dm_state, [3], 1)
    dm_state_x = dm.full_merge(dm_state_x, 4, 3)
    dm_state_y = dm.full_merge(dm_state_y, 4, 3)
    dm_state_z = dm.full_merge(dm_state_z, 4, 3)
    dm_state_x = mat.ptrace(dm_state_x.density_matrix, [3, 4])
    dm_state_y = mat.ptrace(dm_state_y.density_matrix, [3, 4])
    dm_state_z = mat.ptrace(dm_state_z.density_matrix, [3, 4])

    assert np.allclose(nsf_dm_state_x, dm_state_x)
    assert np.allclose(nsf_dm_state_y, dm_state_y)
    assert np.allclose(nsf_dm_state_z, dm_state_z)
