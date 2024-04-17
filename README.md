# Noisy graph states

[![PyPI](https://img.shields.io/pypi/v/noisy-graph-states)](https://pypi.org/project/noisy-graph-states/)
[![Documentation Status](https://readthedocs.org/projects/noisy-graph-states/badge/?version=latest)](https://noisy-graph-states.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/jwallnoefer/noisy_graph_states/actions/workflows/ci.yaml/badge.svg)](https://github.com/jwallnoefer/noisy_graph_states/actions/workflows/ci.yaml)
[![DOI](https://zenodo.org/badge/696296923.svg)](https://zenodo.org/doi/10.5281/zenodo.10625616)


This python package is a tool to track how noisy graph
states transform under operations and measurements
(for an introduction to graph states see e.g.
[arXiv:quant-ph/0602096](https://arxiv.org/abs/quant-ph/0602096)).
It uses the Noisy Stabilizer Formalism introduced in


> Noisy stabilizer formalism <br>
> M. F. Mor-Ruiz, W. Dür <br>
> Phys. Rev. A **107**, 032424 (2023); DOI: [10.1103/PhysRevA.107.032424](https://doi.org/10.1103/PhysRevA.107.032424) <br>
> Preprint: [arXiv:2212.08677 [quant-ph]](https://doi.org/10.48550/arXiv.2212.08677)

that describes how Pauli-diagonal noise on graph states
transforms under various graph operations, such as
local complementation, Pauli measurements and merging
operations.

## Installation

You can install the package into your Python environment
from the Python Package Index:

```
pip install noisy-graph-states
```
As with all Python packages this can
possibly overwrite already installed package versions in your
environment with its dependencies, so installing it in a
dedicated virtual environment may be preferable.

If you encounter any problems, you can try installing the
exact versions of the dependencies of this package, which
were used to develop it (specified in `Pipfile.lock`).
This assume Python 3.9 and `pipenv` are available on your system.
```
git clone https://github.com/jwallnoefer/noisy_graph_states.git
cd noisy_graph_states
git checkout main
pipenv sync
pipenv install .
```
Then you can activate the virtual environment with `pipenv shell`.

## Documentation

The documentation can be built from source with Sphinx,
but it is also hosted at [https://noisy-graph-states.readthedocs.io](https://noisy-graph-states.readthedocs.io)

## Motivation

There are many protocols in quantum information science
that are based on graph states and transformations of
graph states. In any realistic scenario noise and
imperfections have to be taken into account in order
to analyse the performance of such protocols.

While there are existing tools for dealing with
stabilizer states and Clifford circuits,
it can be useful to stay within the graph state
interpretation for the whole protocol.
Furthermore, our approach allows us to explicitly
obtain the density matrix of the output state
without the need to sample from it.


### Working principle

Instead of updating the density matrix, instead track
how the noise on the state transforms along with the
graph state transformation.

For some cases of noise (such as local noise acting on
the initial state before operations are performed)
the Noisy Stabilizer Formalism allows to do this
very efficiently (updating O(n) noises instead of
exponentially many density matrix entries).

The main insight here is that the
noise channels can be tracked individually instead
of being combined to one global channel,
e.g. local depolarizing noise on every qubit is highly
structured, but nonetheless a full rank noise channel
viewed in a global picture.

However, note that this efficiency increase is not
guaranteed in general, as with the general correlated
noise, one inevitably needs to track exponentially many
entries again.

## Use of the code
The noisy graph state package was used for these publications:

> Imperfect quantum networks with tailored resource states <br>
> M. F. Mor-Ruiz, J. Wallnöfer, W. Dür <br>
> Preprint: [arXiv:2403.19778 [quant-ph]](https://doi.org/10.48550/arXiv.2403.19778)
