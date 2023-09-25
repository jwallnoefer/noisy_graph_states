# Noisy graph states

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

TODO: to be added once published

If you encounter any problems, you can try installing the
exact versions of the dependencies of this package, which
were used to develop it (specified in `Pipfile.lock`).
This assume Python 3.9 and `pipenv` are available on your system.


## Documentation

The documentation can be built from source with sphinx,
but it is also hosted at [insert link]

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
