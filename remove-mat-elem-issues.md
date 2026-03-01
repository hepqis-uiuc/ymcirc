# Issues with implementation of remove-mat-elems-plan.md

## Context
The refactor described in remove-mat-elems-plan.md has been implemented. (For more context, see the file remove-matrix-element-merging.md in the recent commit with short hash c5387a6 which served as the initial prompt leading to the creation of the refactor plan.)

There are two outstanding issues from this refactor. One is a MAJOR issue. The other is minor.

## The major issue
The `Plaquette.compute_signature` method in `lattice_data.py` is a static method! This means it has no idea what the actual, extant half-link directions for the current plaquette are at each of its vertices. While this is a non-issue for periodic lattices, this will cause problems in the future once logic for working with non-periodic lattices has been implemented.

The following example illustrates the problem. In d=3/2 with nonperiodic boundary conditions and standard F-order, the signature of a plaquette whose bottom-left vertex is (0,0) will be ((1,), (1,-1), (1,-1), (1,)) since this plaquette is at the "left end" of the lattice. If the lattice has more than 3 plaquettes in it, the signature of the plaquette with bottom-left vertex (1,0) will be ((1,-1), (1,-1), (1,-1), (1,-1)). If the lattice has exactly 3 plaquettes, the signature of the plaquette with bottom-left vertex (2,0) will be ((1,-1), (-1,), (-1,), (1,-1)) since it is on the "right end". So for a d=3/2 lattice with nonperiodic boundary conditions with length greater than or equal to 3, there are three different kinds of plaquette signatures which can crop up. For d=2 (and eventually, d=3, though d=3 is currently out of scope), there are even more kinds of unique signatures which can occur.

To solve this problem, we need to compute the signature using the lattice geometry (shape, dimension, boundary conditions), the plane that the current plaquette is in, and the bottom-left vertex of the current plaquette. Given this information, it should be possible to compute what the half-links actually connected to each vertex are, and then to sort the half-links by F-order. I believe that taking a `LaticeDef` argument, and changing the static method to be a property should be sufficient for providing all information needed to perform this calculation.

## The minor issue
I noticed that in `test_circuit.py`, `test_compute_signature` only check behavior for standard choices of F-order. It would be nice to check the behavior for a nonstandard choice of F-order too.
