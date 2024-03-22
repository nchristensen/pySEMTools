# Derivation

In this example we show how to read data from disk and obtain derivatives in the Spectral Element mesh.

Furthermore, we show how to reduce the discontinouties among elements. Note that this operation will work on elements that have neighbours in the same MPI rank. Searching for neighbours in other ranks is not implemented.
