# Integration

In this example we show how to read data from disk and perform integration and averages in the Spectral Element mesh.

# Interpolation

Furthermore, we show how one would create an unstructured set of points and spectrally interpolate into them.

Note that for this particular operation, the unstructred mesh must be in rank 0. The probes object takes care of scattering the information to all ranks.

In the future we will support providing scattered data but this strategy follows that of the probes in Neko
