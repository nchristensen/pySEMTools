# POD

In this example we show how to perform proper orthogonal decomposition (POD) based on a set of snapshots stored in disk.

Note that the algorithm works in parallel.

We perform an streaming SVD strategy. If the user wish to perform the typical POD, all that needs to be done is to set the batch size to be equal to the number of snapshots.
