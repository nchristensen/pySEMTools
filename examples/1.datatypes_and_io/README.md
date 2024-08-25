
# Data types and IO

In these examples we will show how to interact with the classes that have been designed to hold SEM data.

This means that we will show how to declare them and to populate them with data, as well as some of the options available.

Note that all the codes run in parallel with MPI. Notebooks are not so good at that (although it is possible to do), so to execute in parallel, copy the parts you need to a python script and run from there.

Notebook index:

1. Shows typical ways to populate objects and read/output files
2. Shows how to work on reduced partitions of the SEM domain
3. Shows how to use adios2 to perform data compression