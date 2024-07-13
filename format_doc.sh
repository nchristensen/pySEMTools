#!/bin/bash

# This script is written so you do not forget how to format the documentation.
pyment -o numpydoc $@

# To apply the path, run the following command.
#patch -p1 < <file_name>.py.patch