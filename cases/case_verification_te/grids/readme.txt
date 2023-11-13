#
# Compile and run the program to generate a series of grids.
#

gfortran hw1_generate_grids.f90

./a.out

# or

./a.exe

# and you'll have lots of .grid files.
# Copy the one you wish to use to the above directory
# as, e.g., "te_test.grid",

cp grid_129x129_tria.grid ../te_test.grid

# You will put -> project_name = "te_test"
# in the input.nml file and the CFD code will
# read te_test.grid.

