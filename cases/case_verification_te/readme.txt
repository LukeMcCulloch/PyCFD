#########################################################
# To perform a TE test.  ... sorry this is the c++ readme?  
# it's been a while.'
#########################################################

# Go into the directory 'grids'.

cd grids

# Then, generate grids by compiling and running
# the grid generation code.
#
# Note:
#  The option '-o xxx' generates an executable named xxx.
#  Without this option, the executable will be a.out
#  or a.exe.

gfortran -o hw1 hw1_generate_grids.f90

./hw1

# Now you see lots of grids generated.
# TE should be O(h) for a first-order residual
# on a regular quadrilateral grid.
#
# Compute TE on grid_009x009_quad.grid by copying
# the grid file to the directory above with the
# name "te_test.grid", which will be read by the solver.

cp grid_009x009_quad.grid ../te_test.grid
cd ..

# and run edu2d

./edu2d

# You'll see TE's computed and printed on screen.

# Note: TE is computed for all 4 equations for
#       the Euler equations.

# Next, go back to the grid directory, and copy
# the next level grid, grid_017x017_quad.grid

cp grid_017x017_quad.grid ../te_test.grid
cd ..

# and run edu2d

./edu2d

# You'll see TE comptued and printed on screen.
# Compare these values from the previous ones.
# If the residual is first-order, you should see
# the TE is reduced by half since the mesh spacing
# is reduced by half:
# (9x9 -> 17x17 in nodes: 8x8 -> 16x16 in cells).

# If TE does not go down as expected for any equation,
# you must have a bug. Go back to the residual subroutine,
# and find and fix bugs.

#
# You can continue to finer grids if you want to make sure
# you see O(h) for finer grids. I would check for all grids.


