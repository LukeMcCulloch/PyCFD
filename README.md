# PyCFD

Some aspects of computational fluid dynamics, prototyped in Python



# Requirements

1. NumPy
2. matplotlib
3. weakref


# Optional
1. memory_profiler (optional)

# Some implementation details pictured

![LSQ gradient stencil at the cell colored in green](pics/stencil_57.png)


# Testing

The implemenatation is not yet finished.  Right now, I am working through the explicit Euler solver (Runge Kutta time stepping at the moment)  You can run the "solver.py" file directly and make pictures of LSQ stencils around any cell.  You can also run the "System2D.py" file (sorry for stupid naming) and get plots showing small tri and quad grids, generated in a structured fashion, but used as if they were fully unstructured.  This is by design.  The point of this exercise is to go through the motions of implementing an unstructrued Euler solver before "doing the real thing" in C++.  At any rate, this code should have utility in showing how things fit together.  

Cheers,

Luke
