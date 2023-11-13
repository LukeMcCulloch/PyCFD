# PyCFD

An unstructured solver for the Euler equations on a 2D grid, prototyped in Python.

1. LSQ gradient reconstruction
2. Roe approximate Riemmann solver for the flux 
3. Cell centered.
4. Tri and quad meshes shown



# Requirements

1. NumPy
2. matplotlib
3. weakref


# Optional
1. memory_profile
2. paraview for viewing vtk output files

# Unstructured mesh physics reconstruction

![LSQ gradient stencil at the cell colored in green](pics/stencil_57.png)



# Testing

The implemenatation is pretty well finished, but I am expanding it.  Right now, the explicity steady solver is known to work well for the cylinder and airfoil test cases.  You can also run the "System2D.py" file and get plots showing small tri and quad grids.  The solver treats all meshes as unstructured by design.  The point of this exercise is to go through the motions of implementing an unstructrued Euler solver before "doing the real thing" in C++.  

Here are some shock diffraction (supersonic flow over a backward facing step) plots made with the solver:

![AirfoilDensity](pics/test_cases/shock_diffraction/density.png)
![AirfoilDensity](pics/test_cases/shock_diffraction/mach.png)
![AirfoilDensity](pics/test_cases/shock_diffraction/mach_reflection_gridded.png)
![AirfoilDensity](pics/test_cases/shock_diffraction/pressure.png)


And here's a sample result of flow over an airfoil:

![AirfoilDensity](pics/test_cases/steady_airfoil/density.png)
![AirfoilDensity](pics/test_cases/steady_airfoil/x-velocity.png)
![AirfoilDensity](pics/test_cases/steady_airfoil/y-velocity.png)
![AirfoilDensity](pics/test_cases/steady_airfoil/pressure.png)


Update:  need to update the initial vortex:

![StrongVortex](pics/solution/AlmostVortex.png)



Here are plots of density and pressure, done with two different methods of interpolating unstructured data to a cartesian grid for plotting.  Not sure which is better just yet.

![DensityPress](pics/solution/DensityAndPressure.png)


Hmm... That was for a tri-mesh.  Here is the same on a quad mesh:


![StrongVortex](pics/solution/AlmostVortexQuad.png)


![DensityPress](pics/solution/DensityAndPressureQuad.png)


# Basic Mesh Plots in Python

# # Quad Mesh

![LSQ gradient stencil at the cell colored in green](pics/CarteasianCheckMesh.png)

Normals, centroids, face centers, edges, and vertices shown.

# # Tri mesh

![LSQ gradient stencil at the cell colored in green](pics/TriangularCheckMesh.png)

Normals, centroids, face centers, edges, and vertices shown.




Cheers,

Luke
