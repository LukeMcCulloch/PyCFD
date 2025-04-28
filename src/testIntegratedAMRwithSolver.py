#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:49:46 2025

@author: lukemcculloch


Integration module to connect cell-based AMR with existing System2D classes
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from copy import deepcopy

# Import PyCFD modules
from System2D import System2D
from Grid import Grid
from Solvers import EulerSolver

class AMRIntegrator:
    """
    Class to integrate cell-based AMR with System2D
    """
    def __init__(self, system2d, max_amr_level=3):
        """
        Initialize AMR integrator with an existing System2D instance
        
        Args:
            system2d: Existing System2D instance
            max_amr_level: Maximum AMR refinement level
        """
        self.system2d = system2d
        self.max_amr_level = max_amr_level
        
        # Import AMR modules
        from unstructured_amr import UnstructuredMesh, UnstructuredCell
        from full_amr_example import UnstructuredAMRSystem, AMRGradientTracker
        
        # Create AMR system that will work alongside System2D
        self.amr_system = UnstructuredAMRSystem(
            nx=system2d.grid.nx,
            ny=system2d.grid.ny,
            xmin=system2d.grid.x[0],
            xmax=system2d.grid.x[-1],
            ymin=system2d.grid.y[0],
            ymax=system2d.grid.y[-1],
            gamma=system2d.gamma
        )
        
        # Initialize AMR with current System2D solution
        self._sync_system2d_to_amr()
    
    def _sync_system2d_to_amr(self):
        """Synchronize solution from System2D to AMR system"""
        # Create initial condition function that pulls data from System2D
        def system2d_data(x, y):
            # Find closest cell in System2D
            i = max(0, min(self.system2d.grid.nx-1, int((x - self.system2d.grid.x[0]) / self.system2d.grid.dx)))
            j = max(0, min(self.system2d.grid.ny-1, int((y - self.system2d.grid.y[0]) / self.system2d.grid.dy)))
            
            # Get conserved variables
            return np.array([
                self.system2d.grid.U[0, i, j],  # Density
                self.system2d.grid.U[1, i, j],  # X-Momentum
                self.system2d.grid.U[2, i, j],  # Y-Momentum
                self.system2d.grid.U[3, i, j]   # Energy
            ])
        
        # Initialize AMR system with this data
        self.amr_system.initialize_solution(system2d_data)
    
    def _sync_amr_to_system2d(self):
        """Synchronize solution from AMR system to System2D"""
        # For each cell in System2D
        for i in range(self.system2d.grid.nx):
            for j in range(self.system2d.grid.ny):
                # Get cell center coordinates
                x = self.system2d.grid.x[i] + 0.5 * self.system2d.grid.dx
                y = self.system2d.grid.y[j] + 0.5 * self.system2d.grid.dy
                
                # Find overlapping cells in AMR
                overlapping_cells = self._find_amr_cells_at_point(x, y)
                
                if overlapping_cells:
                    # Calculate weighted average of conserved variables
                    total_area = 0
                    weighted_U = np.zeros(4)
                    
                    for cell_id in overlapping_cells:
                        cell = self.amr_system.mesh.cells[cell_id]
                        if not cell.children and cell.conserved_vars is not None:
                            area = cell.area
                            total_area += area
                            weighted_U += area * cell.conserved_vars
                    
                    if total_area > 0:
                        # Update System2D with weighted average
                        self.system2d.grid.U[:, i, j] = weighted_U / total_area
    
    def _find_amr_cells_at_point(self, x, y):
        """Find AMR cells that contain the point (x, y)"""
        matching_cells = []
        
        # Check all leaf cells
        for cell_id, cell in self.amr_system.mesh.cells.items():
            if cell.children:
                continue  # Skip non-leaf cells
            
            # Check if point is inside cell
            if self._point_in_polygon(x, y, cell.vertices):
                matching_cells.append(cell_id)
        
        return matching_cells
    
    def _point_in_polygon(self, x, y, vertices):
        """Check if point (x, y) is inside a polygon defined by vertices"""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                
                if p1x == p2x or x <= xinters:
                    inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def solve_with_amr(self, tmax, dt=None, save_interval=None, adapt_interval=5):
        """
        Solve the system using AMR up to time tmax
        
        Args:
            tmax: Final time
            dt: Time step (if None, calculated based on CFL)
            save_interval: Interval at which to save solution
            adapt_interval: Interval at which to adapt mesh
            
        Returns:
            t, steps: Final time and number of steps
        """
        # Initial time and step
        t = 0.0
        step = 0
        next_save = 0.0
        
        # Calculate initial dt if not provided
        if dt is None:
            dt = self.system2d.grid.calculate_dt(self.system2d.gamma)
        
        # Initialize plots if save_interval is specified
        if save_interval is not None:
            if not os.path.exists('amr_output'):
                os.makedirs('amr_output')
            
            # Save initial condition
            self.save_solution(0, 0.0)
            next_save = save_interval
        
        # Initial adaptation
        self.amr_system.adapt_mesh(self.max_amr_level)
        
        # Main time loop
        while t < tmax:
            # Adjust dt for the last step
            if t + dt > tmax:
                dt = tmax - t
            
            # Advance AMR solution
            self.amr_system.advance_solution(dt)
            
            # Update time
            t += dt
            step += 1
            
            # Adapt mesh periodically
            if step % adapt_interval == 0:
                self.amr_system.adapt_mesh(self.max_amr_level)
            
            # Synchronize back to System2D periodically (for visualization)
            if save_interval is not None and t >= next_save:
                self._sync_amr_to_system2d()
                self.save_solution(step, t)
                next_save += save_interval
            
            # Print progress
            print(f"Step: {step}, Time: {t:.6f}, Cells: {len([c for c in self.amr_system.mesh.cells.values() if not c.children])}")
        
        # Final synchronization and save
        self._sync_amr_to_system2d()
        
        if save_interval is not None:
            self.save_solution(step, t)
        
        return t, step
    
    def save_solution(self, step, time):
        """Save current solution for visualization"""
        # Create directory if it doesn't exist
        if not os.path.exists('amr_output'):
            os.makedirs('amr_output')
        
        # Save visualization of AMR mesh and solution
        fig = plt.figure(figsize=(15, 10))
        
        # Plot System2D solution
        ax1 = plt.subplot(2, 2, 1)
        self.system2d.plot(ax1)
        ax1.set_title(f'System2D Solution (t = {time:.3f})')
        
        # Plot AMR mesh colored by refinement level
        ax2 = plt.subplot(2, 2, 2)
        self.amr_system._plot_mesh(ax2, color_by_level=True)
        ax2.set_title(f'AMR Mesh ({len([c for c in self.amr_system.mesh.cells.values() if not c.children])} cells)')
        
        # Plot AMR solution - density
        ax3 = plt.subplot(2, 2, 3)
        self.amr_system._plot_solution(ax3, var_idx=0)
        ax3.set_title('AMR Density')
        
        # Plot AMR solution - pressure
        ax4 = plt.subplot(2, 2, 4)
        self.amr_system._plot_solution(ax4, var_idx='pressure')
        ax4.set_title('AMR Pressure')
        
        plt.tight_layout()
        plt.savefig(f'amr_output/solution_{step:06d}.png')
        plt.close()
        
        # Save VTK for ParaView
        self.amr_system._save_vtk(f'amr_output/solution_{step:06d}')
    
    def export_vtk_sequence(self):
        """
        Export the entire solution sequence to VTK format for ParaView
        This creates a .pvd file that can be opened in ParaView to visualize
        the time-dependent solution
        """
        try:
            import pyevtk
        except ImportError:
            print("pyevtk package not found. Cannot create PVD file.")
            return
        
        # Get list of VTK files
        vtk_files = [f for f in os.listdir('amr_output') if f.endswith('.vtu')]
        
        # Sort by step number
        vtk_files.sort()
        
        # Extract time from filename (assuming format solution_NNNNNN.vtu)
        steps = [int(f.split('_')[1].split('.')[0]) for f in vtk_files]
        
        # Create time values (linear spacing is fine for visualization)
        times = np.linspace(0, 1, len(vtk_files))
        
        # Create PVD file
        with open('amr_output/solution.pvd', 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            
            for i, (step, time, vtk_file) in enumerate(zip(steps, times, vtk_files)):
                f.write(f'    <DataSet timestep="{time}" group="" part="0" file="{vtk_file}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        print("Created PVD file for ParaView time-series visualization")


def test_integration():
    """Test function for AMR integration with System2D"""
    # Create System2D instance
    system = System2D(nx=50, ny=50, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, gamma=1.4)
    
    # Initialize with a blast wave
    for i in range(system.grid.nx):
        for j in range(system.grid.ny):
            # Cell center
            x = system.grid.x[i] + 0.5 * system.grid.dx
            y = system.grid.y[j] + 0.5 * system.grid.dy
            
            # Distance from center
            x_c, y_c = 0.5, 0.5
            r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
            
            # Set initial condition
            if r < 0.1:
                # High pressure region
                rho = 1.0
                u = 0.0
                v = 0.0
                p = 1.0
            else:
                # Low pressure region
                rho = 0.125
                u = 0.0
                v = 0.0
                p = 0.1
            
            # Convert to conserved variables
            E = p / (system.gamma - 1) + 0.5 * rho * (u*u + v*v)
            
            # Set in grid
            system.grid.U[0, i, j] = rho
            system.grid.U[1, i, j] = rho * u
            system.grid.U[2, i, j] = rho * v
            system.grid.U[3, i, j] = E
    
    # Create AMR integrator
    integrator = AMRIntegrator(system, max_amr_level=3)
    
    # Run simulation
    t_final = 0.2
    t, steps = integrator.solve_with_amr(
        t_final,
        save_interval=0.04,
        adapt_interval=5
    )
    
    # Export VTK sequence for ParaView
    integrator.export_vtk_sequence()
    
    print(f"Simulation completed: time = {t:.6f}, steps = {steps}")


if __name__ == "__main__":
    test_integration()

