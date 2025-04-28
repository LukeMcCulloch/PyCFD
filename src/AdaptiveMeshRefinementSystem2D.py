#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:43:54 2025

@author: lukemcculloch


Integration of Unstructured AMR with System2D
"""
import numpy as np
import matplotlib.pyplot as plt
from unstructured_amr import UnstructuredMesh, UnstructuredAMR
from System2D import System2D
from Grid import Grid
from Solvers import EulerSolver

class AMRSystem2D(System2D):
    """
    Extended System2D class with AMR capabilities
    """
    def __init__(self, Nx=50, Ny=50, xmin=0., xmax=1., ymin=0., ymax=1., 
                 gamma=1.4, max_amr_level=3):
        # Initialize base System2D
        super().__init__(Nx, Ny, xmin, xmax, ymin, ymax, gamma)
        
        # Create AMR mesh
        self.amr_mesh = self._create_amr_mesh_from_system()
        
        # Initialize AMR controller
        self.amr_controller = UnstructuredAMR(self.amr_mesh, max_level=max_amr_level)
        self.amr_controller.set_refinement_threshold(0.05)
        self.amr_controller.set_coarsening_threshold(0.01)
        
        # Track if AMR mesh needs synchronization with System2D
        self.needs_sync = False
    
    def _create_amr_mesh_from_system(self):
        """Create AMR mesh from System2D grid"""
        mesh = UnstructuredMesh()
        
        # For each cell in the System2D grid
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                # Get cell vertices
                x_left = self.grid.x[i]
                x_right = self.grid.x[i+1]
                y_bottom = self.grid.y[j]
                y_top = self.grid.y[j+1]
                
                vertices = [
                    (x_left, y_bottom),
                    (x_right, y_bottom),
                    (x_right, y_top),
                    (x_left, y_top)
                ]
                
                # Add cell to AMR mesh
                cell_id = mesh.add_cell(vertices)
                
                # Store i,j indices for mapping
                mesh.cells[cell_id].system_i = i
                mesh.cells[cell_id].system_j = j
        
        # Set up neighbors
        self._setup_amr_mesh_neighbors(mesh)
        
        # Copy conserved variables
        self._copy_solution_to_amr_mesh(mesh)
        
        return mesh
    
    def _setup_amr_mesh_neighbors(self, mesh):
        """Set up neighbor relationships in AMR mesh"""
        for cell_id, cell in mesh.cells.items():
            if hasattr(cell, 'system_i') and hasattr(cell, 'system_j'):
                i, j = cell.system_i, cell.system_j
                
                # Check all 4 neighbors
                neighbor_indices = [
                    (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                ]
                
                for ni, nj in neighbor_indices:
                    if 0 <= ni < self.grid.nx and 0 <= nj < self.grid.ny:
                        # Find neighbor cell_id
                        for nid, ncell in mesh.cells.items():
                            if (hasattr(ncell, 'system_i') and hasattr(ncell, 'system_j') and 
                                ncell.system_i == ni and ncell.system_j == nj):
                                cell.neighbors.append(nid)
                                break
    
    def _copy_solution_to_amr_mesh(self, mesh):
        """Copy solution from System2D to AMR mesh"""
        for cell_id, cell in mesh.cells.items():
            if hasattr(cell, 'system_i') and hasattr(cell, 'system_j'):
                i, j = cell.system_i, cell.system_j
                
                # Get conserved variables from System2D
                cell.conserved_vars = np.array([
                    self.grid.U[0, i, j],  # Density
                    self.grid.U[1, i, j],  # X-Momentum
                    self.grid.U[2, i, j],  # Y-Momentum
                    self.grid.U[3, i, j]   # Energy
                ])
    
    def _copy_solution_from_amr_mesh(self):
        """Copy solution from AMR mesh to System2D"""
        # For base level cells (direct mapping)
        for cell_id, cell in self.amr_mesh.cells.items():
            if not cell.children and hasattr(cell, 'system_i') and hasattr(cell, 'system_j'):
                i, j = cell.system_i, cell.system_j
                
                # Only if this is a base-level cell
                if cell.level == 0:
                    # Copy conserved variables to System2D
                    self.grid.U[0, i, j] = cell.conserved_vars[0]  # Density
                    self.grid.U[1, i, j] = cell.conserved_vars[1]  # X-Momentum
                    self.grid.U[2, i, j] = cell.conserved_vars[2]  # Y-Momentum
                    self.grid.U[3, i, j] = cell.conserved_vars[3]  # Energy
        
        # For refined cells (need to average to get values for System2D)
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                # Find all AMR cells that overlap with this System2D cell
                overlapping_cells = self._find_overlapping_cells(i, j)
                
                if overlapping_cells and any(self.amr_mesh.cells[cid].level > 0 for cid in overlapping_cells):
                    # Calculate weighted average of conserved variables
                    total_area = 0
                    weighted_U = np.zeros(4)
                    
                    for cell_id in overlapping_cells:
                        cell = self.amr_mesh.cells[cell_id]
                        if not cell.children:  # Only consider leaf cells
                            area = cell.area
                            total_area += area
                            weighted_U += area * cell.conserved_vars
                    
                    if total_area > 0:
                        # Update System2D with weighted average
                        self.grid.U[:, i, j] = weighted_U / total_area
    
    def _find_overlapping_cells(self, i, j):
        """Find AMR cells that overlap with System2D cell (i,j)"""
        # Get System2D cell boundaries
        x_min = self.grid.x[i]
        x_max = self.grid.x[i+1]
        y_min = self.grid.y[j]
        y_max = self.grid.y[j+1]
        
        overlapping_cells = []
        
        # Check all leaf cells in AMR mesh
        for cell_id in self.amr_mesh.get_leaf_cells():
            cell = self.amr_mesh.cells[cell_id]
            
            # Check if cell overlaps with System2D cell
            cell_x_min = min(v[0] for v in cell.vertices)
            cell_x_max = max(v[0] for v in cell.vertices)
            cell_y_min = min(v[1] for v in cell.vertices)
            cell_y_max = max(v[1] for v in cell.vertices)
            
            # Check for overlap
            if (cell_x_min < x_max and cell_x_max > x_min and
                cell_y_min < y_max and cell_y_max > y_min):
                overlapping_cells.append(cell_id)
        
        return overlapping_cells
    
    def adapt_mesh(self):
        """Adapt the mesh based on solution features"""
        self.amr_controller.adapt_mesh()
        self.needs_sync = True
    
    def solve(self, tmax, dt=None, save_interval=None, adapt_interval=5):
        """
        Solve the system with AMR up to time tmax
        """
        t = 0.0
        step = 0
        next_save = 0.0
        
        # Calculate initial dt if not provided
        if dt is None:
            dt = self.grid.calculate_dt(self.gamma)
        
        # Initialize plots
        if save_interval is not None:
            self.initialize_plots()
            self.save_solution(t)
            next_save = save_interval
        
        # Initial adaptation
        self.adapt_mesh()
        
        # Main time loop
        while t < tmax:
            # Adjust dt for the last step
            if t + dt > tmax:
                dt = tmax - t
            
            # Advance solution using AMR
            t_new, substeps = self.amr_controller.run_simulation(self.solver_adapter(), dt, adapt_interval=adapt_interval)
            t += dt
            step += 1
            
            # Synchronize AMR mesh with System2D if needed
            if self.needs_sync:
                self._copy_solution_from_amr_mesh()
                self.needs_sync = False
            
            # Save solution if needed
            if save_interval is not None and t >= next_save:
                self.save_solution(t)
                next_save += save_interval
            
            # Print progress
            print(f"Step: {step}, Time: {t:.6f}, Cells: {len(self.amr_mesh.get_leaf_cells())}")
        
        # Final save
        if save_interval is not None:
            self.save_solution(t)
        
        return t, step
    
    def solver_adapter(self):
        """Create an adapter for the AMR solver that uses your existing EulerSolver"""
        # Create a wrapper class that adapts EulerSolver for use with AMR
        class SolverAdapter:
            def __init__(self, system):
                self.system = system
                self.gamma = system.gamma
            
            def update_cell(self, U, position, neighbor_states, neighbor_positions, dt):
                # This would need to be implemented to use your existing solver logic
                # For now, we use a simple approach
                
                # Calculate fluxes with each neighbor
                dU = np.zeros_like(U)
                
                for neighbor_U, neighbor_pos in zip(neighbor_states, neighbor_positions):
                    # Calculate direction and distance
                    dx = neighbor_pos[0] - position[0]
                    dy = neighbor_pos[1] - position[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < 1e-10:
                        continue
                    
                    # Unit normal vector
                    nx = dx / distance
                    ny = dy / distance
                    
                    # Calculate flux (simplified)
                    flux = self._calculate_flux(U, neighbor_U, nx, ny)
                    
                    # Add contribution
                    dU -= flux * dt / distance
                
                return U + dU
            
            def _calculate_flux(self, UL, UR, nx, ny):
                # Simplified Riemann solver (Rusanov/Local Lax-Friedrichs)
                # Extract primitive variables
                rhoL, uL, vL, pL = self._conserved_to_primitive(UL)
                rhoR, uR, vR, pR = self._conserved_to_primitive(UR)
                
                # Normal velocities
                vnL = uL * nx + vL * ny
                vnR = uR * nx + vR * ny
                
                # Sound speeds
                cL = np.sqrt(self.gamma * pL / rhoL) if rhoL > 1e-10 else 0
                cR = np.sqrt(self.gamma * pR / rhoR) if rhoR > 1e-10 else 0
                
                # Maximum wave speed
                smax = max(abs(vnL) + cL, abs(vnR) + cR)
                
                # Compute fluxes in normal direction
                FL = self._compute_normal_flux(UL, nx, ny)
                FR = self._compute_normal_flux(UR, nx, ny)
                
                # Local Lax-Friedrichs flux
                F = 0.5 * (FL + FR - smax * (UR - UL))
                
                return F
            
            def _compute_normal_flux(self, U, nx, ny):
                rho, u, v, p = self._conserved_to_primitive(U)
                
                # Normal velocity
                vn = u * nx + v * ny
                
                # Flux components
                mass_flux = rho * vn
                momentum_x_flux = rho * u * vn + p * nx
                momentum_y_flux = rho * v * vn + p * ny
                energy_flux = (U[3] + p) * vn
                
                return np.array([mass_flux, momentum_x_flux, momentum_y_flux, energy_flux])
            
            def _conserved_to_primitive(self, U):
                rho = U[0]
                u = U[1] / rho if rho > 1e-10 else 0
                v = U[2] / rho if rho > 1e-10 else 0
                E = U[3]
                
                # Internal energy
                e_int = E / rho - 0.5 * (u*u + v*v) if rho > 1e-10 else 0
                
                # Pressure
                p = (self.gamma - 1.0) * rho * e_int
                
                return rho, u, v, p
            
            def estimate_wave_speed(self, U):
                rho, u, v, p = self._conserved_to_primitive(U)
                
                # Sound speed
                c = np.sqrt(self.gamma * p / rho) if rho > 1e-10 else 0
                
                # Maximum wave speed
                return np.sqrt(u*u + v*v) + c
        
        return SolverAdapter(self)
    
    def plot_amr_mesh(self, colorby='level', cmap='viridis', show_cell_ids=False):
        """Plot AMR mesh colored by level or solution value"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if colorby == 'level':
            # Color by refinement level
            self.amr_mesh.plot_mesh(ax=ax, color_by_level=True, show_cell_ids=show_cell_ids)
        else:
            # Color by solution value
            var_idx = {'density': 0, 'momentum_x': 1, 'momentum_y': 2, 'energy': 3}.get(colorby, 0)
            self.amr_mesh.plot_solution(variable_index=var_idx, ax=ax, cmap=cmap)
        
        plt.tight_layout()
        return fig, ax
    
    def save_vtk(self, filename, iteration=0):
        """Export mesh and solution to VTK format for visualization with ParaView"""
        try:
            import pyevtk
            from pyevtk.hl import unstructuredGridToVTK
            from pyevtk.vtk import VtkTriangle, VtkQuad
        except ImportError:
            print("pyevtk package not found. Install it with: pip install pyevtk")
            return
        
        # Get leaf cells
        leaf_cells = self.amr_mesh.get_leaf_cells()
        num_cells = len(leaf_cells)
        
        # Prepare points and connectivity
        all_points = []
        point_map = {}  # Maps vertex coordinates to point index
        
        # Cell types and connectivity
        cell_types = []
        cell_conn = []
        
        # Process each leaf cell
        for cell_idx, cell_id in enumerate(leaf_cells):
            cell = self.amr_mesh.cells[cell_id]
            vertices = cell.vertices
            
            # Add vertices to point list if not already present
            cell_conn_indices = []
            for v in vertices:
                v_tuple = (v[0], v[1], 0.0)  # Convert to 3D point with z=0
                if v_tuple not in point_map:
                    point_map[v_tuple] = len(all_points)
                    all_points.append(v_tuple)
                
                cell_conn_indices.append(point_map[v_tuple])
            
            # Add cell connectivity
            cell_conn.extend(cell_conn_indices)
            
            # Determine cell type
            if len(vertices) == 3:
                cell_types.append(VtkTriangle.tid)
            elif len(vertices) == 4:
                cell_types.append(VtkQuad.tid)
            else:
                # For other polygon types, you'd need to convert to triangles or quads
                print(f"Warning: Cell with {len(vertices)} vertices not supported, skipping")
                continue
        
        # Extract point coordinates
        x = [p[0] for p in all_points]
        y = [p[1] for p in all_points]
        z = [p[2] for p in all_points]
        
        # Prepare cell data
        density = np.zeros(num_cells)
        pressure = np.zeros(num_cells)
        velocity_x = np.zeros(num_cells)
        velocity_y = np.zeros(num_cells)
        level = np.zeros(num_cells)
        
        # Get solution data for cells
        for i, cell_id in enumerate(leaf_cells):
            cell = self.amr_mesh.cells[cell_id]
            
            # Get conserved variables
            U = cell.conserved_vars
            
            # Convert to primitive variables
            rho = U[0]
            u = U[1] / rho if rho > 1e-10 else 0
            v = U[2] / rho if rho > 1e-10 else 0
            E = U[3]
            
            # Internal energy
            e_int = E / rho - 0.5 * (u*u + v*v) if rho > 1e-10 else 0
            
            # Pressure
            p = (self.gamma - 1.0) * rho * e_int
            
            # Store values
            density[i] = rho
            pressure[i] = p
            velocity_x[i] = u
            velocity_y[i] = v
            level[i] = cell.level
        
        # Create offset array
        offsets = np.zeros(num_cells, dtype=np.int32)
        vertex_counts = []
        offset = 0
        
        for cell_id in leaf_cells:
            num_vertices = len(self.amr_mesh.cells[cell_id].vertices)
            vertex_counts.append(num_vertices)
            offset += num_vertices
            offsets[cell_idx] = offset
        
        # Write VTK file
        cell_data = {
            "density": density,
            "pressure": pressure,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "level": level
        }
        
        # Write VTK file
        unstructuredGridToVTK(
            filename,
            np.array(x),
            np.array(y),
            np.array(z),
            connectivity=np.array(cell_conn),
            offsets=offsets,
            cell_types=np.array(cell_types),
            cellData=cell_data
        )
        
        print(f"Saved VTK file: {filename}.vtu")


def test_amr_system():
    """Test function for AMRSystem2D"""
    # Create system
    system = AMRSystem2D(Nx=50, Ny=50, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, 
                         gamma=1.4, max_amr_level=3)
    
    # Initialize with a blast wave problem
    def blast_wave_init(x, y):
        """Initialize a 2D blast wave problem"""
        # Center of the domain
        x_c, y_c = 0.5, 0.5
        r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        
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
        gamma = 1.4
        E = p / (gamma - 1) + 0.5 * rho * (u*u + v*v)
        
        return np.array([rho, rho*u, rho*v, E])
    
    # Apply initial condition
    for i in range(system.grid.nx):
        for j in range(system.grid.ny):
            x = system.grid.x[i] + 0.5 * system.grid.dx
            y = system.grid.y[j] + 0.5 * system.grid.dy
            
            U = blast_wave_init(x, y)
            
            system.grid.U[0, i, j] = U[0]  # Density
            system.grid.U[1, i, j] = U[1]  # X-Momentum
            system.grid.U[2, i, j] = U[2]  # Y-Momentum
            system.grid.U[3, i, j] = U[3]  # Energy
    
    # Synchronize with AMR mesh
    system._copy_solution_to_amr_mesh(system.amr_mesh)
    
    # Plot initial condition
    fig, ax = system.plot_amr_mesh(colorby='density')
    plt.savefig('initial_condition.png')
    plt.close()
    
    # Initial adaptation
    system.adapt_mesh()
    
    # Plot adapted mesh
    fig, ax = system.plot_amr_mesh(colorby='level')
    plt.savefig('initial_adaptation.png')
    plt.close()
    
    # Solve to final time
    t_final = 0.2
    t, steps = system.solve(t_final, adapt_interval=5, save_interval=0.05)
    
    # Plot final solution
    fig, ax = system.plot_amr_mesh(colorby='density')
    plt.savefig('final_solution.png')
    plt.close()
    
    # Export VTK for ParaView visualization
    system.save_vtk('blast_wave_amr', iteration=steps)
    
    print(f"Simulation completed: time = {t:.6f}, steps = {steps}")
    print(f"Total cells: {len(system.amr_mesh.cells)}")
    print(f"Leaf cells: {len(system.amr_mesh.get_leaf_cells())}")
    print(f"Maximum refinement level: {system.amr_mesh.max_level}")


if __name__ == "__main__":
    test_amr_system()