#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:46:42 2025

@author: lukemcculloch


Full example demonstrating cell-based AMR for unstructured meshes
integrated with the existing PyCFD codebase
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import existing PyCFD modules
#from System2D import System2D
import System2D
from Grid import Grid
from Solvers import EulerSolver

# Import AMR modules
from unstructured_amr import UnstructuredMesh, UnstructuredAMR, UnstructuredCell

class AMRGradientTracker:
    """
    Class for tracking solution gradients to determine where to refine/coarsen
    """
    def __init__(self, threshold_refine=0.1, threshold_coarsen=0.01, variable_idx=0):
        self.threshold_refine = threshold_refine
        self.threshold_coarsen = threshold_coarsen
        self.variable_idx = variable_idx  # Which variable to track (0=density)
    
    def mark_cells_for_refinement(self, mesh, max_level=3):
        """Mark cells that need refinement based on gradient"""
        # Reset all flags
        for cell in mesh.cells.values():
            cell.needs_refinement = False
            cell.can_coarsen = True
        
        # Mark cells for refinement
        self._mark_by_gradient(mesh, max_level)
        
        # Ensure proper nesting (no level jumps > 1)
        self._ensure_proper_nesting(mesh)
        
        # Return lists of cells to refine/coarsen
        refine_list = [cid for cid, cell in mesh.cells.items() 
                      if cell.needs_refinement and not cell.children]
        coarsen_list = [cid for cid, cell in mesh.cells.items() 
                       if cell.can_coarsen and cell.level > 0 and not cell.children]
        
        return refine_list, coarsen_list
    
    def _mark_by_gradient(self, mesh, max_level):
        """Mark cells based on solution gradient"""
        # For each leaf cell
        for cell_id, cell in mesh.cells.items():
            if cell.children:
                continue  # Skip non-leaf cells
            
            # Check refinement criteria
            if cell.level < max_level:
                gradient = self._calculate_gradient(mesh, cell)
                if gradient > self.threshold_refine:
                    cell.needs_refinement = True
            
            # Check coarsening criteria
            if cell.level > 0:
                gradient = self._calculate_gradient(mesh, cell)
                if gradient <= self.threshold_coarsen:
                    cell.can_coarsen = True
                else:
                    cell.can_coarsen = False
    
    def _calculate_gradient(self, mesh, cell):
        """Calculate approximate gradient magnitude at a cell"""
        if cell.conserved_vars is None or len(cell.neighbors) == 0:
            return 0.0
        
        # Get variable value for this cell
        value = cell.conserved_vars[self.variable_idx]
        
        # Calculate gradient with neighbors
        total_gradient = 0.0
        count = 0
        
        for neighbor_id in cell.neighbors:
            if neighbor_id in mesh.cells:
                neighbor = mesh.cells[neighbor_id]
                
                # Skip neighbors with children
                if neighbor.children:
                    continue
                
                # Skip neighbors without conserved vars
                if neighbor.conserved_vars is None:
                    continue
                
                # Get neighbor value
                neighbor_value = neighbor.conserved_vars[self.variable_idx]
                
                # Calculate distance between centroids
                dx = neighbor.centroid[0] - cell.centroid[0]
                dy = neighbor.centroid[1] - cell.centroid[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 1e-10:
                    # Calculate approximate gradient
                    gradient = abs(neighbor_value - value) / distance
                    total_gradient += gradient
                    count += 1
        
        # Return average gradient
        if count > 0:
            return total_gradient / count
        else:
            return 0.0
    
    def _ensure_proper_nesting(self, mesh):
        """Ensure proper nesting of refinement levels"""
        # Flag cells for refinement if they are adjacent to cells that are
        # more than one level higher
        
        changes = True
        while changes:
            changes = False
            
            # For each leaf cell
            for cell_id, cell in mesh.cells.items():
                if cell.children:
                    continue  # Skip non-leaf cells
                
                # Check neighbors
                for neighbor_id in cell.neighbors:
                    if neighbor_id in mesh.cells:
                        neighbor = mesh.cells[neighbor_id]
                        
                        # If neighbor has children
                        if neighbor.children:
                            # Find the child cells
                            for child_id in neighbor.children:
                                child = mesh.cells[child_id]
                                
                                # If child is more than one level higher
                                if child.level > cell.level + 1:
                                    # Mark this cell for refinement
                                    if not cell.needs_refinement:
                                        cell.needs_refinement = True
                                        changes = True
                        
                        # If neighbor itself is more than one level higher
                        elif neighbor.level > cell.level + 1:
                            if not cell.needs_refinement:
                                cell.needs_refinement = True
                                changes = True
        
        # Also ensure that cells can only be coarsened if their neighbors won't 
        # create level jumps > 1
        for cell_id, cell in mesh.cells.items():
            if not cell.can_coarsen or cell.children:
                continue  # Skip cells that can't be coarsened or non-leaf cells
            
            # Check if coarsening would create improper nesting
            for neighbor_id in cell.neighbors:
                if neighbor_id in mesh.cells:
                    neighbor = mesh.cells[neighbor_id]
                    
                    # If neighbor is more than one level lower than this cell would be after coarsening
                    if neighbor.level < cell.level - 1:
                        cell.can_coarsen = False
                        break


class UnstructuredAMRSystem:
    """
    System class for unstructured AMR integrated with existing PyCFD system
    """
    def __init__(self, nx=50, ny=50, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, gamma=1.4):
        # Initialize basic parameters
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.gamma = gamma
        
        # Create original System2D for initial condition and visualization
        self.base_system = System2D(nx, ny, xmin, xmax, ymin, ymax, gamma)
        
        # Create unstructured mesh
        self.mesh = self._create_initial_mesh()
        
        # Initialize AMR tracker
        self.gradient_tracker = AMRGradientTracker(
            threshold_refine=0.05,
            threshold_coarsen=0.01,
            variable_idx=0  # Track density gradients
        )
        
        # Create solver
        self.solver = EulerSolver(gamma)
        
        # For visualization
        self.output_dir = "amr_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _create_initial_mesh(self):
        """Create initial unstructured mesh from base system grid"""
        mesh = UnstructuredMesh()
        
        # For each cell in the base system
        for i in range(self.nx):
            for j in range(self.ny):
                # Get cell boundaries
                x_left = self.base_system.grid.x[i]
                x_right = self.base_system.grid.x[i+1]
                y_bottom = self.base_system.grid.y[j]
                y_top = self.base_system.grid.y[j+1]
                
                # Create cell vertices
                vertices = [
                    (x_left, y_bottom),
                    (x_right, y_bottom),
                    (x_right, y_top),
                    (x_left, y_top)
                ]
                
                # Add cell to mesh
                cell_id = mesh.add_cell(vertices)
                
                # Store grid indices for later reference
                mesh.cells[cell_id].grid_i = i
                mesh.cells[cell_id].grid_j = j
        
        # Set up neighbors
        for cell_id, cell in mesh.cells.items():
            i, j = cell.grid_i, cell.grid_j
            
            # Check all 4 neighbors
            neighbor_indices = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            ]
            
            for ni, nj in neighbor_indices:
                if 0 <= ni < self.nx and 0 <= nj < self.ny:
                    # Find corresponding cell in mesh
                    for nid, ncell in mesh.cells.items():
                        if hasattr(ncell, 'grid_i') and hasattr(ncell, 'grid_j'):
                            if ncell.grid_i == ni and ncell.grid_j == nj:
                                cell.neighbors.append(nid)
                                break
        
        return mesh
    
    def initialize_solution(self, initial_condition_func):
        """Initialize solution on the mesh"""
        # First initialize the base system
        for i in range(self.nx):
            for j in range(self.ny):
                # Cell center coordinates
                x = self.base_system.grid.x[i] + 0.5 * self.base_system.grid.dx
                y = self.base_system.grid.y[j] + 0.5 * self.base_system.grid.dy
                
                # Initialize with provided function
                U = initial_condition_func(x, y)
                
                # Set conserved variables
                self.base_system.grid.U[0, i, j] = U[0]  # Density
                self.base_system.grid.U[1, i, j] = U[1]  # X-Momentum
                self.base_system.grid.U[2, i, j] = U[2]  # Y-Momentum
                self.base_system.grid.U[3, i, j] = U[3]  # Energy
        
        # Then copy to unstructured mesh
        for cell_id, cell in self.mesh.cells.items():
            if hasattr(cell, 'grid_i') and hasattr(cell, 'grid_j'):
                i, j = cell.grid_i, cell.grid_j
                
                # Copy from base system
                cell.conserved_vars = np.array([
                    self.base_system.grid.U[0, i, j],
                    self.base_system.grid.U[1, i, j],
                    self.base_system.grid.U[2, i, j],
                    self.base_system.grid.U[3, i, j]
                ])
    
    def adapt_mesh(self, max_level=3):
        """Adapt the mesh based on solution features"""
        # Identify cells to refine/coarsen
        refine_list, coarsen_list = self.gradient_tracker.mark_cells_for_refinement(
            self.mesh, max_level
        )
        
        # Execute refinement
        for cell_id in refine_list:
            self.mesh.refine_cell(cell_id)
        
        # Execute coarsening (group by parent)
        parent_to_children = {}
        for cell_id in coarsen_list:
            cell = self.mesh.cells[cell_id]
            parent_id = cell.parent
            
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            
            parent_to_children[parent_id].append(cell_id)
        
        # Coarsen cell groups
        for parent_id, children in parent_to_children.items():
            # Only coarsen if all siblings are marked
            parent = self.mesh.cells[parent_id]
            if set(children) == set(parent.children):
                self.mesh.coarsen_cells(children)
    
    def calculate_time_step(self, cfl=0.5):
        """Calculate appropriate time step based on CFL condition"""
        dt_min = float('inf')
        
        # For each leaf cell
        for cell_id, cell in self.mesh.cells.items():
            if cell.children:
                continue  # Skip non-leaf cells
            
            if cell.conserved_vars is None:
                continue
            
            # Calculate characteristic length
            h = np.sqrt(cell.area)
            
            # Calculate wave speed
            rho = cell.conserved_vars[0]
            u = cell.conserved_vars[1] / rho if rho > 1e-10 else 0
            v = cell.conserved_vars[2] / rho if rho > 1e-10 else 0
            E = cell.conserved_vars[3]
            
            # Internal energy
            e_int = E / rho - 0.5 * (u*u + v*v) if rho > 1e-10 else 0
            
            # Pressure
            p = (self.gamma - 1.0) * rho * e_int
            
            # Sound speed
            c = np.sqrt(self.gamma * p / rho) if rho > 1e-10 and p > 0 else 0
            
            # Maximum wave speed
            wave_speed = np.sqrt(u*u + v*v) + c
            
            # Time step for this cell
            if wave_speed > 1e-10:
                dt = cfl * h / wave_speed
                dt_min = min(dt_min, dt)
        
        # If no valid time step was found, use a default
        if dt_min == float('inf'):
            dx = (self.xmax - self.xmin) / self.nx
            dy = (self.ymax - self.ymin) / self.ny
            min_dx = min(dx, dy)
            dt_min = cfl * min_dx
        
        return dt_min
    
    def advance_solution(self, dt):
        """Advance solution by one time step"""
        # Get all leaf cells
        leaf_cells = [cid for cid, cell in self.mesh.cells.items() if not cell.children]
        
        # Create dictionary to store updated solution
        updated_U = {}
        
        # For each leaf cell
        for cell_id in leaf_cells:
            cell = self.mesh.cells[cell_id]
            
            # Gather states from neighbors
            neighbor_states = []
            neighbor_positions = []
            
            for neighbor_id in cell.neighbors:
                if neighbor_id in self.mesh.cells:
                    neighbor = self.mesh.cells[neighbor_id]
                    
                    # If neighbor is refined, use appropriate child cells
                    if neighbor.children:
                        # Find child cells that are adjacent to this cell
                        for child_id in neighbor.children:
                            child = self.mesh.cells[child_id]
                            if self._cells_share_edge(cell, child):
                                neighbor_states.append(child.conserved_vars)
                                neighbor_positions.append(child.centroid)
                    else:
                        # Use neighbor directly
                        neighbor_states.append(neighbor.conserved_vars)
                        neighbor_positions.append(neighbor.centroid)
            
            # Update cell using modified version of EulerSolver
            new_U = self._update_cell(cell, neighbor_states, neighbor_positions, dt)
            updated_U[cell_id] = new_U
        
        # Apply updates
        for cell_id, new_U in updated_U.items():
            self.mesh.cells[cell_id].conserved_vars = new_U
    
    def _update_cell(self, cell, neighbor_states, neighbor_positions, dt):
        """Update a cell using a finite volume scheme"""
        # If no valid neighbors, return unchanged
        if not neighbor_states or cell.conserved_vars is None:
            return cell.conserved_vars
        
        # Get cell state
        U = cell.conserved_vars
        
        # Initialize residual
        dU = np.zeros_like(U)
        
        # Loop over all neighbors
        for neighbor_U, neighbor_pos in zip(neighbor_states, neighbor_positions):
            if neighbor_U is None:
                continue
            
            # Calculate vector from cell center to neighbor center
            dx = neighbor_pos[0] - cell.centroid[0]
            dy = neighbor_pos[1] - cell.centroid[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-10:
                continue
            
            # Unit normal vector
            nx = dx / dist
            ny = dy / dist
            
            # Calculate numerical flux
            flux = self._calculate_numerical_flux(U, neighbor_U, nx, ny)
            
            # Update residual
            dU -= flux * dt / dist
        
        # Return updated state
        return U + dU
    
    def _calculate_numerical_flux(self, UL, UR, nx, ny):
        """Calculate numerical flux between two states"""
        # This is a simple Rusanov/Local Lax-Friedrichs flux
        
        # Extract primitive variables
        rhoL, uL, vL, pL = self._conserved_to_primitive(UL)
        rhoR, uR, vR, pR = self._conserved_to_primitive(UR)
        
        # Normal velocities
        vnL = uL * nx + vL * ny
        vnR = uR * nx + vR * ny
        
        # Sound speeds
        cL = np.sqrt(self.gamma * pL / rhoL) if rhoL > 1e-10 and pL > 0 else 0
        cR = np.sqrt(self.gamma * pR / rhoR) if rhoR > 1e-10 and pR > 0 else 0
        
        # Maximum wave speed
        smax = max(abs(vnL) + cL, abs(vnR) + cR)
        
        # Calculate fluxes in normal direction
        FL = self._physical_flux(UL, nx, ny)
        FR = self._physical_flux(UR, nx, ny)
        
        # Local Lax-Friedrichs flux
        flux = 0.5 * (FL + FR - smax * (UR - UL))
        
        return flux
    
    def _physical_flux(self, U, nx, ny):
        """Calculate physical flux in the given direction"""
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
        """Convert conserved variables to primitive"""
        rho = U[0]
        u = U[1] / rho if rho > 1e-10 else 0
        v = U[2] / rho if rho > 1e-10 else 0
        E = U[3]
        
        # Internal energy
        e_int = E / rho - 0.5 * (u*u + v*v) if rho > 1e-10 else 0
        
        # Pressure
        p = (self.gamma - 1.0) * rho * e_int
        
        return rho, u, v, p
    
    def _cells_share_edge(self, cell1, cell2):
        """Check if two cells share an edge"""
        # Get all edges of both cells
        edges1 = []
        for i in range(len(cell1.vertices)):
            v1 = cell1.vertices[i]
            v2 = cell1.vertices[(i + 1) % len(cell1.vertices)]
            edges1.append((v1, v2))
        
        edges2 = []
        for i in range(len(cell2.vertices)):
            v1 = cell2.vertices[i]
            v2 = cell2.vertices[(i + 1) % len(cell2.vertices)]
            edges2.append((v1, v2))
        
        # Check if any edge is shared (in reverse order for one cell)
        for e1 in edges1:
            for e2 in edges2:
                # Check if e1 and e2 are the same edge but in opposite directions
                if (np.isclose(e1[0][0], e2[1][0]) and np.isclose(e1[0][1], e2[1][1]) and
                    np.isclose(e1[1][0], e2[0][0]) and np.isclose(e1[1][1], e2[0][1])):
                    return True
                
                # Check if e1 and e2 are the same edge in the same direction
                if (np.isclose(e1[0][0], e2[0][0]) and np.isclose(e1[0][1], e2[0][1]) and
                    np.isclose(e1[1][0], e2[1][0]) and np.isclose(e1[1][1], e2[1][1])):
                    return True
        
        return False
    
    def run_simulation(self, t_final, cfl=0.5, adapt_interval=5, save_interval=None):
        """Run simulation up to final time"""
        t = 0.0
        step = 0
        next_save = 0.0
        
        # Initial adaptation
        self.adapt_mesh()
        
        # Save initial state if requested
        if save_interval is not None:
            self.save_solution(step, t)
            next_save = save_interval
        
        # Main time loop
        while t < t_final:
            # Calculate time step
            dt = self.calculate_time_step(cfl)
            
            # Adjust for final time
            if t + dt > t_final:
                dt = t_final - t
            
            # Advance solution
            self.advance_solution(dt)
            
            # Update time
            t += dt
            step += 1
            
            # Adapt mesh periodically
            if step % adapt_interval == 0:
                self.adapt_mesh()
            
            # Save solution periodically
            if save_interval is not None and t >= next_save:
                self.save_solution(step, t)
                next_save += save_interval
            
            # Print progress
            print(f"Step: {step}, Time: {t:.6f}, Cells: {len([c for c in self.mesh.cells.values() if not c.children])}")
        
        # Final save
        if save_interval is not None:
            self.save_solution(step, t)
        
        return t, step
    
    def save_solution(self, step, time):
        """Save current solution to disk"""
        # Save visualization
        plt.figure(figsize=(12, 10))
        
        # Plot AMR mesh colored by refinement level
        ax1 = plt.subplot(2, 2, 1)
        self._plot_mesh(ax1, color_by_level=True)
        ax1.set_title(f'AMR Mesh (t = {time:.3f})')
        
        # Plot density
        ax2 = plt.subplot(2, 2, 2)
        self._plot_solution(ax2, var_idx=0)
        ax2.set_title('Density')
        
        # Plot pressure
        ax3 = plt.subplot(2, 2, 3)
        self._plot_solution(ax3, var_idx='pressure')
        ax3.set_title('Pressure')
        
        # Plot velocity magnitude
        ax4 = plt.subplot(2, 2, 4)
        self._plot_solution(ax4, var_idx='velocity')
        ax4.set_title('Velocity Magnitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'solution_{step:06d}.png'))
        plt.close()
        
        # Save VTK for ParaView
        self._save_vtk(os.path.join(self.output_dir, f'solution_{step:06d}'))
    
    def _plot_mesh(self, ax, color_by_level=True):
        """Plot AMR mesh"""
        # Define colors for different levels
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # Plot each cell
        for cell_id, cell in self.mesh.cells.items():
            if cell.children:
                continue  # Skip non-leaf cells
                
            # Get cell vertices
            vertices = cell.vertices
            # Close the polygon
            vertices = vertices + [vertices[0]]
            
            # Extract x and y coordinates
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            
            # Choose color based on level
            if color_by_level:
                color = colors[cell.level % len(colors)]
            else:
                color = 'b'
            
            # Plot cell
            ax.plot(x, y, color=color, linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_solution(self, ax, var_idx=0):
        """Plot solution on the mesh"""
        from matplotlib.tri import Triangulation
        
        # Prepare data for triangulation
        x = []
        y = []
        z = []
        triangles = []
        
        point_index = 0
        
        # Process each leaf cell
        for cell_id, cell in self.mesh.cells.items():
            if cell.children:
                continue  # Skip non-leaf cells
                
            if cell.conserved_vars is None:
                continue
                
            # Get cell vertices
            vertices = cell.vertices
            n_vertices = len(vertices)
            
            # Add vertices to the points list
            for vertex in vertices:
                x.append(vertex[0])
                y.append(vertex[1])
            
            # Get solution value
            if var_idx == 'pressure':
                # Calculate pressure
                rho, u, v, p = self._conserved_to_primitive(cell.conserved_vars)
                value = p
            elif var_idx == 'velocity':
                # Calculate velocity magnitude
                rho, u, v, p = self._conserved_to_primitive(cell.conserved_vars)
                value = np.sqrt(u*u + v*v)
            else:
                # Use specified conserved variable
                value = cell.conserved_vars[var_idx]
            
            # Assign the same value to all vertices of this cell
            for _ in range(n_vertices):
                z.append(value)
            
            # Create triangles (fan triangulation)
            for i in range(1, n_vertices - 1):
                triangles.append([
                    point_index,
                    point_index + i,
                    point_index + i + 1
                ])
            
            point_index += n_vertices
        
        # Create triangulation
        if len(triangles) > 0:
            triang = Triangulation(x, y, triangles)
            
            # Plot solution
            contour = ax.tripcolor(triang, z, cmap='viridis', shading='gouraud')
            
            # Add colorbar
            plt.colorbar(contour, ax=ax)
        
        ax.set_aspect('equal')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _save_vtk(self, filename):
        """Save solution in VTK format for ParaView"""
        try:
            import pyevtk
            from pyevtk.hl import unstructuredGridToVTK
            from pyevtk.vtk import VtkTriangle, VtkQuad
        except ImportError:
            print("pyevtk package not found. Cannot save VTK file.")
            return
        
        # Get leaf cells
        leaf_cells = [cid for cid, cell in self.mesh.cells.items() if not cell.children]
        
        # Skip if no leaf cells
        if not leaf_cells:
            return
        
        # Prepare points and connectivity
        points = []
        point_indices = {}  # Map vertex coordinates to index
        
        cell_types = []
        cell_offsets = []
        cell_conn = []
        
        # Cell data
        density = []
        pressure = []
        velocity_x = []
        velocity_y = []
        level = []
        
        # Process each leaf cell
        offset = 0
        for cell_id in leaf_cells:
            cell = self.mesh.cells[cell_id]
            
            # Skip cells without solution
            if cell.conserved_vars is None:
                continue
            
            # Process vertices
            cell_points = []
            for vertex in cell.vertices:
                # Convert to 3D point
                point = (vertex[0], vertex[1], 0.0)
                
                # Add to points list if not already present
                if point not in point_indices:
                    point_indices[point] = len(points)
                    points.append(point)
                
                # Add to cell connectivity
                cell_points.append(point_indices[point])
            
            # Add cell connectivity
            cell_conn.extend(cell_points)
            
            # Set cell type
            if len(cell.vertices) == 3:
                cell_types.append(VtkTriangle.tid)
            elif len(cell.vertices) == 4:
                cell_types.append(VtkQuad.tid)
            else:
                # Skip other cell types
                cell_conn = cell_conn[:-len(cell_points)]
                continue
            
            # Update offset
            offset += len(cell_points)
            cell_offsets.append(offset)
            
            # Extract solution data
            rho, u, v, p = self._conserved_to_primitive(cell.conserved_vars)
            
            # Add cell data
            density.append(rho)
            pressure.append(p)
            velocity_x.append(u)
            velocity_y.append(v)
            level.append(cell.level)
        
        # Skip if no valid cells
        if not cell_types:
            return
        
        # Extract point coordinates
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]
        
        # Prepare cell data
        cell_data = {
            "density": np.array(density),
            "pressure": np.array(pressure),
            "velocity_x": np.array(velocity_x),
            "velocity_y": np.array(velocity_y),
            "level": np.array(level)
        }
        
        # Write VTK file
        unstructuredGridToVTK(
            filename,
            np.array(x),
            np.array(y),
            np.array(z),
            connectivity=np.array(cell_conn),
            offsets=np.array(cell_offsets),
            cell_types=np.array(cell_types),
            cellData=cell_data
        )
        
        print(f"Saved VTK file: {filename}.vtu")


def blast_wave_initial_condition(x, y):
    """Initial condition for a blast wave test case"""
    # Domain center
    x_c, y_c = 0.5, 0.5
    
    # Distance from center
    r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
    
    # Initial condition
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


def shock_bubble_initial_condition(x, y):
    """Initial condition for a shock-bubble interaction test case"""
    # Domain parameters
    x_shock = 0.2
    x_bubble = 0.5
    y_bubble = 0.5
    r_bubble = 0.15
    
    # Check if point is inside the bubble
    r = np.sqrt((x - x_bubble)**2 + (y - y_bubble)**2)
    in_bubble = r < r_bubble
    
    # Check if point is left of the shock
    left_of_shock = x < x_shock
    
    # Set up initial condition
    if left_of_shock:
        # Post-shock state
        rho = 3.81
        u = 2.58
        v = 0.0
        p = 10.33
    else:
        if in_bubble:
            # Bubble state (helium)
            rho = 0.138
            u = 0.0
            v = 0.0
            p = 1.0
        else:
            # Pre-shock state (air)
            rho = 1.0
            u = 0.0
            v = 0.0
            p = 1.0
    
    # Convert to conserved variables
    gamma = 1.4
    E = p / (gamma - 1) + 0.5 * rho * (u*u + v*v)
    
    return np.array([rho, rho*u, rho*v, E])


def kelvin_helmholtz_initial_condition(x, y):
    """Initial condition for a Kelvin-Helmholtz instability test case"""
    # Domain parameters
    y_mid = 0.5
    thickness = 0.05
    
    # Base flow
    if y <= y_mid - thickness/2:
        # Lower layer
        rho = 2.0
        u = -0.5
        perturbation = 0.01 * np.sin(4 * np.pi * x)
    elif y >= y_mid + thickness/2:
        # Upper layer
        rho = 1.0
        u = 0.5
        perturbation = 0.01 * np.sin(4 * np.pi * x)
    else:
        # Transition layer
        t = (y - (y_mid - thickness/2)) / thickness
        rho = 2.0 + t * (1.0 - 2.0)
        u = -0.5 + t * (0.5 - (-0.5))
        perturbation = 0.01 * np.sin(4 * np.pi * x)
    
    # Add perturbation to y-velocity
    v = perturbation
    
    # Pressure (constant)
    p = 2.5
    
    # Convert to conserved variables
    gamma = 1.4
    E = p / (gamma - 1) + 0.5 * rho * (u*u + v*v)
    
    return np.array([rho, rho*u, rho*v, E])


def main():
    """Main function to run the AMR simulation"""
    # Create AMR system
    system = UnstructuredAMRSystem(
        nx=50, ny=50,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        gamma=1.4
    )
    
    # Initialize with blast wave test case
    system.initialize_solution(blast_wave_initial_condition)
    
    # Run simulation
    t_final = 0.25
    t, steps = system.run_simulation(
        t_final,
        cfl=0.5,
        adapt_interval=5,
        save_interval=0.05
    )
    
    print(f"Simulation completed: time = {t:.6f}, steps = {steps}")
    print(f"Total cells: {len(system.mesh.cells)}")
    print(f"Leaf cells: {len([c for c in system.mesh.cells.values() if not c.children])}")


if __name__ == "__main__":
    main()