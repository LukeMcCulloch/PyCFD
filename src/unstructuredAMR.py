#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:53:15 2025

@author: lukemcculloch


Unstructured grid adaptive mesh refinement for PyCFD
"""
import math as math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

class UnstructuredCell:
    """
    Represents a single cell in an unstructured mesh
    """
    def __init__(self, cell_id, vertices, neighbors=None, level=0, parent=None):
        self.cell_id = cell_id
        self.vertices = vertices  # List of (x, y) coordinates for cell vertices
        self.neighbors = neighbors if neighbors is not None else []  # List of neighbor cell IDs
        self.level = level  # Refinement level
        self.parent = parent  # Parent cell ID if this is a refined cell
        self.children = []  # List of child cell IDs if this cell is refined
        self.centroid = self._compute_centroid()
        self.area = self._compute_area()
        self.needs_refinement = False
        self.can_coarsen = False
        self.conserved_vars = None  # Will store U values for this cell
    
    def _compute_centroid(self):
        """Compute cell centroid"""
        return np.mean(self.vertices, axis=0)
    
    def _compute_area(self):
        """Compute cell area using shoelace formula"""
        x = [v[0] for v in self.vertices]
        y = [v[1] for v in self.vertices]
        return 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - 
                           x[(i + 1) % len(x)] * y[i] 
                           for i in range(len(x))))
    
    def get_edge_midpoints(self):
        """Get midpoints of all edges of the cell"""
        midpoints = []
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            midpoint = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
            midpoints.append(midpoint)
        return midpoints
    
    def get_faces(self):
        """Get all faces (edges) of the cell"""
        faces = []
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            faces.append((v1, v2))
        return faces


class UnstructuredMesh:
    """
    Represents an unstructured mesh with adaptive refinement capability
    """
    def __init__(self):
        self.cells = {}  # Dictionary mapping cell_id to UnstructuredCell
        self.next_cell_id = 0
        self.max_level = 0
        self.num_vars = 4  # For Euler equations: [rho, rho*u, rho*v, E]
    
    def add_cell(self, vertices, neighbors=None, level=0, parent=None):
        """Add a new cell to the mesh"""
        cell = UnstructuredCell(self.next_cell_id, vertices, neighbors, level, parent)
        self.cells[self.next_cell_id] = cell
        self.next_cell_id += 1
        self.max_level = max(self.max_level, level)
        return cell.cell_id
    
    def initialize_from_structured_grid(self, nx, ny, x_min, x_max, y_min, y_max):
        """Initialize mesh from a structured grid"""
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        
        # Create cells
        for i in range(nx):
            for j in range(ny):
                # Cell vertices (counterclockwise order)
                vertices = [
                    (x_min + i * dx, y_min + j * dy),
                    (x_min + (i + 1) * dx, y_min + j * dy),
                    (x_min + (i + 1) * dx, y_min + (j + 1) * dy),
                    (x_min + i * dx, y_min + (j + 1) * dy)
                ]
                
                # Add cell
                cell_id = self.add_cell(vertices)
                
                # Store i,j indices for easier neighbor identification
                self.cells[cell_id].i = i
                self.cells[cell_id].j = j
        
        # Set up neighbors
        for cell_id, cell in self.cells.items():
            i, j = cell.i, cell.j
            
            # Check all 4 neighboring cells
            neighbor_indices = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            ]
            
            for ni, nj in neighbor_indices:
                if 0 <= ni < nx and 0 <= nj < ny:
                    # Find cell_id of neighbor
                    for nid, ncell in self.cells.items():
                        if hasattr(ncell, 'i') and hasattr(ncell, 'j') and ncell.i == ni and ncell.j == nj:
                            cell.neighbors.append(nid)
                            break
    
    def initialize_solution(self, initial_condition_func):
        """Initialize solution on the mesh using a provided function"""
        for cell_id, cell in self.cells.items():
            # Get cell centroid
            x, y = cell.centroid
            
            # Initialize conserved variables
            cell.conserved_vars = initial_condition_func(x, y)
    
    def refine_cell(self, cell_id):
        """Refine a single cell into multiple cells"""
        parent_cell = self.cells[cell_id]
        
        if parent_cell.children:
            # Cell is already refined
            return
        
        # Get cell vertices and centroid
        vertices = parent_cell.vertices
        centroid = parent_cell.centroid
        
        # Get edge midpoints
        edge_midpoints = parent_cell.get_edge_midpoints()
        
        # Create new cells (for a quadrilateral cell, we create 4 new cells)
        new_cell_ids = []
        
        # For each vertex, create a new cell with:
        # - The original vertex
        # - The midpoints of the two edges connected to this vertex
        # - The cell centroid
        for i in range(len(vertices)):
            v = vertices[i]
            m1 = edge_midpoints[i]
            m2 = edge_midpoints[(i - 1) % len(vertices)]
            
            new_vertices = [v, m1, centroid, m2]
            new_cell_id = self.add_cell(
                new_vertices, 
                level=parent_cell.level + 1,
                parent=cell_id
            )
            new_cell_ids.append(new_cell_id)
            
            # Add to parent's children list
            parent_cell.children.append(new_cell_id)
        
        # Update the maximum refinement level
        self.max_level = max(self.max_level, parent_cell.level + 1)
        
        # Set up neighbors for the new cells
        self._update_neighbors_after_refinement(cell_id, new_cell_ids)
        
        # Interpolate solution to children
        self._interpolate_solution_to_children(cell_id)
        
        return new_cell_ids
    
    def _update_neighbors_after_refinement(self, parent_id, child_ids):
        """Update neighbor relationships after refinement"""
        parent_cell = self.cells[parent_id]
        
        # First, make the child cells neighbors of each other
        num_children = len(child_ids)
        for i, child_id in enumerate(child_ids):
            next_child = child_ids[(i + 1) % num_children]
            prev_child = child_ids[(i - 1) % num_children]
            
            self.cells[child_id].neighbors.extend([next_child, prev_child])
        
        # Now handle external neighbors
        parent_neighbors = parent_cell.neighbors
        for neighbor_id in parent_neighbors:
            neighbor = self.cells[neighbor_id]
            
            if not neighbor.children:  # Neighbor is not refined
                # Find which child cells are adjacent to this neighbor
                for child_id in child_ids:
                    child = self.cells[child_id]
                    
                    # If child shares an edge with neighbor, they are neighbors
                    if self._cells_share_edge(child, neighbor):
                        child.neighbors.append(neighbor_id)
                        
                        # Also update the neighbor's neighbors
                        if parent_id in neighbor.neighbors:
                            neighbor.neighbors.remove(parent_id)
                        neighbor.neighbors.append(child_id)
            else:  # Neighbor is refined
                # We need to connect child cells to the appropriate neighbor's children
                for child_id in child_ids:
                    child = self.cells[child_id]
                    
                    for neighbor_child_id in neighbor.children:
                        neighbor_child = self.cells[neighbor_child_id]
                        
                        if self._cells_share_edge(child, neighbor_child):
                            child.neighbors.append(neighbor_child_id)
                            neighbor_child.neighbors.append(child_id)
    
    def _cells_share_edge(self, cell1, cell2):
        """Check if two cells share an edge"""
        # Get all edges of both cells
        edges1 = self._get_cell_edges(cell1)
        edges2 = self._get_cell_edges(cell2)
        
        # Check if any edge is shared (in reverse order for the second cell)
        for e1 in edges1:
            for e2 in edges2:
                if (np.isclose(e1[0][0], e2[1][0]) and np.isclose(e1[0][1], e2[1][1]) and
                    np.isclose(e1[1][0], e2[0][0]) and np.isclose(e1[1][1], e2[0][1])):
                    return True
                    
                if (np.isclose(e1[0][0], e2[0][0]) and np.isclose(e1[0][1], e2[0][1]) and
                    np.isclose(e1[1][0], e2[1][0]) and np.isclose(e1[1][1], e2[1][1])):
                    return True
                    
        return False
    
    def _get_cell_edges(self, cell):
        """Get all edges of a cell"""
        edges = []
        for i in range(len(cell.vertices)):
            v1 = cell.vertices[i]
            v2 = cell.vertices[(i + 1) % len(cell.vertices)]
            edges.append((v1, v2))
        return edges
    
    def _interpolate_solution_to_children(self, parent_id):
        """Interpolate solution from parent cell to children"""
        parent = self.cells[parent_id]
        parent_u = parent.conserved_vars
        
        # For simplicity, we'll just copy the parent's solution to all children
        # In a more advanced implementation, you might use linear or higher-order interpolation
        for child_id in parent.children:
            child = self.cells[child_id]
            child.conserved_vars = np.copy(parent_u)
    
    def coarsen_cells(self, cell_ids):
        """Coarsen a set of cells (replace them with their parent)"""
        if not cell_ids:
            return None
        
        # Get parent cell
        parent_id = self.cells[cell_ids[0]].parent
        if parent_id is None:
            return None  # Can't coarsen base level cells
        
        parent = self.cells[parent_id]
        
        # Make sure all cells are children of the same parent
        for cell_id in cell_ids:
            if self.cells[cell_id].parent != parent_id:
                return None
        
        # Make sure all children of the parent are in the list
        if set(parent.children) != set(cell_ids):
            return None
        
        # Project solution from children to parent
        self._project_solution_to_parent(parent_id)
        
        # Update neighbor relationships
        self._update_neighbors_after_coarsening(parent_id)
        
        # Clear children list
        parent.children = []
        
        # Remove child cells
        for cell_id in cell_ids:
            del self.cells[cell_id]
        
        return parent_id
    
    def _project_solution_to_parent(self, parent_id):
        """Project solution from children to parent (conservation)"""
        parent = self.cells[parent_id]
        
        # Get all children
        child_cells = [self.cells[child_id] for child_id in parent.children]
        
        # Initialize parent solution to zeros
        if parent.conserved_vars is None:
            parent.conserved_vars = np.zeros(self.num_vars)
        else:
            parent.conserved_vars[:] = 0
        
        # Compute weighted average of children solutions
        total_area = sum(child.area for child in child_cells)
        
        for child in child_cells:
            weight = child.area / total_area
            parent.conserved_vars += weight * child.conserved_vars
    
    def _update_neighbors_after_coarsening(self, parent_id):
        """Update neighbor relationships after coarsening"""
        parent = self.cells[parent_id]
        
        # Clear parent neighbors
        parent.neighbors = []
        
        # Find all neighbors of parent cell
        for cell_id, cell in self.cells.items():
            if cell_id != parent_id and not cell_id in parent.children:
                # Check if this cell is adjacent to parent
                if any(n == parent_id for n in cell.neighbors):
                    parent.neighbors.append(cell_id)
                    
                # If this cell has any of parent's children as neighbors, update it
                child_neighbors = [n for n in cell.neighbors if n in parent.children]
                if child_neighbors:
                    for child_id in child_neighbors:
                        cell.neighbors.remove(child_id)
                    if parent_id not in cell.neighbors:
                        cell.neighbors.append(parent_id)
    
    def mark_cells_for_refinement(self, criteria_func):
        """Mark cells for refinement based on a criteria function"""
        for cell_id, cell in self.cells.items():
            if not cell.children:  # Only consider leaf cells
                cell.needs_refinement = criteria_func(cell)
    
    def mark_cells_for_coarsening(self, criteria_func):
        """Mark cells for coarsening based on a criteria function"""
        # Group cells by parent
        parent_to_children = {}
        for cell_id, cell in self.cells.items():
            if cell.parent is not None and not cell.children:  # Only consider leaf cells with a parent
                if cell.parent not in parent_to_children:
                    parent_to_children[cell.parent] = []
                parent_to_children[cell.parent].append(cell_id)
        
        # Check each group of siblings
        for parent_id, child_ids in parent_to_children.items():
            # Only consider complete sets of siblings
            if len(child_ids) == len(self.cells[parent_id].children):
                # Check if all children can be coarsened
                if all(criteria_func(self.cells[child_id]) for child_id in child_ids):
                    for child_id in child_ids:
                        self.cells[child_id].can_coarsen = True
    
    def execute_refinement(self):
        """Execute refinement for all marked cells"""
        cells_to_refine = [cell_id for cell_id, cell in self.cells.items() 
                          if cell.needs_refinement and not cell.children]
        
        for cell_id in cells_to_refine:
            self.refine_cell(cell_id)
            
        # Reset refinement flags
        for cell in self.cells.values():
            cell.needs_refinement = False
    
    def execute_coarsening(self):
        """Execute coarsening for all marked cells"""
        # Group cells by parent
        parent_to_children = {}
        for cell_id, cell in self.cells.items():
            if cell.can_coarsen:
                if cell.parent not in parent_to_children:
                    parent_to_children[cell.parent] = []
                parent_to_children[cell.parent].append(cell_id)
        
        # Coarsen each group
        for parent_id, child_ids in parent_to_children.items():
            # Only coarsen if all siblings are marked for coarsening
            if len(child_ids) == len(self.cells[parent_id].children):
                self.coarsen_cells(child_ids)
        
        # Reset coarsening flags
        for cell in self.cells.values():
            cell.can_coarsen = False
    
    def get_leaf_cells(self):
        """Get all leaf cells (cells without children)"""
        return [cell_id for cell_id, cell in self.cells.items() if not cell.children]
    
    def plot_mesh(self, ax=None, color_by_level=True, show_cell_ids=False):
        """Plot the mesh"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            standalone = True
        else:
            standalone = False
        
        # Define colors for different levels
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # Plot each cell
        for cell_id, cell in self.cells.items():
            if not cell.children:  # Only plot leaf cells
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
                
                # Show cell ID if requested
                if show_cell_ids:
                    cx, cy = cell.centroid
                    ax.text(cx, cy, str(cell_id), ha='center', va='center', fontsize=8)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Unstructured Mesh')
        
        if standalone:
            plt.tight_layout()
            plt.show()
    
    def plot_solution(self, variable_index=0, ax=None, cmap='viridis'):
        """Plot solution on the mesh"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            standalone = True
        else:
            standalone = False
        
        # Prepare data for triangulation
        x = []
        y = []
        z = []
        triangles = []
        
        point_index = 0
        
        # Process each leaf cell
        for cell_id, cell in self.cells.items():
            if not cell.children:  # Only consider leaf cells
                # Get cell vertices
                vertices = cell.vertices
                n_vertices = len(vertices)
                
                # Add vertices to the points list
                for vertex in vertices:
                    x.append(vertex[0])
                    y.append(vertex[1])
                
                # Get solution value at cell centroid
                value = cell.conserved_vars[variable_index]
                
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
        triang = Triangulation(x, y, triangles)
        
        # Plot solution
        contour = ax.tripcolor(triang, z, cmap=cmap, shading='gouraud')
        
        # Add colorbar
        plt.colorbar(contour, ax=ax)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        variable_names = ["Density", "X-Momentum", "Y-Momentum", "Energy"]
        if variable_index < len(variable_names):
            ax.set_title(f'{variable_names[variable_index]} Distribution')
        else:
            ax.set_title(f'Variable {variable_index} Distribution')
        
        if standalone:
            plt.tight_layout()
            plt.show()


class UnstructuredAMR:
    """
    Class for adaptive mesh refinement on unstructured grids
    """
    def __init__(self, mesh, max_level=3):
        self.mesh = mesh
        self.max_level = max_level
        self.refinement_threshold = 0.1  # Default threshold
        self.coarsening_threshold = 0.01  # Default threshold
    
    def set_refinement_threshold(self, threshold):
        """Set threshold for refinement decision"""
        self.refinement_threshold = threshold
    
    def set_coarsening_threshold(self, threshold):
        """Set threshold for coarsening decision"""
        self.coarsening_threshold = threshold
    
    def refinement_criteria(self, cell):
        """Default criteria for cell refinement based on solution gradients"""
        if cell.level >= self.max_level:
            return False
        
        # Calculate approximate gradient at this cell
        grad_magnitude = self._estimate_gradient_magnitude(cell)
        
        return grad_magnitude > self.refinement_threshold
    
    def coarsening_criteria(self, cell):
        """Default criteria for cell coarsening based on solution smoothness"""
        if cell.parent is None:
            return False  # Base level cells cannot be coarsened
        
        # Calculate approximate gradient at this cell
        grad_magnitude = self._estimate_gradient_magnitude(cell)
        
        return grad_magnitude < self.coarsening_threshold
    
    def _estimate_gradient_magnitude(self, cell):
        """Estimate gradient magnitude at a cell"""
        # Get cell's conserved variables
        U = cell.conserved_vars
        
        if U is None:
            return 0.0
        
        # Simple gradient estimate based on neighbors
        gradient_sum = 0.0
        
        # Collect neighbor values
        n_neighbors = 0
        for neighbor_id in cell.neighbors:
            neighbor = self.mesh.cells[neighbor_id]
            
            # Skip if neighbor is not a leaf cell (has children)
            if neighbor.children:
                continue
                
            # Get neighbor's conserved variables
            neighbor_U = neighbor.conserved_vars
            
            if neighbor_U is None:
                continue
                
            # Calculate distance between cell centroids
            dx = neighbor.centroid[0] - cell.centroid[0]
            dy = neighbor.centroid[1] - cell.centroid[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-10:
                continue
                
            # Calculate gradient for each variable and sum their magnitudes
            for var_idx in range(len(U)):
                du = (neighbor_U[var_idx] - U[var_idx]) / dist
                gradient_sum += abs(du)
            
            n_neighbors += 1
        
        # Return average gradient magnitude
        if n_neighbors > 0:
            return gradient_sum / (n_neighbors * len(U))
        else:
            return 0.0
    
    def adapt_mesh(self):
        """Execute one adaptation cycle"""
        # Mark cells for refinement and coarsening
        self.mesh.mark_cells_for_refinement(self.refinement_criteria)
        self.mesh.mark_cells_for_coarsening(self.coarsening_criteria)
        
        # Execute refinement first, then coarsening
        self.mesh.execute_refinement()
        self.mesh.execute_coarsening()
    
    def advance_solution(self, solver, dt):
        """Advance solution using provided solver"""
        # Get all leaf cells
        leaf_cells = self.mesh.get_leaf_cells()
        
        # Create separate time step size for each level
        level_dt = {}
        for level in range(self.mesh.max_level + 1):
            level_dt[level] = dt / (2**level)  # Each level uses half the time step of the level above
        
        # Advance solution level by level, starting from the coarsest
        for level in range(self.mesh.max_level + 1):
            # Get cells at this level
            level_cells = [cell_id for cell_id in leaf_cells 
                          if self.mesh.cells[cell_id].level == level]
            
            # Number of subcycles for this level
            subcycles = 2**level
            
            # Subcycle
            for _ in range(subcycles):
                # Update each cell
                for cell_id in level_cells:
                    cell = self.mesh.cells[cell_id]
                    
                    # Get neighbor states
                    neighbor_states = []
                    neighbor_positions = []
                    
                    for neighbor_id in cell.neighbors:
                        neighbor = self.mesh.cells[neighbor_id]
                        
                        # If neighbor has children, use appropriate child cells
                        if neighbor.children:
                            # Find children that are adjacent to this cell
                            for child_id in neighbor.children:
                                child = self.mesh.cells[child_id]
                                if self.mesh._cells_share_edge(cell, child):
                                    neighbor_states.append(child.conserved_vars)
                                    neighbor_positions.append(child.centroid)
                        else:
                            neighbor_states.append(neighbor.conserved_vars)
                            neighbor_positions.append(neighbor.centroid)
                    
                    # Call solver to update the cell
                    new_state = solver.update_cell(
                        cell.conserved_vars,
                        cell.centroid,
                        neighbor_states,
                        neighbor_positions,
                        level_dt[level]
                    )
                    
                    # Update cell state
                    cell.conserved_vars = new_state
    
    def run_simulation(self, solver, t_final, dt_base=None, adapt_interval=5):
        """Run a simulation with AMR"""
        t = 0.0
        step = 0
        
        if dt_base is None:
            # Calculate a suitable time step based on CFL condition
            dt_base = self._calculate_suitable_time_step(solver)
        
        # Main time loop
        while t < t_final:
            # Adjust dt for the last step
            dt = min(dt_base, t_final - t)
            
            # Advance solution
            self.advance_solution(solver, dt)
            
            # Update time
            t += dt
            step += 1
            
            # Adapt mesh at specified intervals
            if step % adapt_interval == 0:
                self.adapt_mesh()
                
                # Recalculate time step after adaptation
                dt_base = self._calculate_suitable_time_step(solver)
            
            # Print progress
            print(f"Step: {step}, Time: {t:.6f}, Cells: {len(self.mesh.get_leaf_cells())}")
        
        return t, step
    
    def _calculate_suitable_time_step(self, solver):
        """Calculate a suitable time step based on CFL condition"""
        min_dx = float('inf')
        max_wave_speed = 0.0
        
        # Get all leaf cells
        leaf_cells = self.mesh.get_leaf_cells()
        
        # Find minimum cell size
        for cell_id in leaf_cells:
            cell = self.mesh.cells[cell_id]
            
            # Estimate cell size as sqrt(area)
            dx_estimate = np.sqrt(cell.area)
            min_dx = min(min_dx, dx_estimate)
            
            # Estimate wave speed in this cell
            if cell.conserved_vars is not None:
                if (cell.conserved_vars[3] < 0.0):
                    print('\n\n Debugging nan')
                    print('cell_id = {},  p = {}'.format(cell_id, cell.conserved_vars[3]))
                    #exit()
                    1.0/0.0
                wave_speed = solver.estimate_wave_speed(cell.conserved_vars)
                max_wave_speed = max(max_wave_speed, wave_speed)
            
            if (math.isnan(wave_speed)):
                print('\n\n Debugging nan')
                print('cell.conserved_vars = {}'.format(cell.conserved_vars))
                print('cell_id = {}, wave_speed = {}'.format(cell_id, wave_speed))
                1./0.
        # Prevent division by zero
        if max_wave_speed < 1e-10:
            max_wave_speed = 1.0
        
        # Calculate dt using CFL condition
        cfl = 0.5  # Conservative CFL number
        dt = cfl * min_dx / max_wave_speed
        
        return dt


class SimplifiedUnstructuredSolver:
    """
    A simplified solver for demonstration purposes
    """
    def __init__(self, gamma=1.4):
        self.gamma = gamma
    
    def update_cell(self, U, position, neighbor_states, neighbor_positions, dt):
        """
        Update a cell's state based on its neighbors
        This is a very simplified solver for demonstration
        """
        # For a real CFD solver, you would implement a proper flux calculation
        # but for demonstration, we'll use a simple update based on averages
        
        # If no neighbors, return unchanged
        if not neighbor_states:
            return np.copy(U)
        
        # Calculate flux contribution from each neighbor
        dU = np.zeros_like(U)
        
        for i, (neighbor_U, neighbor_pos) in enumerate(zip(neighbor_states, neighbor_positions)):
            # Calculate direction vector
            dx = neighbor_pos[0] - position[0]
            dy = neighbor_pos[1] - position[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-10:
                continue
                
            # Normalize
            nx = dx / dist
            ny = dy / dist
            
            # Calculate primitive variables for both cells
            rho, u, v, p = self._conserved_to_primitive(U)
            rho_n, u_n, v_n, p_n = self._conserved_to_primitive(neighbor_U)
            
            # Simple upwind flux
            velocity_normal = u * nx + v * ny
            velocity_normal_n = u_n * nx + v_n * ny
            
            if velocity_normal + velocity_normal_n >= 0:
                # Flow is from cell to neighbor (outflow)
                flux = self._calculate_flux(U, nx, ny)
            else:
                # Flow is from neighbor to cell (inflow)
                flux = self._calculate_flux(neighbor_U, nx, ny)
            
            # Add contribution to dU
            dU -= flux * dt / dist
        
        # Return updated state
        return U + dU
    
    def _conserved_to_primitive(self, U):
        """Convert conserved variables to primitive"""
        rho = U[0]
        u = U[1] / rho if rho > 1e-10 else 0.0
        v = U[2] / rho if rho > 1e-10 else 0.0
        E = U[3]
        
        # Internal energy
        e_int = E / rho - 0.5 * (u*u + v*v)
        
        # Pressure
        p = (self.gamma - 1.0) * rho * e_int
        
        return rho, u, v, p
    
    def _primitive_to_conserved(self, rho, u, v, p):
        """Convert primitive variables to conserved"""
        # Energy
        e_int = p / ((self.gamma - 1.0) * rho) if rho > 1e-10 else 0
        E = rho * (e_int + 0.5 * (u*u + v*v))
        
        return np.array([rho, rho*u, rho*v, E])
    
    def _calculate_flux(self, U, nx, ny):
        """Calculate flux in the given direction"""
        rho, u, v, p = self._conserved_to_primitive(U)
        
        # Normal velocity
        v_n = u * nx + v * ny
        
        # Calculate flux
        mass_flux = rho * v_n
        momentum_x_flux = rho * u * v_n + p * nx
        momentum_y_flux = rho * v * v_n + p * ny
        energy_flux = (U[3] + p) * v_n
        
        return np.array([mass_flux, momentum_x_flux, momentum_y_flux, energy_flux])
    
    def estimate_wave_speed(self, U):
        """Estimate maximum wave speed for a cell
        
        
        U = np.asarray([1.01194126 , -6.00703981  ,-5.29210918 ,-56.6350399])
        rho, u, v, p = self._conserved_to_primitive(U)
        
        
        self = amr
        self = solver
        """
        rho, u, v, p = self._conserved_to_primitive(U)
        
        # Sound speed
        if (rho < 0.0):
            print('rho = {}'.format(rho))
        c = np.sqrt(self.gamma * p / rho) #if rho > 1e-10 else 0.0
        
        # Maximum wave speed
        operand = u*u + v*v
        return np.sqrt(operand) + c if operand > 1.e-10 else c


class ShockCellRefinementCriteria:
    """
    Criteria for refining cells near shocks
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold
    
    def __call__(self, cell):
        """Check if cell should be refined"""
        if cell.conserved_vars is None or len(cell.neighbors) == 0:
            return False
        
        # Check density gradient
        rho = cell.conserved_vars[0]
        
        # Calculate average and maximum neighbor density
        neighbor_densities = []
        for neighbor_id in cell.neighbors:
            neighbor = cell.mesh.cells[neighbor_id]
            if neighbor.conserved_vars is not None:
                neighbor_densities.append(neighbor.conserved_vars[0])
        
        if not neighbor_densities:
            return False
        
        # Calculate density gradient
        max_density_diff = max([abs(rho - n_rho) for n_rho in neighbor_densities])
        max_density = max([rho] + neighbor_densities)
        
        # Normalize by maximum density
        if max_density > 1e-10:
            normalized_diff = max_density_diff / max_density
            return normalized_diff > self.threshold
        else:
            return False


def create_unstructured_mesh_from_structured(nx, ny, x_min, x_max, y_min, y_max):
    """Create an unstructured mesh starting from a structured grid"""
    mesh = UnstructuredMesh()
    mesh.initialize_from_structured_grid(nx, ny, x_min, x_max, y_min, y_max)
    return mesh


def shock_tube_initial_condition(x, y):
    """Initial condition for 2D shock tube problem"""
    if x < 0.5:
        # Left state
        rho = 1.0
        u = 0.0
        v = 0.0
        p = 1.0
    else:
        # Right state
        rho = 0.125
        u = 0.0
        v = 0.0
        p = 0.1
    
    # Convert to conserved variables
    gamma = 1.4
    E = p / (gamma - 1) + 0.5 * rho * (u*u + v*v)
    
    return np.array([rho, rho*u, rho*v, E])


def blast_wave_initial_condition(x, y):
    """Initial condition for 2D blast wave problem"""
    # Center of the domain
    x_c = 0.5
    y_c = 0.5
    
    # Distance from center
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


def main():
    """Main function to demonstrate AMR on unstructured mesh"""
    # Create mesh
    nx, ny = 40, 40
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    mesh = create_unstructured_mesh_from_structured(nx, ny, x_min, x_max, y_min, y_max)
    
    # Initialize solution
    mesh.initialize_solution(blast_wave_initial_condition)
    
    # Create solver
    solver = SimplifiedUnstructuredSolver(gamma=1.4)
    
    # Create AMR controller
    amr = UnstructuredAMR(mesh, max_level=3)
    amr.set_refinement_threshold(0.05)  # Lower for more refinement
    amr.set_coarsening_threshold(0.01)
    
    # Plot initial mesh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    mesh.plot_mesh(ax=ax1, color_by_level=True)
    mesh.plot_solution(variable_index=0, ax=ax2)  # Plot density
    plt.tight_layout()
    plt.savefig('initial_mesh.png')
    plt.close()
    
    # Initial mesh adaptation
    amr.adapt_mesh()
    
    # Plot refined mesh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    mesh.plot_mesh(ax=ax1, color_by_level=True)
    mesh.plot_solution(variable_index=0, ax=ax2)  # Plot density
    plt.tight_layout()
    plt.savefig('refined_mesh.png')
    plt.close()
    
    # Run simulation
    #t_final = 0.2
    t_final = 0.02
    t, steps = amr.run_simulation(solver, t_final, adapt_interval=5)
    
    # Plot final results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    mesh.plot_mesh(ax=ax1, color_by_level=True)
    mesh.plot_solution(variable_index=0, ax=ax2)  # Plot density
    ax1.set_title(f'Final Mesh (t = {t:.3f}, {len(mesh.get_leaf_cells())} cells)')
    ax2.set_title(f'Density at t = {t:.3f}')
    plt.tight_layout()
    plt.savefig('final_results.png')
    plt.close()
    
    print(f"Simulation completed: time = {t:.6f}, steps = {steps}")
    print(f"Total cells: {len(mesh.cells)}")
    print(f"Leaf cells: {len(mesh.get_leaf_cells())}")
    print(f"Maximum refinement level: {mesh.max_level}")


if __name__ == "__main__":
    '''
    U = np.asarray([1.01194126  -6.00703981  -5.29210918 -56.6350399])
    rho, u, v, p = self._conserved_to_primitive(U)
    self = amr
    self = solver
    '''
    main()