#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 14:32:58 2025

@author: lukemcculloch

 Methodology reference:
 The gradient-based error indicator follows the approach
 of using the magnitude of reconstructed solution gradients to drive
 refinement [see Berger & Colella, J. Comput. Phys. 82(1989) and
 Mavriplis, J. Comput. Phys. 128(1996)].

"""

import numpy as np

# Adaptive Mesh Refinement (AMR) module

class AMR:
    def __init__(self, mesh, state_w, refine_thresh, coarsen_thresh=None):
        """
        mesh: instance of Grid (unstructured mesh)
        state_w: numpy array of primitive variables, shape (nCells, nVars)
        refine_thresh: error threshold above which cells are refined
        coarsen_thresh: error threshold below which cells are coarsened
        """
        self.mesh = mesh
        self.w = state_w
        self.refine_thresh = refine_thresh
        # default coarsen threshold is half of refine
        self.coarsen_thresh = coarsen_thresh if coarsen_thresh is not None else refine_thresh * 0.5
        self.error = np.zeros(mesh.nCells)
        self.refine_ids = []
        self.coarsen_ids = []

    def compute_error_indicator(self):
        """
        Simple gradient-based error: magnitude of density gradient per cell
        Requires mesh to have LSQ gradient reconstruction implemented
        """
        # ensure LSQ stencil is set up
        self.mesh.compute_compact_gradient_reconstruction_A_matrix()
        # reconstruct per-cell gradients
        for cell in self.mesh.cells:
            cid = cell.cid
            # w[cid,0] is density
            grad = cell.reconstruct_gradient(self.w[:,0])  # user must implement reconstruct_gradient
            self.error[cid] = np.linalg.norm(grad)
        return self.error

    def mark_cells(self):
        """
        Determine which cells to refine or coarsen
        """
        self.refine_ids = np.where(self.error > self.refine_thresh)[0]
        self.coarsen_ids = np.where(self.error < self.coarsen_thresh)[0]
        return self.refine_ids, self.coarsen_ids

    def refine(self):
        """
        Refine all marked cells in-place
        """
        for cid in self.refine_ids:
            cell = self.mesh.cells[cid]
            if len(cell.nodes) == 3:
                refine_triangle(self.mesh, cell)
            elif len(cell.nodes) == 4:
                refine_quad(self.mesh, cell)
        self._update_mesh_connectivity()

    def coarsen(self):
        """
        Coarsen all marked cells where possible (requires tracking refinement history)
        """
        # Placeholder: implement coarsening logic based on refinement tree
        self._update_mesh_connectivity()

    def _update_mesh_connectivity(self):
        """
        Rebuild mesh connectivity after refinement/coarsening
        """
        self.mesh.make_FaceCellMap()
        self.mesh.make_neighbors()
        self.mesh.buildCellToFaceIncidence()
        self.mesh.buildFaceToNodeIncidence()
        self.mesh.buildVertexToCellIncidence()


def refine_triangle(mesh, cell):
    """
    Split a triangular cell into 4 sub-triangles by connecting edge midpoints
    """
    # 1) Compute midpoints of each edge and add new nodes
    mid = {}
    nodes = cell.nodes
    for i in range(3):
        a = nodes[i]
        b = nodes[(i+1)%3]
        key = tuple(sorted((a.id, b.id)))
        if key not in mesh._edge_midpoints:
            pos = 0.5*(a.vector + b.vector)
            new_node = mesh.add_node(pos)
            mesh._edge_midpoints[key] = new_node
        mid[i] = mesh._edge_midpoints[key]
    # 2) Create 4 new triangles
    n0, n1, n2 = nodes
    m0, m1, m2 = mid[0], mid[1], mid[2]
    new_cells = [
        mesh.add_cell([n0, m0, m2]),
        mesh.add_cell([m0, n1, m1]),
        mesh.add_cell([m2, m1, n2]),
        mesh.add_cell([m0, m1, m2])
    ]
    # 3) Remove parent cell
    mesh.remove_cell(cell)


def refine_quad(mesh, cell):
    """
    Split a quadrilateral cell into 4 sub-quads by connecting edge midpoints and center
    """
    # Compute midpoints on each edge
    mid = {}
    nodes = cell.nodes
    for i in range(4):
        a = nodes[i]
        b = nodes[(i+1)%4]
        key = tuple(sorted((a.id, b.id)))
        if key not in mesh._edge_midpoints:
            pos = 0.5*(a.vector + b.vector)
            new_node = mesh.add_node(pos)
            mesh._edge_midpoints[key] = new_node
        mid[i] = mesh._edge_midpoints[key]
    # Center point
    center_pos = sum([n.vector for n in nodes]) / 4.0
    center = mesh.add_node(center_pos)
    # Create 4 quads
    new_cells = []
    for i in range(4):
        n0 = nodes[i]
        n1 = mid[i]
        n2 = center
        n3 = mid[(i-1)%4]
        new_cells.append(mesh.add_cell([n0, n1, n2, n3]))
    mesh.remove_cell(cell)
