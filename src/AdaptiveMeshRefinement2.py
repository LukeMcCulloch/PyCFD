#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:00:26 2025

@author: lukemcculloch
"""

import numpy as np
import System2D as s2d

# ----- Helpers for mesh mutation -----------------------------------------

def ensure_edge_midpoints(mesh):
    if not hasattr(mesh, '_edge_midpoints'):
        mesh._edge_midpoints = {}

def add_node(mesh, coords):
    """Add a new node to mesh, return its ID."""
    nid = mesh.nNodes
    mesh.nodes.append(s2d.Node(coords))
    mesh.nNodes += 1
    return nid

def add_cell(mesh, node_ids):
    """Add a new cell to mesh using node_ids list, return its ID."""
    cid = mesh.nCells
    mesh.cells.append(s2d.Cell(node_ids))
    mesh.nCells += 1
    return cid

def remove_cell(mesh, cid):
    """Mark cell cid as removed; actual renumbering happens on rebuild."""
    mesh.cells[cid] = None

def get_midpoint(mesh, n0, n1):
    """Get or create the midpoint node ID between nodes n0 and n1."""
    ensure_edge_midpoints(mesh)
    key = tuple(sorted((n0, n1)))
    if key in mesh._edge_midpoints:
        return mesh._edge_midpoints[key]
    p0 = mesh.nodes[n0].coords
    p1 = mesh.nodes[n1].coords
    mid = 0.5 * (p0 + p1)
    mid_id = add_node(mesh, mid)
    mesh._edge_midpoints[key] = mid_id
    return mid_id

# ----- Refinement routines ------------------------------------------------

def refine_triangle(mesh, cell):
    """Split a triangular cell into 4 smaller triangles."""
    n0, n1, n2 = cell.nodes
    a = get_midpoint(mesh, n0, n1)
    b = get_midpoint(mesh, n1, n2)
    c = get_midpoint(mesh, n2, n0)
    remove_cell(mesh, cell.cid)
    children = []
    for nodes in ([n0, a, c],
                  [a, n1, b],
                  [c, b, n2],
                  [a, b, c]):
        cid = add_cell(mesh, nodes)
        children.append(mesh.cells[cid])
    return children

def refine_quad(mesh, cell):
    """Split a quadrilateral cell into 4 smaller quads."""
    n0, n1, n2, n3 = cell.nodes
    e01 = get_midpoint(mesh, n0, n1)
    e12 = get_midpoint(mesh, n1, n2)
    e23 = get_midpoint(mesh, n2, n3)
    e30 = get_midpoint(mesh, n3, n0)
    center_coords = (mesh.nodes[n0].coords +
                     mesh.nodes[n1].coords +
                     mesh.nodes[n2].coords +
                     mesh.nodes[n3].coords) / 4.0
    center = add_node(mesh, center_coords)
    remove_cell(mesh, cell.cid)
    children = []
    for nodes in ([n0, e01, center, e30],
                  [e01, n1, e12, center],
                  [center, e12, n2, e23],
                  [e30, center, e23, n3]):
        cid = add_cell(mesh, nodes)
        children.append(mesh.cells[cid])
    return children

# ----- Gradient-based error indicator ------------------------------------

def cell_reconstruct_gradient(mesh, state):
    """
    Compute a least-squares gradient of `state` for each cell.
    `state` is an array of shape (nCells, nVars).
    Returns an array of shape (nCells, nVars, 2).
    """
    nCells, nVars = state.shape
    grads = np.zeros((nCells, nVars, 2))
    if not hasattr(mesh, 'node_to_cells'):
        mesh.build_node_to_cells()
    for cell in mesh.cells:
        if cell is None:
            continue
        cid = cell.cid
        x0 = cell.centroid
        A_rows, B_rows = [], []
        for nid in cell.nodes:
            for adj in mesh.node_to_cells[nid]:
                if adj == cid:
                    continue
                xj = mesh.cells[adj].centroid
                dx = xj - x0          # shape (2,)
                df = state[adj] - state[cid]  # shape (nVars,)
                A_rows.append(dx)
                B_rows.append(df)
        if len(A_rows) >= 2:
            A = np.array(A_rows)       # (m,2)
            B = np.vstack(B_rows)      # (m,nVars)
            sol, *_ = np.linalg.lstsq(A, B, rcond=None)
            grads[cid] = sol.T         # (nVars,2)
    return grads

# ----- Coarsening stub ----------------------------------------------------

def coarsen_group(mesh, parent_cid, child_ids):
    """
    Placeholder for coarsening: remove children and restore the original parent.
    """
    # TODO: implement the reverse of refine_triangle/refine_quad
    pass

# ----- AMR Controller -----------------------------------------------------

class AMR:
    def __init__(self, mesh, state, refine_thresh, coarsen_thresh=None):
        self.mesh = mesh
        self.state = state
        self.refine_thresh = refine_thresh
        self.coarsen_thresh = coarsen_thresh if coarsen_thresh is not None else refine_thresh * 0.5
        self.parent_to_children = {}
        self.child_to_parent = {}
        self.refine_ids = []
        self.coarsen_ids = []

    def compute_error_indicator(self):
        grads = cell_reconstruct_gradient(self.mesh, self.state)
        err = np.linalg.norm(grads.reshape(grads.shape[0], -1), axis=1)
        return err

    def mark_cells(self):
        err = self.compute_error_indicator()
        self.refine_ids = [i for i, e in enumerate(err) if e > self.refine_thresh]
        self.coarsen_ids = [i for i, e in enumerate(err)
                            if e < self.coarsen_thresh and i not in self.parent_to_children]
        return self.refine_ids, self.coarsen_ids

    def refine(self):
        ensure_edge_midpoints(self.mesh)
        for pid in list(self.refine_ids):
            parent = self.mesh.cells[pid]
            if parent is None:
                continue
            if len(parent.nodes) == 3:
                children = refine_triangle(self.mesh, parent)
            else:
                children = refine_quad(self.mesh, parent)
            child_ids = [c.cid for c in children]
            self.parent_to_children[pid] = child_ids
            for cid in child_ids:
                self.child_to_parent[cid] = pid
        # rebuild connectivity & geometry
        self.mesh.build_all()

    def coarsen(self):
        for pid, kids in list(self.parent_to_children.items()):
            if all(k in self.coarsen_ids for k in kids):
                coarsen_group(self.mesh, pid, kids)
                for k in kids:
                    del self.child_to_parent[k]
                del self.parent_to_children[pid]
        self.mesh.build_all()