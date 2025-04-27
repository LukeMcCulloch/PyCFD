#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:12:09 2025

@author: lukemcculloch



whichTest = {0:TestInviscidVortex,
             1:TestSteadyAirfoil,
             2:TestSteadyCylinder,
             3:TestTEgrid,
             4:TestShockDiffractiongrid}
        
"""
import os # file and directory handling
import numpy as np

import System2D as s2d

import Parameters as Parameters
import Solvers as FiniteVolumeSolver
import Integrator as RKIntegrator

import AdaptiveMeshRefinement as AMR
#cell_reconstruct_gradient = AMR.cell_reconstruct_gradient

from ProblemTypesDefinitions import vtkNames, whichSolver, solvertype


whichTest = FiniteVolumeSolver.whichTest # TestShockDiffractiongrid


def stage1_static_solve():
    
    
    # 1) read parameters (including mesh filename, output folder, CFL, etc.)
    #params = Parameters('input.nml')

    thisTest = 4
    
    
    # # 2) load an initial mesh + solution from VTK
    test = whichTest[thisTest]()
    mesh = test.grid
    
    
    # # 3) set up solver + RK integrator
    # solver     = FiniteVolumeSolver(mesh, params)
    # integrator = RKIntegrator(mesh, solver, params)
    explicitSolver = FiniteVolumeSolver.Solvers(mesh = test.grid)
    explicitSolver.solver_boot(flowtype = whichSolver[thisTest])

    # # 4) one time‐step
    # dt = integrator.compute_dt(w)
    # w_new = integrator.advance(w, dt)
    explicitSolver.solver_solve(
        tfinal=1.0, 
        dt=.01,
        max_steps = 1,
        solver_type = solvertype[thisTest])
    
    
    explicitSolver.print_nml_data()
    
    
    w_new = explicitSolver.w
    

    # 5) write out for ParaView (this is done implicitly at steps in the solver but))
    # write_vtk(mesh, w_new, params.output_dir + '/stage1.vtu')
    #os.makedirs(path, exist_ok=True) 
    explicitSolver.write_solution_to_vtk('/stage1/amr_phase1_'+vtkNames[thisTest])
    print("Stage 1: wrote {params.output_dir}/stage1.vtu")
    
    # 5) set up AMR and mark once
    amr = AMR.AMR(
        mesh, 
        w_new, 
        explicitSolver.Parameters.refine_threshold, 
        explicitSolver.Parameters.coarsen_threshold)
    
    refine_ids, coarsen_ids = amr.mark_cells()
    print(f"AMR would refine {len(refine_ids)} cells, coarsen {len(coarsen_ids)} cells")



def test_amr_loop():
    #params = Parameters('input.nml')
    #mesh, w = load_vtk(params.mesh_vtk)
    
    # 1) read parameters (including mesh filename, output folder, CFL, etc.)
    thisTest = 4
    # # 2) load an initial mesh + solution from VTK
    test = whichTest[thisTest]()
    mesh = test.grid
    

    # # 3) set up solver + RK integrator
    # solver     = FiniteVolumeSolver(mesh, params)
    # integrator = RKIntegrator(mesh, solver, params)
    explicitSolver = FiniteVolumeSolver.Solvers(mesh = test.grid)
    explicitSolver.solver_boot(flowtype = whichSolver[thisTest])
    
    
    
    amr = AMR.AMR(
        mesh, 
        explicitSolver.w, 
        explicitSolver.Parameters.refine_threshold, 
        explicitSolver.Parameters.coarsen_threshold)
''' # tlm todo: start here!
    for step in range(params.nSteps):
        dt = integrator.compute_dt(w)
        w = integrator.advance(w, dt)

        # mark
        refine_ids, coarsen_ids = amr.mark_cells()
        print(f"[step {step}] refine={len(refine_ids)} coarsen={len(coarsen_ids)}")

        if refine_ids or coarsen_ids:
            # save old mesh + state
            old_mesh = copy.deepcopy(mesh)
            old_w    = w.copy()
            old_grads = cell_reconstruct_gradient(old_mesh, old_w)

            # refine & coarsen
            amr.refine()
            amr.coarsen()

            # reset mesh reference in solver/integrator
            mesh = amr.mesh
            solver.mesh     = mesh
            integrator.mesh = mesh

            # reproject old_w → new w
            nVars = old_w.shape[1]
            w_new = np.zeros((mesh.nCells, nVars))
            # parent→children
            for pid, child_ids in amr.parent_to_children.items():
                xp = old_mesh.cells[pid].centroid
                for cid in child_ids:
                    xc = mesh.cells[cid].centroid
                    w_new[cid] = old_w[pid] + old_grads[pid].dot(xc - xp)
            # untouched cells
            for cid in range(mesh.nCells):
                if cid not in amr.child_to_parent:
                    # if it’s an old cell not refined, just copy
                    w_new[cid] = old_w[cid]
            w = w_new

        # optional: write out every N steps
        if step % params.output_interval == 0:
            write_vtk(mesh, w, f"{params.output_dir}/step{step}.vtu")
#'''



if __name__ == '__main__':
    
    stage1_static_solve()
    
    test_amr_loop()




# def bigMessFromSolvers():
#     """
#     This will be removed once the AMR tests are built


#     """
#     # gd = Grid(type_='quad',m=10,n=10,
#     #           winding='ccw')
    
#     #mesh = Grid(type_='tri',m=42,n=21,
#     #          winding='ccw')
    
#     #mesh = Grid(type_='quad',m=42,n=21,
#     #          winding='ccw')
#     #mesh = Grid(generated=True,type_='quad',m=42,n=21,
#     #          winding='ccw')
    
#     #cell = mesh.cellList[44]
#     #face = cell.faces[0]
    
#     #cell.plot_cell()
    
#     vtkNames = {0:'vortex.vtk',
#                 1:'airfoil.vtk',
#                 2:'cylinder.vtk',
#                 3:'test.vtk',
#                 4:'shock_diffraction.vtk'}
    
#     thisTest = 4
#     whichTest = {0:FiniteVolumeSolver.TestInviscidVortex,
#                  1:FiniteVolumeSolver.TestSteadyAirfoil,
#                  2:FiniteVolumeSolver.TestSteadyCylinder,
#                  3:FiniteVolumeSolver.TestTEgrid,
#                  4:FiniteVolumeSolver.TestShockDiffractiongrid}
#     #test = TestInviscidVortex()
#     #test = TestSteadyAirfoil()
#     #test = TestSteadyCylinder()
#     #test = TestTEgrid()
#     #test = TestShockDiffractiongrid()
#     test = whichTest[thisTest]()
    
    
#     #if False:
#     if True:
        
#         #'''
#         # AMR
#         # choose thresholds (you may want to sweep these in tests)
#         refine_thresh = 0.1
#         coarsen_thresh = 0.05
#         #amr = AMR.AMR(test.grid, w, refine_thresh, coarsen_thresh)
        
#         self = FiniteVolumeSolver.Solvers(mesh = test.grid)
        
#         #cc = self.cclsq[35]
#         #cc.plot_lsq_reconstruction()
        
        
#         #----------------------------
#         # plot LSQ gradient stencils
#         #show_LSQ_grad_area_plots(self)
        
        
#         # cc = self.cclsq[57]
#         # cc.plot_lsq_reconstruction()
#         # cell = cc.cell
#         # cell.plot_cell() #normals should be outward facing
        
#         #'''
        
#         #"""
#         whichSolver = {0: 'vortex',
#                         1: 'freestream',
#                         2: 'freestream',
#                         3: 'mms',
#                         4:'shock-diffraction'}
#         # whichSolver = {0: 'vortex',
#         #                1: 'freestream',
#         #                2: 'freestream',
#         #                3: 'mms',
#         #                4:'freestream'}
#         #self.solver_boot(flowtype = 'mms') #TODO fixme compute_manufactured_sol_and_f_euler return vals
#         #self.solver_boot(flowtype = 'freestream')
#         #self.solver_boot(flowtype = 'vortex')
#         #self.solver_boot(flowtype = 'shock-diffraction')
        
#         self.solver_boot(flowtype = whichSolver[thisTest])
        
#         #self.plot_flow_at_cell_centers(title = 'Initial Solution')
        
#         self.write_solution_to_vtk('init_'+vtkNames[thisTest])
        
#         solvertype = {0:'explicit_unsteady_solver',
#                       1:'explicit_steady_solver',
#                       2:'explicit_steady_solver',
#                       3:'mms_solver',
#                       4:'explicit_unsteady_solver_efficient_shockdiffraction'}
#         # solvertype = {0:'explicit_unsteady_solver',
#         #               1:'explicit_steady_solver',
#         #               2:'explicit_steady_solver',
#         #               3:'mms_solver',
#         #               4:'explicit_unsteady_solver'}
#         #'''
#         self.print_nml_data()
#         self.solver_solve( tfinal=0.7, dt=.01, 
#                           solver_type = solvertype[thisTest])
        
#         self.write_solution_to_vtk(vtkNames[thisTest])
#         #'''
#         ################################
#         '''
#         self.solver_solve( tfinal=0.2, dt=.01, 
#                            solver_type = solvertype[1])
#         print ('nfaces :',len(self.mesh.faceList))
#         #self.write_solution_to_vtk('test.vtk')
#         self.write_solution_to_vtk(vtkNames[thisTest])
#         #'''
#         ################################
#         '''
#         self.solver_solve( tfinal=0.2, dt=.01, 
#                            solver_type = solvertype[2])
#         #'''
        
#         '''
#         self.plot_solution( title='Final ')
#         #'''
        
    
#         # print('--------------------------------')
#         # print('validate normals on boundaries')
#         # for bound in self.mesh.bound:
#         #     print(bound.bc_type)
#         # for face in self.mesh.boundaryList:
#         #     print(face.compute_normal(True))
    
        
#         '''
#         # if memory issues are encountered:
#         del(self)
#         del(mesh)
        
#         canvas = plotmesh = PlotGrid(self.mesh)
#         plotmesh.plot_boundary() #normals should be outward facing
        
#         for bface in self.mesh.boundaryList:
#             print(bface.parentcell.cid,bface.face_nrml_mag)
        
#         plotmesh = PlotGrid(self.mesh)
#         axTri = plotmesh.plot_cells()
#         axTri = plotmesh.plot_centroids(axTri)
#         axTri = plotmesh.plot_face_centers(axTri)
#         axTri = plotmesh.plot_normals(axTri)
        
#         plotmesh = PlotGrid(self.mesh)
#         axRect = plotmesh.plot_cells()
#         axRect = plotmesh.plot_centroids(axRect)
#         axRect = plotmesh.plot_face_centers(axRect)
#         axRect = plotmesh.plot_normals(axRect)
        
        
#         #'''
#         self.print_nml_data()
    
