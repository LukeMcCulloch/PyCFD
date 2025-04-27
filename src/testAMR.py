#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:12:09 2025

@author: lukemcculloch
"""

import Solvers as solve

import AdaptiveMeshRefinement  as AMR




if __name__ == '__main__':
    # gd = Grid(type_='quad',m=10,n=10,
    #           winding='ccw')
    
    #mesh = Grid(type_='tri',m=42,n=21,
    #          winding='ccw')
    
    #mesh = Grid(type_='quad',m=42,n=21,
    #          winding='ccw')
    #mesh = Grid(generated=True,type_='quad',m=42,n=21,
    #          winding='ccw')
    
    #cell = mesh.cellList[44]
    #face = cell.faces[0]
    
    #cell.plot_cell()
    
    vtkNames = {0:'vortex.vtk',
                1:'airfoil.vtk',
                2:'cylinder.vtk',
                3:'test.vtk',
                4:'shock_diffraction.vtk'}
    
    thisTest = 4
    whichTest = {0:solve.TestInviscidVortex,
                 1:solve.TestSteadyAirfoil,
                 2:solve.TestSteadyCylinder,
                 3:solve.TestTEgrid,
                 4:solve.TestShockDiffractiongrid}
    #test = TestInviscidVortex()
    #test = TestSteadyAirfoil()
    #test = TestSteadyCylinder()
    #test = TestTEgrid()
    #test = TestShockDiffractiongrid()
    test = whichTest[thisTest]()
    
    
    #if False:
    if True:
        
        #'''
        # AMR
        # choose thresholds (you may want to sweep these in tests)
        refine_thresh = 0.1
        coarsen_thresh = 0.05
        amr = AMR.AMR(test.grid, w, refine_thresh, coarsen_thresh)
        
        self = solve.Solvers(mesh = test.grid)
        
        #cc = self.cclsq[35]
        #cc.plot_lsq_reconstruction()
        
        
        #----------------------------
        # plot LSQ gradient stencils
        #show_LSQ_grad_area_plots(self)
        
        
        # cc = self.cclsq[57]
        # cc.plot_lsq_reconstruction()
        # cell = cc.cell
        # cell.plot_cell() #normals should be outward facing
        
        #'''
        
        #"""
        whichSolver = {0: 'vortex',
                        1: 'freestream',
                        2: 'freestream',
                        3: 'mms',
                        4:'shock-diffraction'}
        # whichSolver = {0: 'vortex',
        #                1: 'freestream',
        #                2: 'freestream',
        #                3: 'mms',
        #                4:'freestream'}
        #self.solver_boot(flowtype = 'mms') #TODO fixme compute_manufactured_sol_and_f_euler return vals
        #self.solver_boot(flowtype = 'freestream')
        #self.solver_boot(flowtype = 'vortex')
        #self.solver_boot(flowtype = 'shock-diffraction')
        
        self.solver_boot(flowtype = whichSolver[thisTest])
        
        #self.plot_flow_at_cell_centers(title = 'Initial Solution')
        
        self.write_solution_to_vtk('init_'+vtkNames[thisTest])
        
        solvertype = {0:'explicit_unsteady_solver',
                      1:'explicit_steady_solver',
                      2:'explicit_steady_solver',
                      3:'mms_solver',
                      4:'explicit_unsteady_solver_efficient_shockdiffraction'}
        # solvertype = {0:'explicit_unsteady_solver',
        #               1:'explicit_steady_solver',
        #               2:'explicit_steady_solver',
        #               3:'mms_solver',
        #               4:'explicit_unsteady_solver'}
        #'''
        self.print_nml_data()
        self.solver_solve( tfinal=0.7, dt=.01, 
                          solver_type = solvertype[thisTest])
        
        self.write_solution_to_vtk(vtkNames[thisTest])
        #'''
        ################################
        '''
        self.solver_solve( tfinal=0.2, dt=.01, 
                           solver_type = solvertype[1])
        print ('nfaces :',len(self.mesh.faceList))
        #self.write_solution_to_vtk('test.vtk')
        self.write_solution_to_vtk(vtkNames[thisTest])
        #'''
        ################################
        '''
        self.solver_solve( tfinal=0.2, dt=.01, 
                           solver_type = solvertype[2])
        #'''
        
        '''
        self.plot_solution( title='Final ')
        #'''
        
    
        # print('--------------------------------')
        # print('validate normals on boundaries')
        # for bound in self.mesh.bound:
        #     print(bound.bc_type)
        # for face in self.mesh.boundaryList:
        #     print(face.compute_normal(True))
    
        
        '''
        # if memory issues are encountered:
        del(self)
        del(mesh)
        
        canvas = plotmesh = PlotGrid(self.mesh)
        plotmesh.plot_boundary() #normals should be outward facing
        
        for bface in self.mesh.boundaryList:
            print(bface.parentcell.cid,bface.face_nrml_mag)
        
        plotmesh = PlotGrid(self.mesh)
        axTri = plotmesh.plot_cells()
        axTri = plotmesh.plot_centroids(axTri)
        axTri = plotmesh.plot_face_centers(axTri)
        axTri = plotmesh.plot_normals(axTri)
        
        plotmesh = PlotGrid(self.mesh)
        axRect = plotmesh.plot_cells()
        axRect = plotmesh.plot_centroids(axRect)
        axRect = plotmesh.plot_face_centers(axRect)
        axRect = plotmesh.plot_normals(axRect)
        
        
        #'''
        self.print_nml_data()