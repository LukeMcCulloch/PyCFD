# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:35:53 2020

@author: Luke.McCulloch
"""
import weakref
import os
try:
    from memory_profiler import profile
    MEM_PROFILE = True
except:
    print( 'please install memory_profiler' )
    MEM_PROFILE = False
#
import numpy as np
import matplotlib.pyplot as plt

from Overload import Overload
from PlotGrids import PlotGrid

from Utilities import normalize, normalized, norm, dot, cross, \
    normalize2D, normalized2D, triangle_area, triangle_area_from_raw_data
    
from FileTools import GetLines, GetLineByLine

from DataHandler import DataHandler

class Node(Overload):
    def __init__(self, vector, nid, nConserved=3):
        self.x0 = vector[0]
        self.x1 = vector[1]
        self.nid = nid
        self.vector = vector
        self.parent_faces = [] #nodes are created before faces
        self.parent_cells = []
        # and before cells, so we can wait until face creation
        # to start filling this in.
        
        self.nConserved     = nConserved
        self.Q              = np.zeros((nConserved,1),float)
        
    def __repr__(self):
        '''initialization and value'''
        #print('{}'.format(self.vector))
        return str('node({}) = {}'.format(self.nid,self.vector))
        #return self.vector
    
    def __str__(self):
        '''print() value '''
        return str(self.vector)
        #return self.vector
    
    def compute_nodal_phi(self):
        """
        simple averaging over neighboring nodes
        """
        N = len(self.faces)
        for face in self.faces:
            self.Q += face.parentcell.Q
        self.Q /= N
        return

class Face(object):
    '''
    In 2D, a face is an edge
    '''
    def __init__(self, nodes, parentcell, fid, 
                 nConserved=3, isBoundary = False):
        self.nodes = nodes # Node data type
        #self._nodes = weakref.ref(nodes) if nodes else nodes
        #self.parentcell = parentcell
        self._parentcell = weakref.ref(parentcell) if parentcell else parentcell
        self.fid = fid
        self.adjacentface   = None
        self.isBoundary     = isBoundary
        #
        # neighbor vectors
        self.e_xi           = None
        self.e_eta          = None
        self.heta           = None
        self.hxi            = None
        #
        # conserved quantities
        self.nConserved     = nConserved
        self.Q              = np.zeros((nConserved,1),float)
        self.isBoundary = True
        #
        # Basic connectivity
        #
        for node in self.nodes:
            node.parent_faces.append(self)
        self.N = 2 #len(nodes)
        self.center = .5*(self.nodes[0] + self.nodes[1])
        #self.cell = parentcell #FIXME:  _parentcell? -- redundent anyway
        #
        # Basic geometry
        #
        self.area = np.linalg.norm(self.nodes[1]-self.nodes[0])
        self.normal_vector, self.face_nrml_mag = self.compute_normal(normalizeIt = True)
        normal_vector, face_nrml_mag = self.compute_normalfancy(normalizeIt = True)
        #if self.isBoundary:
        #    print('normal vec: ', self.normal_vector)
        #    print('fancy normal vec: ',normal_vector)
        
    @property
    def parentcell(self):
        if not self._parentcell:
            return self._parentcell
        _parentcell = self._parentcell()
        if _parentcell:
            return _parentcell
        else:
            raise LookupError("Parent cell was destroyed")
            
            
    #    @property
    #    def nodes(self):
    #        if not self._nodes:
    #            return self._nodes
    #        _nodes = self._nodes()
    #        if _nodes:
    #            return _nodes
    #        else:
    #            raise LookupError("node was destroyed")
            
            
            
    #def __del__(self):
    #    print("delete", self.fid)
        
        
        
    def compute_normalfancy(self, normalizeIt=True):
        """
        cross vector with 0 in x2 dir
        with vector with 1 in the x2 dir
         
        this is of course (x1,-x0, 0)
         
        normals point in? no
        normals point out.
         
        """
        #vec = self.nodes[1] - self.nodes[0]
        
        dumvec1 = np.zeros((3),float)
        dumvec2 = np.zeros_like((dumvec1))
        dumvec1[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[-1] = 1.
        
        n3 = normalize(cross(dumvec1,dumvec2) )
        
        #if self.isBoundary:
        #    print ('bface normal', n3[:-1])
        return n3[:-1], np.linalg.norm(cross(dumvec1,dumvec2) )
        
                  
    # reversed               
    # def compute_normal(self, normalizeIt=True):
    #     """ 2D specific face normals
    #     normalized(x1,-x0)
    #     """
    #     vec = self.nodes[1] - self.nodes[0]
    #     vec = np.asarray([-vec[1],vec[0]])
        
    #     if normalizeIt:
    #         vec, nmag =  normalize2D(vec[:],return_mag=True)
    #         #vec =  normalize2D(vec,return_mag=False)
            
    #         #if self.isBoundary:
    #         #    print ('bface normal', vec)
    #         return vec, nmag
    #     else:
    #         return vec  
    
    
    # standard
    def compute_normal(self, normalizeIt=True):
        """ 2D specific face normals
        normalized(x1,-x0)
        """
        vec = self.nodes[1] - self.nodes[0]
        vec = np.asarray([vec[1],-vec[0]])
        
        if normalizeIt:
            vec, nmag =  normalize2D(vec[:],return_mag=True)
            #vec =  normalize2D(vec,return_mag=False)
            
            #if self.isBoundary:
            #    print ('bface normal', vec)
            return vec, nmag
        else:
            return vec
    
    
    def compute_e_xi(self):
        """
        centroid-centroid unit normal vector
        across adjacent faces
        
        in general this is not aligned with 
        the face normal, 
        though for orthognal meshes it will be.
        """
        Xp = self.parentcell.centroid
        Xa =  self.adjacentface.parentcell.centroid
        Xidiff = Xa-Xp
        magXi = 1./np.linalg.norm(Xidiff)
        self.hxi, self.e_xi = magXi, Xidiff*magXi
        return
    
    
    def compute_e_eta(self):
        """
        transverse face unit normal vector
        """
        A = self.nodes[0]
        B = self.nodes[1]
        Etadiff = B-A
        magEtadiff = 1./np.linalg.norm(Etadiff)
        self.heta, self.e_eta =  magEtadiff, Etadiff*magEtadiff
        return
    
    
    def compute_direct_diffusion_constant(self, Gamma=1.):
        """
        That part of the direct diffusion 
        which is constant 
        while geometry is constant
        """
        scale = self.area/(self.hxi * np.dot(self.normal_vector,self.e_xi))
        return Gamma * np.dot(self.normal_vector,self.normal_vector) *scale
         
    
    
    def compute_cross_diffusion_constant(self, Gamma=1.):
        """
        That part of the cross diffusion 
        which is constant 
        while geometry is constant
        (note that this is zero for orthogonal meshes)
        """
        scale = self.area/(self.heta * np.dot(self.normal_vector,self.e_xi))
        return -Gamma * np.dot(self.normal_vector,self.e_eta) *scale
    
    
    def compute_Dphi_Dxi(self):
        Qa = self.parentcell.Q
        Qp =  self.adjacentface.parentcell.Q
        return Qa-Qp #(Qa-Qp)/self.hxi
        
    
    def compute_Dphi_Deta(self):
        """
        assumes these have been 
        approximated by, e.g. averaging from 
        cell centers
        """
        Qa = self.nodes[0].Q
        Qb = self.nodes[1].Q
        return Qb-Qa #(Qb-Qa)/self.hxi
    
    def plot_face_normal(self, canvas = None,
                         alpha=.4):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
            
        normal_vector = self.area * self.normal_vector/np.linalg.norm(self.normal_vector)
        
        norm0 = self.normal_vector + self.center
        norm0 = normal_vector + self.center
                
        ax.plot([ norm0[0],self.center[0] ],
                [ norm0[1],self.center[1] ],
                color='black',
                alpha = alpha)
        ax.scatter(self.center[0],self.center[1], marker='x',
                color='black',
                alpha = alpha)
        ax.scatter(norm0[0],norm0[1],marker = '.',
                color='black',
                alpha = alpha)
                
        return ax
    
        
class BGrid(object):
    '''
    ----------------------------------------------------------
     Data type for boundary quantities (for both node/cell-centered schemes)
     Note: Each boundary segment has the following data.
    ----------------------------------------------------------
    '''
    def __init__(self, bc_type, nbnodes, bnode, nbfaces):
        self.bc_type = bc_type #type of boundary nodes for this segment
        self.nbnodes = nbnodes #number of boundary nodes for this segment
        self.bnode = bnode     #list of boundary nodes for this segment
        
        #to be constructed from the code
        self.nbfaces = nbfaces # number of boundary faces
        self.bfnx    = None #x-component of the face outward normal
        self.bfny    = None #y-component of the face outward normal
        self.bfn     = None #magnitude of the face normal vector
        self.bnx     = None #x-component of the outward normal
        self.bny     = None #y-component of the outward normal
        self.bn      = None #magnitude of the normal vector
        self.belm    = None #list of elm adjacent to boundary face
        self.kth_nghbr_of_1 = None
        self.kth_nghbr_of_2 = None
        
        
    
        
        
class Cell(object):
    """
    The Cij'th cell, tris and quads only
    
    ccw-winding: normals point out
    
    cw-winding: normals point in
    """
    def __init__(self, nodes, cid, nface,
                 nConserved=3, facelist=None): #, FaceCellMap):
        self.nodes = nodes #Node data type
        self.N = len(self.nodes)
        self.num_faces = self.N
        self.num_nodes = self.N
        self.volume = 0.
        self.set_centroid()
        self.set_volume()
        self.F = np.zeros((self.num_faces),float)
        self.G = np.zeros((self.num_faces),float)
        self.faces = []
        self.nghbr = [] #list of neighbor cells
        self.cid = cid
        self.set_face_vectors(nface, facelist)
        self.Q = np.zeros((nConserved,1),float)
        for node in self.nodes:
            node.parent_cells.append(self)
        
    def set_centroid(self):
        """The 'center' of this face
        -center as average point location
        
        -circumcentric?
        -barycentric?
        """
        scale = 1./float(self.N)
        self.centroid = scale * sum([el.vector for el in self.nodes]) 
        return self.centroid
    
    def set_volume(self):
        """
        for tris and convex, 2D polyhedra
        """
        if self.N == 3:
            self.volume = triangle_area(*self.nodes)
        elif (self.N == 4):
            v1 = Node(self.nodes[0].vector,-1)
            v2 = Node(self.nodes[1].vector,-1)
            v3 = Node(self.nodes[2].vector,-1)
            v4 = Node(self.nodes[3].vector,-1)
            
            self.volume = 0.0
            
            self.volume += triangle_area(v1,v2,v4)
            self.volume += triangle_area(v2,v3,v4)
        else:
            assert(False),"Error: cell is neither quad nor tri"
            # vol = 0.
            # for i in range(self.N-3+1):
            #     nlist1 = self.nodes[i:i+3]
            #     vol += triangle_area(*nlist1)
            # self.volume = -vol
        return 
        
    def set_face_vectors(self, nface, facelist):   #, n, FaceCellMap):
        for i in range(self.N):
            face = Face( [self.nodes[i],
                                      self.nodes[(i+1)%self.N] 
                                      ],
                               parentcell=self,
                               fid = nface)
            self.faces.append( face )
            facelist.append(   face )
            nface += 1
        return #n, FaceCellMap
    
    def get(self, i):
        assert(i<=self.N),'error, i>N'
        return self.N%i
    
    def face(self, i):
        return 
    
    def print_vertices(self):
        for vert in self.nodes:
            print(vert.vector )
        return
    
    
    def print_faces(self, normals=False):
        for face in self.faces:
            print('face' )
            for vert in face.nodes:
                print( vert.vector )
            if normals:
                print('face normal' )
                print( face.compute_normal() )
        return
    
    def compute_compact_gradient_reconstruction_A_matrix(self):
        """
        compact stencil
        """
        self.A = np.zeros((self.N,2),float)
        for i,face in enumerate(self.faces):
            #cell = face.adjacentface.parentcell
            self.A[i] = face.e_xi * face.hxi
        
        self.AAinv = np.linalg.inv(
                                    np.dot(self.A.T,self.A) 
                                  )
        qi, ri = np.linalg.qr(self.A)
        self.rinvqt = np.dot(np.linalg.inv(ri), qi.T)
        return
    
    
    
#    def compute_extended_gradient_reconstruction(self):
#        """
#        compact stencil
#        """
#        self.A = np.zeros((self.N,2),float)
#        for i,face in enumerate(self.faces):
#            #cell = face.adjacentface.parentcell
#            self.A[i] = face.e_xi * face.hxi
#        
#        qi, ri = np.linalg.qr(cell.A)
#        self.rinvqt = np.dot(np.linalg.inv(ri), qi.T)
#        return
    
    def plot_centroid(self, canvas = None,
                       alpha=.4):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        
        cell = self
        ax.plot(cell.centroid[0],
                cell.centroid[1],
                color='black',
                marker='o',
                alpha = alpha,)
        name = str(cell.cid)
        plt.annotate(name,
                     cell.centroid)
        return ax
    
    
    def plot_face_centers(self, canvas = None,
                       alpha=.1):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        
        cell  =self
        for face in cell.faces:
            
            
            ax.plot(face.center[0],
                    face.center[1],
                    color='yellow',
                    marker='o',
                    alpha = alpha)
            
            ax.plot(face.center[0],
                    face.center[1],
                    color='yellow',
                    marker='o',
                    alpha = alpha)
        
        return ax
    
    def plot_normals(self, canvas = None,
                       alpha=.4,
                       scale = .25):
        if canvas is None:
            fig, ax = plt.subplots()
        else:
            ax = canvas
        
        cell = self
        for face in cell.faces:
            # print('new face'
            # print('\n Normal vector'
            # print(face.normal_vector
            # print('\n center'
            # print(face.center
            
            norm0 = .5*face.normal_vector*face.area**2 + face.center
            #norm0 = norm0*face.area
            
            
            fnorm = face.normal_vector
            norm = 2.*np.linalg.norm(face.normal_vector)*face.area
            
            norm = np.linalg.norm(face.normal_vector)
            
            #scalearrow = np.linalg.norm(norm0)
            # plt.arrow(face.center[0],
            #           face.center[1],
            #           norm0[0]-face.center[0] ,
            #           norm0[1]-face.center[1] )
            
            
            plt.arrow(x=face.center[0],
                      y=face.center[1],
                      dx=scale*fnorm[0]/norm ,
                      dy=scale*fnorm[1]/norm )
        return ax
        
    def plot_cell(self, canvas = None,
                  fig=None,alpha=.1,
                  fillcolor=None):
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
            
        cell = self
        
    
        for face in cell.faces:
            n0 = face.nodes[0]
            n1 = face.nodes[1]
            
            ax.plot(n0.x0,n0.x1,
                    color='red',
                    marker='o',
                    alpha = alpha)
            ax.plot(n1.x0,n1.x1,
                    color='red',
                    marker='o',
                    alpha = alpha)
            
            x = [n0.x0,n1.x0]
            y = [n0.x1,n1.x1]
            ax.plot(x, y,
                    color='black',
                    alpha = alpha)
            
        if fillcolor is not None:
            xf = [ node.vector[0] for node in cell.nodes]
            yf = [ node.vector[1] for node in cell.nodes]
            xf.append(cell.nodes[0].vector[0])
            yf.append(cell.nodes[0].vector[1])
            ph = plt.fill(xf,yf, fillcolor, lw=2)
        else:  
            ax = self.plot_normals(canvas = ax)
            ax = self.plot_centroid(canvas = ax)
            ax = self.plot_face_centers(canvas = ax)
        return ax


class Grid(object):
    """
    cw-winding: normals point in
    
    ccw-winding: normals point out
    """
    def __init__(self, 
                 generated=True,
                 dhandle = False,
                 mesh=None, 
                 xb = -20.,
                 yb = -10.,
                 xe = 20.,
                 ye = 10.,
                 m=10,n=10,
                 type_='quad', 
                 winding='ccw'):
        
        self.m = m
        self.n = n
        self.xb = xb
        self.xe = xe
        self.yb = yb
        self.ye = ye
        
        self.winding = winding
        
        self.generated = generated
        self.gridtype = {'quad':0,
                         'tri':1}
        self.dim = 2    #2D grid
        self.nCells = 0
        self.nFaces = 0
        
        self.nodeList = []
        self.cellList = []
        self.faceList = []
        self.boundaryList = []
        
        self.FToV = None # face to vertex (node) connectivity matrix
        self.EToV = None # cell (element) to vertex
        self.EToF = None # cell to face
        self.EToE = None # cell to cell
        self.VToE = None # vertex to cell incidence
        self.Cellindex2Nodesindex = {} # cell index to nodes indicies map
        
        self.nodes = []
        self.cells = []
        self.tria = []
        self.quad = []
        self.FaceCellMap = {}
        
        # check mesh
        self.heffv = None   # Effecrtive spacing based on # of nodes
        self.heffc = None   # Effecrtive spacing based on # of cells
        
        if self.generated:
            self.nNodes = m*n
            
            self.xb = xb
            self.xe = xe
            self.yb = yb
            self.ye = ye
            
            
            self.type = type_
            if mesh is None:
                mesh = np.zeros((2,m,n),float) # C-ordering.  last index is most rapid
                self.dim = 2
            else:
                shp = np.shape(mesh)
                self.dim = shp[0]
                self.m = shp[1]
                self.n = shp[2]
                
            #mms = np.linspace(0.,1.,m)
            #nms = np.linspace(0.,1.,n)
            mms = np.linspace(self.xb,self.xe,m)
            nms = np.linspace(self.yb,self.ye,n)
            self.mesh = mesh
            #
            """
            Nodes are defined first
            """
            nid = 0
            for i in range(self.m):
                self.nodes.append([])
                for j in  range(self.n):
                    self.mesh[0,i,j] = mms[i]
                    self.mesh[1,i,j] = nms[j]
                    node = Node(self.mesh[:,i,j], nid)
                    nid += 1
                    self.nodes[i].append(node) #to become 2D array (nicer for building the grid)
                    self.nodeList.append(node) #will stay as list
                    
            self.nodes = np.asarray(self.nodes)
            self.nodes_array = np.asarray(self.nodeList) #is this at all faster?
        
        else: #read text files...
            self.generated = generated
            self.elm = []
            handle = GetLineByLine(directory = dhandle.path_to_inputs_folder,
                                       filename = dhandle.filename_grid)
            
            handleBC = GetLineByLine(directory = dhandle.path_to_inputs_folder,
                                       filename = dhandle.filename_bc)
            print('\n\nReading the grid file....{}'.format(
                  dhandle.filename_grid))
            nnodes, ntria, nquad = (handle.readline()).split()
            nnodes = int(nnodes)
            ntria = int(ntria)
            nquad = int(nquad)
            print('nnodes {}, ntria {}, nquad {}'.format(
                nnodes, ntria, nquad))
            self.mesh = np.zeros((2,m,n),float) 
            
            nid = 0
            for i in range(nnodes):
                node = (handle.readline()).split()
                node = [float(nd) for nd in node]
                node = Node(np.asarray(node),nid)
                self.nodes.append(node)
                #self.nodes[i].append(node) #to become 2D array (nicer for building the grid)
                self.nodeList.append(node) #will stay as list
                nid += 1
            self.nodes = np.asarray(self.nodes)
            self.nodes_array = np.asarray(self.nodeList) 
            self.nNodes = nnodes
            self.ntria = ntria
            self.nquad = nquad
            
            
            
            #cid = 0
            #if ntria>0:
            for i in range(ntria):
                elm = (handle.readline()).split()
                
                elm = [int(nd)-1 for nd in elm]#mesh indicies start from 1 so convert to c indexing
                self.elm.append(elm)
                #print('elm = ',elm)
                nodesOfImport = [self.nodes[elm[0]],
                             self.nodes[elm[1]],
                             self.nodes[elm[2]]
                            ]
                #print('nodes of import = ',nodesOfImport)
                #print('node.vector  = ',nodesOfImport[0].vector)
                #for el in nodesOfImport:
                #    print('type(el.vector) = ',type(el.vector))
                #    print('el.vector = ',el.vector)
                centroid = sum([el.vector for el in nodesOfImport]) 
                #print('centroid = ',centroid)
                cell = Cell(
                            [self.nodes[elm[0]],
                             self.nodes[elm[1]],
                             self.nodes[elm[2]]
                            ],
                            cid = self.nCells,
                            nface = self.nFaces,
                            facelist = self.faceList
                            )
                self.Cellindex2Nodesindex[self.nCells] = elm
                
                self.cells.append(cell)
                self.tria.append(cell)
                self.cellList.append(self.cells[-1])
                self.nCells +=1
                self.nFaces += 3
                
    
            self.elm = np.asarray(self.elm)
            
            for i in range(nquad):
                elm = (handle.readline()).split()
                
                elm = [int(nd)-1 for nd in elm] #mesh indicies start from 1 so convert to c indexing
                
                
                cell = Cell(
                            [self.nodes[elm[0]],
                             self.nodes[elm[1]],
                             self.nodes[elm[2]],
                             self.nodes[elm[3]]
                            ],
                            cid = self.nCells,
                            nface = self.nFaces,
                            facelist = self.faceList
                            )
                
                self.cells.append(cell)
                self.quad.append(cell)
                self.cellList.append(self.cells[-1])
                self.nFaces += 4
                self.nCells +=1
            self.elm = np.asarray(self.elm)
            
            self.cells = np.asarray(self.cells)
            self.tria = np.asarray(self.tria)
            self.quad = np.asarray(self.quad)
            
            print('\n Total Numbers')
            print('          nodes = {}'.format(nnodes))
            print('      triangles = {}'.format(self.ntria))
            print('          quads = {}'.format(self.nquad))
            
            #read boundary data
            nbound = int((handle.readline()).split()[0]) #number of boundaries each with possible different conditions
            self.bound = []
            self.boundcount = []
            
            for i in range(nbound):#loop over the  different boundaries
                nbn = (handle.readline()).split()
                if len(nbn)>0: 
                    nbn = nbn[0]
                    self.boundcount.append(int(nbn))
                
            handle.readline()#hard coded line break is a smell!
            #nface = 0
            for i in range(nbound):#loop over the  different boundaries
                print (i)
                self.bound.append(BGrid('unknownType', 
                                        nbnodes=self.boundcount[i], 
                                        bnode = [],
                                        nbfaces = self.boundcount[i]-1))
                for j in range(self.boundcount[i]): #read in this many nodes on this boundary
                    thing = (handle.readline()).split()
                    if len(thing)>0:
                        thing = int(thing[0])-1
                        self.bound[i].bnode.append(thing)
                # bface = Face([self.nodes[i],
                #                  self.nodes[(i+1)%self.N] 
                #                 ],
                #                 parentcell=self,
                #                 fid = nface
                #                 )
                #     nface += 1
            
            self.bound = np.asarray(self.bound)
            handle.close() #done with  grid file read
            
            # print('\n Boundary nodes:')
            # print('    segments = {}'.format(nbound))
            # for i in range(nbound):
            #     print(' boundary, {},   bnodes = {}'.format(i,self.bound[i].bnode))
            #     print('                bfaces = {}'.format(self.bound[i].nbnodes-1))
                
            print('\n Boundary nodes:')
            print('    segments = {}'.format(nbound))
            for i in range(nbound):
                print(' boundary, {},   bnodes = {}'.format(i,self.bound[i].nbnodes))
                print('                bfaces = {}'.format(self.bound[i].nbfaces))
                
            '''
            read Boundary Conditions
            '''
            handleBC.readline()#hard coded line break is a smell!
            print('\n\nReading the boundary condition file.... {}\n'.format(dhandle.filename_bc))
            for i in range(nbound):
                thing = (handleBC.readline()).split()
                if len(thing)>0:
                    self.bound[i].bc_type = thing[1]
                    print(' boundary, {},   bc_type = {}'.format(i,self.bound[i].bc_type))
                
            handleBC.close() #done with  grid file read
            self.nbound = len(self.bound) #todo: use property to make this more official (probably needed lots of places...()
            # now cells and faces:
            self.FaceCellMap = {}
            # maps
            self.make_FaceCellMap()
            
            self.make_neighbors()
            
            # build incidence tables
            self.buildCellToFaceIncidence() # self.EToF
            self.buildFaceToNodeIncidence() # self.FToV
            self.buildVertexToCellIncidence() # self.VToC
            
        if self.generated:
            # now cells and faces:
            self.FaceCellMap = {}
            self.make_cells(winding = winding)
            # maps
            self.make_FaceCellMap()
            
            self.make_neighbors()
            
            # build incidence tables
            self.buildCellToFaceIncidence() # self.EToF
            self.buildFaceToNodeIncidence() # self.FToV
            self.buildVertexToCellIncidence() # self.VToC
            
        #----------------------------------------------
        # check the mesh:
        #----------------------------------------------
        self.compute_mesh_statistics()
        print("\n >>> Verifying the ccfv grid data...\n")
        self.check_centroids()
        self.check_volume()
        print( " --- Check the sum of all boundary normals (must be zero):")
        self.check_boundary_normals()
        self.check_normals()
        return
    
    def make_cells(self, winding ='cw'):
        """
        Nodes are defined first
        so, 
        Coincident Cells share nodes
        """
        if self.type == 'rect':
            self.make_rect_cells(winding = winding)
        else:
            self.make_tri_cells(winding = winding)
        return
    
    
    def make_rect_cells(self, winding='ccw'):
        #self.cells = []
        
        if winding=='cw':
            for i in range(self.m-1):
                #self.cells.append([])
                for j in  range(self.n-1):
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j  ],
                                              self.nodes[i  ,j+1],
                                              self.nodes[i+1,j+1],
                                              self.nodes[i+1,j  ] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nFaces += 4
                    self.nCells +=1
                
        else: #(winding is ccw)
            for i in range(self.m-1):
                #self.cells.append([])
                for j in  range(self.n-1):
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j  ],
                                              self.nodes[i+1,j  ],
                                              self.nodes[i+1,j+1],
                                              self.nodes[i  ,j+1] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nFaces += 4
                    self.nCells +=1
                
            
        self.cells = np.asarray(self.cells)
        return
    
    
    def make_tri_cells(self, winding='ccw'):
        #self.cells = []
        
        if winding=='cw':
            for i in range(self.m-1):
                #self.cells.append([])
                for j in  range(self.n-1):
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j  ],
                                              self.nodes[i  ,j+1],
                                              self.nodes[i+1,j  ] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nCells +=1
                    self.nFaces += 3
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j+1],
                                              self.nodes[i+1,j+1],
                                              self.nodes[i+1,j  ] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nCells += 1
                    self.nFaces += 3
                    
        else: #(winding is ccw)
            for i in range(self.m-1):
                #self.cells.append([])
                for j in  range(self.n-1):
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j  ],
                                              self.nodes[i+1,j  ],
                                              self.nodes[i  ,j+1] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nCells +=1
                    self.nFaces += 3
                    
                    self.cells.append(
                                        Cell([self.nodes[i  ,j+1],
                                              self.nodes[i+1,j  ],
                                              self.nodes[i+1,j+1] ],
                                             cid=self.nCells,
                                             nface = self.nFaces,
                                             facelist = self.faceList)
                                        )
                    self.cellList.append(self.cells[-1])
                    self.nCells += 1
                    self.nFaces += 3
            
        self.cells = np.asarray(self.cells)
        return
    
    def make_FaceCellMap(self):
        """
        iterate over all faces, 
        and save the map from face to cell it belongs to
        
        also, find adjacent faces and map them
        
        also, find neighbor cells and map them
            
        see:
            https://scicomp.stackexchange.com/questions/24981/
            getting-adjacent-cells-map-for-an-unstructured-polyhedral-mesh
        """
        for cell in self.cellList:
            for face in cell.faces:
                self.FaceCellMap[face.fid] = cell #old fashioned way
                #self.FaceCellMap[face] = cell #or hash the object directly
                #node0 = face.nodes[0]
                #node1 = face.nodes[1]
                #face_nodes = set(face.nodes)
                #lenfn = len(face_nodes)
                
                #face_set0 = set(face.nodes[0].parent_faces)
                face_set1 = set(face.nodes[1].parent_faces)
                ck_face_set = face_set1# face_set1 - (face_set0 & face_set1)
                
                for el in ck_face_set:
                    if el.adjacentface is None:
                        if  el.nodes[1] is face.nodes[0] :
                            #
                            face.adjacentface = el
                            el.adjacentface = face
                            el.isBoundary = False
                            face.isBoundary = False
                            #
                            # go ahead and set local vectors 
                            # between adjacent cells
                            face.compute_e_xi()
                            face.compute_e_eta()
                            el.compute_e_xi()
                            el.compute_e_eta()
                            break
        return
    
    def make_neighbors(self):
        """
        loop over all cells, all faces of each cell
        if a face has an adjacent face, (not None)
        i.e. if there is a neighbor,
        then add that to the cells nghbr list
        """
        for cell in self.cellList:
            for face in cell.faces:
                #https://stackoverflow.com/questions/2710940/python-if-x-is-not-none-or-if-not-x-is-none
                if face.adjacentface is not None:
                    cell.nghbr.append(face.adjacentface.parentcell)
                else:
                    # you've got the information to define the boundary 
                    # so do it here!:
                    face.isBoundary = True
                    self.boundaryList.append(face)
        self.nBoundaries = len(self.boundaryList)
        #self.bound = np.asarray(self.boundaryList)
        return
    
    
    def buildFaceToNodeIncidence(self):
        """
        Note the similarity to 
        Exterior differential forms
        """
        self.FToV = np.zeros((self.nFaces,self.nNodes),int)
        
        for edge in self.faceList:
            vh1 = edge.nodes[0]
            vh2 = edge.nodes[1]
            self.FToV[edge.fid,vh1.nid] = -1
            self.FToV[edge.fid,vh2.nid] = 1
        return

    
    def buildCellToFaceIncidence(self):
        """
        Note the similarity to 
        Exterior differential forms
        """
        self.EToF = np.zeros((self.nCells,self.nFaces),int)
        
        for cell in self.cellList:
            c = cell.cid
            for edge in cell.faces:
                e = edge.fid
                self.EToF[c,e] = 1
        return
    
    
    
    def buildVertexToCellIncidence(self):
        """
        Note the similarity to 
        Exterior differential forms
        
        e.g. 
        self.nodes_array[10].parent_cells[0] is self.cells[0]
        >>True
        self.nodes_array[10].parent_cells[1] is self.cells[1]
        >> True
        self.nodes_array[10].parent_cells[2] is self.cellList[18]
        >> True
        
        """
        self.VToE = np.zeros((self.nNodes,self.nCells),int)
        
        for cell in self.cellList:
            c = cell.cid
            for node in cell.nodes:
                n = node.nid
                self.VToE[n,c] = 1
        return
    
    
    #-------------------------------------------------------------------------#
    # Mesh checks
    #-------------------------------------------------------------------------#
    def check_volume(self, tol=1e-8):
        vol_domain = self.sum_volume_green_gauss()
        print("  Total volume of the domain = ", vol_domain)
        vol_domain_cells = self.sum_volume_cell_sum()
        print("     Sum of the cell volumes = ", vol_domain_cells)
        print("                  Difference = ", abs(vol_domain - vol_domain_cells))
        print()
        if (abs(vol_domain-vol_domain_cells)>tol):
            print(" Volume difference is larger than round-off error (actually tolerance)... Something is wrong. Stop." )
            #assert(False),"ERROR: Volumes are not consistent"
        return
    
    def sum_volume_green_gauss(self):
        '''
        assume ___ faceing normal
        '''
        vol = 0.
        for bound in self.boundaryList:
            mid = bound.center
            vol += np.dot( mid,bound.normal_vector )*bound.face_nrml_mag
        return 0.5*vol
    
    # def sum_volume_cell_sum(self):
    #     vol = 0.
    #     for cell in self.cellList:
    #         vol += cell.volume
    #     return -vol
    
    def sum_volume_cell_sum(self):
        vol = 0.
        for cell in self.cellList:
            vol += cell.volume
        return vol
    
    
    
    #--------------------------------------------------------
    # Boundary normal sum check
    #--------------------------------------------------------
    def check_boundary_normals(self, tol=1.e-10):
        sum_bnormal = np.zeros(2,float)
        for bound in self.boundaryList:
            sum_bnormal[0] += bound.normal_vector[0] * bound.face_nrml_mag
            sum_bnormal[1] += bound.normal_vector[1] * bound.face_nrml_mag
        
        print("     Sum of boundary face normal (nx) = ", sum_bnormal[0])
        print("     Sum of boundary face normal (ny) = ", sum_bnormal[1])
        print()
        if (abs(sum_bnormal[0])>tol or abs(sum_bnormal[1])>tol):
            print(" Boundary face vector sum is larger than machine zero....." )
            print(" Something is wrong. Stop.")
            assert(False),"ERROR: Sum of normals around the boundaries is not zero"
        return
    
    
    #--------------------------------------------------------
    # Face normal sum check for each cell
    #--------------------------------------------------------
    def check_normals(self, tol=1e-10):
        sum_face_normal_method1 = np.zeros((self.nCells,2),float)
        sum_face_normal = np.zeros((self.nCells,3),float)
        #----------------------------------------------------
        # Accumulate face normals at cells by looping over interior faces.
        #'''
        for i, cell in enumerate(self.cells):
            for face in cell.faces:
                sum_face_normal_method1[i,0] += face.normal_vector[0] * face.face_nrml_mag
                sum_face_normal_method1[i,1] += face.normal_vector[1] * face.face_nrml_mag
        #'''
        
        #'''
        for face in self.faceList:
            if face.isBoundary:
                pass
                #cid1 = face.parentcell.cid
                #sum_face_normal[cid1,0] += face.normal_vector[0] * face.face_nrml_mag
                #sum_face_normal[cid1,1] += face.normal_vector[1] * face.face_nrml_mag
            else:
                cid1 = face.parentcell.cid
                sum_face_normal[cid1,0] += face.normal_vector[0] * face.face_nrml_mag
                sum_face_normal[cid1,1] += face.normal_vector[1] * face.face_nrml_mag
                sum_face_normal[cid1,2] = 0.0 # not a boundary
            
                #face = face.adjacentface
                #cid2 = face.parentcell.cid
                #sum_face_normal[cid2,0] += face.normal_vector[0] * face.face_nrml_mag
                #sum_face_normal[cid2,1] += face.normal_vector[1] * face.face_nrml_mag
                #sum_face_normal[cid2,2] = 0.0 #not a boundary
        print("\nFaces are currently counted twice - once for each cell,")
        print(" and nodes are aranged to match depending on which cell this version of the face belongs to\n")
        #'''
        #----------------------------------------------------
        # Add boundary face normal contributions to cells by looping over boundary faces.
        
        # actually I think these are included above?
        #'''
        for bound in self.boundaryList:
            cid = bound.parentcell.cid
            sum_face_normal[cid,0] += bound.normal_vector[0] * bound.face_nrml_mag
            sum_face_normal[cid,1] += bound.normal_vector[1] * bound.face_nrml_mag
            sum_face_normal[cid,2] = 1.0 # a boundary
            
        #'''
        
        self.sum_face_normal = sum_face_normal
        print("   Display the maximum over all cells (must be zero):")
        print("   (the sum of face normals over each cell)")
        print()
        print(" method1 Max of |sum_faces face normal (nx)| = ", np.max(abs(sum_face_normal_method1[:,0])))
        print(" method1 Max of |sum_faces face normal (ny)| = ", np.max(abs(sum_face_normal_method1[:,1])))
        print("     Max of |sum_faces face normal (nx)| = ", np.max(abs(sum_face_normal[:,0])))
        print("     Max of |sum_faces face normal (ny)| = ", np.max(abs(sum_face_normal[:,1])))
        
        
        #----------------------------------------------------
        # Check the maximum sum over all cells. Must be zero.
        
        #Use the maximum sqrt(vol) as a reference magnitude 
        #for checking zero face normal sum.
        sqrts = np.sqrt(np.asarray([[cell.volume for cell in self.cells]]))
        ref_mag = np.max( sqrts )
        print()
        print("              Reference magnitude = ", ref_mag)
        print()
        
        if (np.max(abs(sum_face_normal[:,0]))>tol or np.max(abs(sum_face_normal[:,1]))>tol):
            print(" Max face vector sum over a cell is larger than machine zero... " )
            print(" Something is wrong. Stop.")
            assert(False),"ERROR: Sum of normals around the cells is not zero"
        #print(" Max face vector sum over a cell is larger than machine zero... Something is wrong. Stop.")
        return
    
    
    #--------------------------------------------------------
    # Centroids for each cell are always within the cell
    #--------------------------------------------------------
    def check_centroids(self):
        print(" --- Check the centroids:")
        for i, cell in enumerate(self.cells):
            # nodes are ordered counterclockwise.  Take two consecutive nodes:
            for k in range(cell.num_nodes):
                v1 = cell.nodes[k] #node object representing the vertex
                if (k+1 == cell.num_nodes):
                    v2 = cell.nodes[0] #<back to the first vertex
                else:
                    v2 = cell.nodes[k+1]
                
                x1,y1 = v1.vector
                x2,y2 = v2.vector
                
                #if self.winding == 'cw':
                volk = triangle_area_from_raw_data(x1,x2, cell.centroid[0], y1,y2, cell.centroid[1])
                
                #else:
                #    volk = triangle_area_from_raw_data(cell.centroid[0],x2,x1, cell.centroid[1],y2,y1)
                
                # If volume_k is negative, stop.
                if (volk < 0.0):
                    print(" A centroid is outside the cell... Cell = ", i, " Stop...")
                    print("                      (xc,yc) = ", cell.centroid)
                    print("                      (x1,y1) = ", x1, y1)
                    print("                      (x2,y2) = ", x2, y2)
                    print("  Partial volume (tria c-1-2) = ", volk)
                    assert(False),"ERROR: Centroids outside the cell!"
        print("  No centroids found to be located outside a cell. Good.")
            
        return
    
    
    #--------------------------------------------------------
    # Volume check
    #--------------------------------------------------------
    #def check_volume2(self):
    #    print(" --- Check the total volume:")
    #    sum_ = 1.0e-15
    #    for i, cell in enumerate(self.cells):
    #        sum_ += cell.volume * sum_
        
    
    
    #------------------------------------------------------------------------------------
    # Compute mesh spacing statistics.
    #
    #   heffn     = Effecrtive spacing based on # of nodes
    #   heffc     = Effecrtive spacing based on # of cells
    #   heffv     = Average of sqrt(volume).
    #   heffv_min = Minimum sqrt(volume).
    #   heffv_max = Maximum sqrt(volume).
    #
    def compute_mesh_statistics(self):
        one = 1.0
        
        self.heffc = np.sqrt( one/float(self.nCells) ) # Effecrtive spacing based on # of cells
        
        self.heffv     = np.sqrt( self.cells[0].volume )
        heffv_min = np.sqrt( self.cells[0].volume )
        heffv_max = np.sqrt( self.cells[0].volume )
        
        for i in range(1, self.nCells):
            cell = self.cells[i]
            self.heffv += np.sqrt( cell.volume  )
            heffv_min = min( heffv_min, np.sqrt( cell.volume  ) )
            heffv_max = max( heffv_max, np.sqrt( cell.volume ) )
        
        self.heffv /= float(self.nCells)
        return
    
    #-------------------------------------------------------------------------#
    # Done with checks
    #-------------------------------------------------------------------------#

class TestInviscidVortex(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2vortex = os.path.join(uplevel, 'case_unsteady_vortex')
        self.DHandler = DataHandler(project_name = 'vortex',
                                       path_to_inputs_folder = path2vortex)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='tri',
                         winding='ccw')
    
    
class TestTEgrid(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2vortex = os.path.join(uplevel, 'case_verification_te')
        self.DHandler = DataHandler(project_name = 'te_test',
                                       path_to_inputs_folder = path2vortex)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='tri',
                         winding='ccw')
        
        
class TestSteadyCylinder(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_steady_cylinder')
        self.DHandler = DataHandler(project_name = 'cylinder',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='tri',
                         winding='ccw')
        
        
class TestSteadyAirfoil(object):
    
    def __init__(self):
        # up a level
        #uplevel = os.path.join(os.path.dirname(__file__), '..','cases')
        uplevel = os.path.join(os.path.dirname(os.getcwd()), 'cases')
        #path2vortex = uplevel+'\\cases\case_unsteady_vortex'
        path2case = os.path.join(uplevel, 'case_steady_airfoil')
        self.DHandler = DataHandler(project_name = 'airfoil',
                                       path_to_inputs_folder = path2case)
        
        
        self.grid = Grid(generated=False,
                         dhandle = self.DHandler,
                         type_='quad',
                         winding='ccw')

    
    
    
if __name__ == '__main__':
    gd = Grid(type_='rect',m=10,n=10, winding='ccw')
    self = Grid(type_='tri',m=10,n=10, winding='ccw')
    
    cell = self.cellList[44]
    face = cell.faces[0]
    
    """
    plotRect = PlotGrid(gd)
    axRect = plotRect.plot_cells()
    axRect = plotRect.plot_centroids(axRect)
    axRect = plotRect.plot_face_centers(axRect)
    axRect = plotRect.plot_normals(axRect)
    #axRect = plotRect.plot_boundary(axRect) #BGrid does not exist for generated grids
    #"""
    
    #print(face.e_eta
    #print(face.e_xi
    
    #cell = self.cellList[0]
    #face = cell.faces[0]
    
    
    #plotTri = PlotGrid(self)
    """
    axTri = plotTri.plot_cells()
    axTri = plotTri.plot_centroids(axTri)
    axTri = plotTri.plot_face_centers(axTri)
    axTri = plotTri.plot_normals(axTri)
    #axTri = plotTri.plot_boundary(axTri) #BGrid does not exist for generated grids
    
    plotRect = PlotGrid(gd)
    axRect = plotRect.plot_cells()
    axRect = plotRect.plot_centroids(axRect)
    axRect = plotRect.plot_face_centers(axRect)
    axRect = plotRect.plot_normals(axRect)
    #"""
    
    #del(self)
    #del(gd)
    
    
    #test = TestInviscidVortex()
    test = TestTEgrid()
    #test = TestSteadyAirfoil()
    #test = TestSteadyCylinder()
    
    #'''
    #
    self = test.grid
    #
    #'''
    plotter = PlotGrid(self)
    
    #"""
    #ax = plotter.plot_cells()
    #ax = plotter.plot_centroids(ax)
    #ax = plotter.plot_face_centers(ax)
    #ax = plotter.plot_normals(ax)
    ax = plotter.plot_boundary()
    #"""
    
    """
    ax = plotter.plot_boundary()
    ax = plotter.plot_centroids(ax)
    ax = plotter.plot_face_centers(ax)
    ax = plotter.plot_normals(ax)
    #"""