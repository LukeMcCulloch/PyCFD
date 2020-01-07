# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:35:53 2020

@author: Luke.McCulloch
"""

import numpy as np

from Overload import Overload
from PlotGrids import PlotGrid

from Utilities import normalize, normalized, norm, dot, cross, \
    normalize2D, normalized2D

class Node(Overload):
    def __init__(self, vector, nConserved=3):
        self.x0 = vector[0]
        self.x1 = vector[1]
        self.vector = vector
        self.parent_faces = [] #nodes are created before faces
        # and before cells, so we can wait until face creation
        # to start filling this in.
        
        self.nConserved     = nConserved
        self.Q              = np.zeros((nConserved,1),float)
        
    #def __call__(self):
    #    return self.vector
    
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
                 nConserved=3):
        self.nodes = nodes
        self.parentcell = parentcell
        self.fid = fid
        self.adjacentface   = None
        self.e_xi           = None
        self.e_eta          = None
        self.nConserved     = nConserved
        self.Q              = np.zeros((nConserved,1),float)
        self.isBoundary = True
        for node in self.nodes:
            node.parent_faces.append(self)
            
        self.cell = parentcell
        self.N = 2 #len(nodes)
        self.center = .5*(self.nodes[0] + self.nodes[1])
        #sumx0 = sum( [el.x0 for el in self.nodes] )
        #sumx1 = sum( [el.x1 for el in self.nodes] )
        self.area = np.linalg.norm(self.nodes[1]-self.nodes[0])
        self.normal_vector = self.compute_normal(normalize = True)
        
    def compute_normalfancy(self):
        """
        cross vector with 0 in x2 dir
        with vector with 1 in the x2 dir
         
        this is of course (x1,-x0, 0)
         
        normals point in
         
        """
        #vec = self.nodes[1] - self.nodes[0]
        
        dumvec1 = np.zeros((3),float)
        dumvec2 = np.zeros_like((dumvec1))
        dumvec1[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[:-1] = self.nodes[1] - self.nodes[0]
        dumvec2[-1] = 1.
        
        n3 = normalize(cross(dumvec1,dumvec2) )
        return n3[:-1]
        
                      
    def compute_normal(self, normalize=True):
        """ 2D specific face normals
        normalized(x1,-x0)
        """
        vec = self.nodes[1] - self.nodes[0]
        vec = np.asarray([vec[1],-vec[0]])
        if normalize:
            vec =  normalize2D(vec)
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
        return magXi, Xidiff*magXi
    
    
    def compute_e_eta(self):
        """
        transverse face unit normal vector
        """
        A = self.nodes[0]
        B = self.nodes[1]
        Etadiff = B-A
        magEtadiff = 1./np.linalg.norm(Etadiff)
        return magEtadiff, Etadiff*magEtadiff
    
    
    def compute_direct_diffusion_constant(self):
        """
        That part of the direct diffusion 
        which is constant 
        while geometry is constant
        """
        return
    
    def compute_Dphi_Dxi(self):
        Qa = self.parentcell.Q
        Qp =  self.adjacentface.parentcell.Q
        return (Qa-Qp)/self.hxi
        
    def compute_cross_diffusion_constant(self):
        """
        That part of the cross diffusion 
        which is constant 
        while geometry is constant
        """
        return
        
        
class Cell(object):
    """
    The Cij'th cell
    
    ccw-winding
    """
    def __init__(self, nodes, cid, nface,
                 nConserved=3): #, FaceCellMap):
        self.nodes = nodes
        self.N = len(self.nodes)
        self.num_faces = self.N
        self.set_centroid()
        self.F = np.asarray((self.num_faces),float)
        self.G = np.asarray((self.num_faces),float)
        self.faces = []
        self.cid = cid
        self.set_face_vectors(nface)
        self.Q = np.zeros((nConserved,1),float)
        
    def set_centroid(self):
        """The 'center' of this face
        -center as average point location
        
        -circumcentric?
        -barycentric?
        """
        scale = 1./float(self.N)
        self.centroid = scale * sum([el.vector for el in self.nodes]) 
        return self.centroid
        
    def set_face_vectors(self, nface):   #, n, FaceCellMap):
        for i in range(self.N):
            self.faces.append( Face( [self.nodes[i],
                                      self.nodes[(i+1)%self.N] 
                                      ],
                               parentcell=self,
                               fid = nface))
            nface += 1
        return #n, FaceCellMap
    
    def get(self, i):
        assert(i<=self.N),'error, i>N'
        return self.N%i
    
    def face(self, i):
        return 
    
    def print_vertices(self):
        for vert in self.nodes:
            print vert.vector
        return
    
    
    def print_faces(self, normals=False):
        for face in self.faces:
            print 'face'
            for vert in face.nodes:
                print vert.vector
            if normals:
                print 'face normal'
                print face.compute_normal()
        return


class Grid(object):
    
    def __init__(self, mesh=None, m=10,n=10,type_='rect'):
        self.gridtype = {'rect':0,
                         'tri':1}
        self.nCells = 0
        self.nFaces = 0
        self.nNodes = m*n
        
        self.cellList = []
        
        self.type = type_
        if mesh is None:
            mesh = np.zeros((2,m,n),float) # C-ordering.  last index is most rapid
            self.dim = 2
            self.m = m
            self.n = n
        else:
            shp = np.shape(mesh)
            self.dim = shp[0]
            self.m = shp[1]
            self.n = shp[2]
            
        mms = np.linspace(0.,1.,m)
        nms = np.linspace(0.,1.,n)
        self.mesh = mesh
        #
        """
        Nodes are defined first
        """
        self.nodes = []
        for i in range(self.m):
            self.nodes.append([])
            for j in  range(self.n):
                self.mesh[0,i,j] = mms[i]
                self.mesh[1,i,j] = nms[j]
                self.nodes[i].append(Node(self.mesh[:,i,j]))
                
        self.nodes = np.asarray(self.nodes)
        
        
        # now cells and faces:
        self.FaceCellMap = {}
        self.make_cells()
        # maps
        self.make_FaceCellMap()
        
    
    def make_cells(self):
        """
        Nodes are defined first
        so, 
        Coincident Cells share nodes
        """
        if self.type == 'rect':
            self.make_rect_cells()
        else:
            self.make_tri_cells()
        return
    
    
    def make_rect_cells(self):
        self.cells = []
        for i in range(self.m-1):
            self.cells.append([])
            for j in  range(self.n-1):
                
                self.cells[i].append(
                                    Cell([self.nodes[i  ,j  ],
                                          self.nodes[i  ,j+1],
                                          self.nodes[i+1,j+1],
                                          self.nodes[i+1,j  ] ],
                                         cid=self.nCells,
                                         nface = self.nFaces)
                                    )
                self.cellList.append(self.cells[i][-1])
                self.nFaces += 4
                self.nCells +=1
        self.cells = np.asarray(self.cells)
        return
    
    
    def make_tri_cells(self):
        self.cells = []
        for i in range(self.m-1):
            self.cells.append([])
            for j in  range(self.n-1):
                
                self.cells[i].append(
                                    Cell([self.nodes[i  ,j  ],
                                          self.nodes[i  ,j+1],
                                          self.nodes[i+1,j  ] ],
                                         cid=self.nCells,
                                         nface = self.nFaces)
                                    )
                self.cellList.append(self.cells[i][-1])
                self.nCells +=1
                self.nFaces += 3
                
                self.cells[i].append(
                                    Cell([self.nodes[i  ,j+1],
                                          self.nodes[i+1,j+1],
                                          self.nodes[i+1,j  ] ],
                                         cid=self.nCells,
                                         nface = self.nFaces)
                                    )
                self.cellList.append(self.cells[i][-1])
                self.nCells += 1
                self.nFaces += 3
                
        self.cells = np.asarray(self.cells)
        return
    
    def make_FaceCellMap(self):
        """
        iterate over all faces, 
        and save the map from face to cell it belongs to
        
        also, find adjacent faces and map them
            
        see:
            https://scicomp.stackexchange.com/questions/24981/
            getting-adjacent-cells-map-for-an-unstructured-polyhedral-mesh
        """
        for cell in self.cellList:
            for face in cell.faces:
                self.FaceCellMap[face.fid] = cell #old fashioned way
                #self.FaceCellMap[face] = cell
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
                            #adjacentface = el
                            face.adjacentface = el
                            el.adjacentface = face
                            el.isBoundary = False
                            face.isBoundary = False
                            #print face, face.adjacentface
                            # go ahead and set local vectors 
                            # between adjacent cells
                            face.hxi, face.e_xi   = face.compute_e_xi()
                            face.heta, face.e_eta = face.compute_e_eta()
                            el.hxi, el.e_xi      = el.compute_e_xi()
                            el.heta, el.e_eta    = el.compute_e_eta()
                            break
        return
    
    
    def make_AdjacentFaceMap(self):
        return

        
if __name__ == '__main__':
    gd = Grid(type_='rect',m=10,n=10)
    self = Grid(type_='tri',m=10,n=10)
    
    cell = self.cellList[44]
    face = cell.faces[0]
    
    #print face.e_eta
    #print face.e_xi
    
    #cell = self.cellList[0]
    #face = cell.faces[0]
    
    
    plotTri = PlotGrid(self)
    axTri = plotTri.plot_cells()
    axTri = plotTri.plot_centroids(axTri)
    axTri = plotTri.plot_face_centers(axTri)
    axTri = plotTri.plot_normals(axTri)
    
    plotRect = PlotGrid(gd)
    axRect = plotRect.plot_cells()
    axRect = plotRect.plot_centroids(axRect)
    axRect = plotRect.plot_face_centers(axRect)
    axRect = plotRect.plot_normals(axRect)