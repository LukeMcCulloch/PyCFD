# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:35:53 2020

@author: Luke.McCulloch
"""

import numpy as np

from Overload import Overload
from PlotGrids import PlotGrid

from Utilities import normalize, normalized, norm, dot, cross, \
    normalize2D, normalized2D, triangle_area

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
        self.cell = parentcell
        #
        # Basic geometry
        #
        self.area = np.linalg.norm(self.nodes[1]-self.nodes[0])
        self.normal_vector, self.bface_nrml_mag = self.compute_normal(normalize = True)
        
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
            vec, nmag =  normalize2D(vec,return_mag=True)
            #vec =  normalize2D(vec,return_mag=False)
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
    
    
        
    
        
        
class Cell(object):
    """
    The Cij'th cell, tris and quads only
    
    ccw-winding
    """
    def __init__(self, nodes, cid, nface,
                 nConserved=3, facelist=None): #, FaceCellMap):
        self.nodes = nodes
        self.N = len(self.nodes)
        self.num_faces = self.N
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
        else:
            vol = 0.
            for i in range(self.N-3+1):
                nlist1 = self.nodes[i:i+3]
                vol += triangle_area(*nlist1)
            self.volume = vol
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
    
    def compute_gradient_reconstruction_A_matrix(self):
        self.A = np.zeros((self.N,2),float)
        for i,face in enumerate(self.faces):
            #cell = face.adjacentface.parentcell
            self.A[i] = face.e_xi * face.hxi
        
        self.AAinv = np.linalg.inv(
                                    np.dot(self.A.T,self.A) 
                                  )
        return


class Grid(object):
    
    def __init__(self, mesh=None, m=10,n=10,type_='rect'):
        self.gridtype = {'rect':0,
                         'tri':1}
        self.nCells = 0
        self.nFaces = 0
        self.nNodes = m*n
        
        self.nodeList = []
        self.cellList = []
        self.faceList = []
        self.boundaryList = []
        
        self.FToV = None #face to vertex (node) connectivity matrix
        self.EToV = None #cell (element) to vertex
        self.EToE = None #cell to cell
        
        
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
        nid = 0
        self.nodes = []
        for i in range(self.m):
            self.nodes.append([])
            for j in  range(self.n):
                self.mesh[0,i,j] = mms[i]
                self.mesh[1,i,j] = nms[j]
                node = Node(self.mesh[:,i,j], nid)
                nid += 1
                self.nodes[i].append(node) #to become 2D array (not necessary, but I do for python-fun)
                self.nodeList.append(node) #will stay as list
                
        self.nodes = np.asarray(self.nodes)
        
        
        # now cells and faces:
        self.FaceCellMap = {}
        self.make_cells()
        # maps
        self.make_FaceCellMap()
        
        self.make_neighbors()
    
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
        self.cells = np.asarray(self.cells)
        return
    
    
    def make_tri_cells(self):
        self.cells = []
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
                    self.boundaryList.append(face)
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
    
    
    #-------------------------------------------------------------------------#
    # Mesh checks
    #-------------------------------------------------------------------------#
    def check_volume(self, tol=1e-14):
        v1 = self.sum_volume_green_gauss()
        v2 = self.sum_volume_cell_sum()
        if (abs(v1-v2)>tol):
            print " Volume difference is larger than round-off error... Something is wrong. Stop."
        return
    
    def sum_volume_green_gauss(self):
        vol = 0.
        for bound in self.boundaryList:
            mid = bound.center
            vol += np.dot( mid,bound.normal_vector )*bound.bface_nrml_mag
        return -0.5*vol
    
    def sum_volume_cell_sum(self):
        vol = 0.
        for cell in self.cellList:
            vol += cell.volume
        return vol
    #-------------------------------------------------------------------------#
    # Done with checks
    #-------------------------------------------------------------------------#
        
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