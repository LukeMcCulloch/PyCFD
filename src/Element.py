"""
Luke McCulloch

triangle class
"""

import numpy as np

from Utilities import normalize, normalized, norm, dot, cross

#
# The invariant for how staticGeometry works is:
#   - The staticGeometry flag should only be changed by setting/unsetting it
#     in the mesh which holds this object (to ensure it gets set/unset everywhere,
#     since the result of cached computation may be stored in a distant object)
#   - If there is anything in the cache, it must be valid to return it. Thus,
#     the cache must be emptied when staticGeometry is set to False.
#
# It would be nice to automatically empty the cache whenever a vertex position is
# changed and forget about the flag. However, this would require recusively updating
# all of the caches which depend on that value. Possible, but a little complex and
# maybe slow.
def cacheGeometry(f):
    name = f.__name__
    def cachedF(self=None):
        if name in self._cache: return self._cache[name]
        res = f(self)
        if self.staticGeometry: self._cache[name] = res
        return res
    return cachedF



class halfedge(object):
    def __init__(self, staticGeometry=False):
        ### Members
        self.twin = None
        self.next = None
        self.vertex = None
        self.edge = None
        self.face = None
        
        self._cache = dict()
        self.staticGeometry = staticGeometry
        
        # Global id number, mainly for debugging
        global NEXT_HALFEDGE_ID
        self.id = NEXT_HALFEDGE_ID
        NEXT_HALFEDGE_ID += 1
        
    def __str__(self):
        return "<HalfEdge #{}>".format(self.id)
    
    def __repr__(self):
        return self.__str__()
    
    # Return a boolean indicating whether this is on the boundary of the mesh
    @property
    def isBoundary(self):
        return not self.twin.isReal

    @property
    @cacheGeometry
    def vector(self):
        """The vector represented by this halfedge"""
        v = self.vertex.position - self.twin.vertex.position
        return v

class edge(self, staticGeometry=False)):
    def __init__(self):
        
class triangle(object, staticGeometry=False)):
    def __init__(self, p1,p2,p3):
        self.make_nodes(p1,p2,p3)
        self.make_edges(p1,p2,p3)
        
    def make_nodes(self, p1,p2,p3):
        self.nodes = {}
        self.nodes[0] = p1
        self.nodes[1] = p2
        self.nodes[2] = p3
        return
    def make_edges(self, p1,p2,p3):
        self.edge = {}
        self.edge[1] = np.asarray([p2,p1])
        self.edge[2] = np.asarray([p3,p2])
        self.edge[3] = np.asarray([p1,p3])
        return
    
    
    def embedded_normal(self):
        """The normal vector for this face
        (2D embedded in 3D)"""
        v = self.nodes
        n = normalize(cross(v[1].position - v[0].position, 
                            v[2].position - v[0].position))

        return n
    
    
    def area(self):
        """The area of this tri"""
        v = self.nodes
        a = 0.5 * norm(cross(v[1].position - v[0].position, 
                             v[2].position - v[0].position))
        return a
    
    
    
class quad(object, staticGeometry=False)):
    def __init__(self, p1,p2,p3,p4):
        self.make_nodes(p1,p2,p3,p4)
        self.make_edges(p1,p2,p3,p4)
        return
    def make_nodes(self, p1,p2,p3,p4):
        self.nodes = {}
        self.nodes[0] = p1
        self.nodes[1] = p2
        self.nodes[2] = p3
        self.nodes[3] = p4
        return
    def make_edges(self, p1,p2,p3,p4):
        self.edge = {}
        self.edge[1] = np.asarray([p2,p1])
        self.edge[2] = np.asarray([p3,p2])
        self.edge[3] = np.asarray([p4,p3])
        self.edge[4] = np.asarray([p1,p4])
        return