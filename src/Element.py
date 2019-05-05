"""
Luke McCulloch

triangle class
"""

import numpy as np
    
        
class triangle(object):
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
    
    
    
class quad(object):
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