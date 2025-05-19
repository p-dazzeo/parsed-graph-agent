# Custom Graph implementation to replace the missing google.adk.graph module
import networkx as nx

class Graph:
    """A simple wrapper around NetworkX DiGraph to provide compatibility with the expected API."""
    
    def __init__(self):
        self._graph = nx.DiGraph()
    
    def add_node(self, node_id, **attr):
        """Add a node to the graph with the given attributes."""
        self._graph.add_node(node_id, **attr)
    
    def add_edge(self, source, target, **attr):
        """Add an edge from source to target with the given attributes."""
        self._graph.add_edge(source, target, **attr)
    
    def nodes(self, data=False):
        """Return a list of nodes in the graph.
        
        If data=True, return a list of (node, attribute_dict) tuples.
        """
        return self._graph.nodes(data=data)
    
    def edges(self, data=False):
        """Return a list of edges in the graph.
        
        If data=True, return a list of (u, v, attribute_dict) tuples.
        """
        return self._graph.edges(data=data)
    
    def number_of_nodes(self):
        """Return the number of nodes in the graph."""
        return self._graph.number_of_nodes()
    
    def number_of_edges(self):
        """Return the number of edges in the graph."""
        return self._graph.number_of_edges()
    
    def has_path(self, source, target):
        """Return True if there is a path from source to target."""
        return nx.has_path(self._graph, source, target)
    
    def remove_edge(self, source, target):
        """Remove the edge from source to target."""
        self._graph.remove_edge(source, target)
    
    def remove_node(self, node):
        """Remove node from the graph."""
        self._graph.remove_node(node)
    
    def in_degree(self, node):
        """Return the in-degree of node."""
        return self._graph.in_degree(node)
        
    def has_edge(self, source, target):
        """Return True if the graph contains the edge (source, target)."""
        return self._graph.has_edge(source, target)