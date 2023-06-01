"""File Contains Class Definitions for Web Graph and Document Nodes"""

###################################################################
#                           Imports                               #
###################################################################
import pickle
from urllib.parse import urlparse

###################################################################
#                Global Variabls and Constants                    #
###################################################################
NON_HOST_NAME_PAGE_RANK_FACTOR = 10


###################################################################
#                           Classes                               #
###################################################################
class Graph:
    """Graph of entire courpus."""
    def __init__(self) -> None:
        """Nodes are stored in dictionary where key is first url and later the docID.  Value is Node."""
        self.nodes = {}


    def addNode(self, url:str, node:object):
        """Add Node to graph nodes using url."""
        self.nodes[url] = node


    def convertUrlsToNodes(self) -> None:
        """Takes graph that has nodes with only the information of the urls of their children.
            Adds node as a parent to child node (indexed by child url in graph.nodes).  Then adds
            node as child of parents to create a working graph."""
        # Iterate through all nodes
        for node in self.nodes.values():
            # First iterate through url's of children
            for child in node.children_urls:
                if child in self.nodes:
                    self.nodes[child].parents.add(node) # Find child of Node through graph.nodes with url as key and add current Node as parent

        # Iterate thorugh all nodes
        for node in self.nodes.values():
            # Iterate through parent nodes
            for parent in node.parents:
                parent.children.add(node) # Add curernt node as child of each parent node


    def convertGraphForSaving(self) -> None:
        """Can't pickle graph in working state because of recursion error.  So instead converts
            children and parent Nodes in each Node to just the docId of the node which can be used
            to rebuild the graph later similarly to urls."""
        # Convert self.nodes so it indexes by DocId instead of url
        self.nodes = {node.docID: node for node in self.nodes.values()}

        # Iterate through nodes
        for node in self.nodes.values():
            # Iterate through children
            for child in [node for node in node.children]:
                node.children.remove(child) # Remove child Node Objects from current Node
                node.children.add(child.docID) # Replace with DocID of child node
            # Iterate through parents
            for parent in [node for node in node.parents]:
                node.parents.remove(parent) # Remove parent Node Objects from current Node
                node.parents.add(parent.docID) # Replace with DocID of parent node

    
    def convertGraphForLoading(self) -> None:
        """Takes graph that has nodes with only the information of the doc'id's of their children.
            Adds node as a parent to child node (indexed by docId in graph.nodes).  Then adds
            node as child of parents to create a working graph."""
        # Iterate through nodes
        for node in self.nodes.values():
            # Iterate through children
            for docId in [node for node in node.children]:
                node.children.remove(docId) # Remove DocID of child
                node.children.add(self.nodes[docId]) # Replace with node corresponding to DocID
            # Iterate through parents
            for docId in [node for node in node.parents]:
                node.parents.remove(docId) # Remove DocID of parent
                node.parents.add(self.nodes[docId]) # Replace with node corresponding to DocID

    
    def runHits(self):
        """Run's one iteration of HITS Algorithm."""
        for node in self.nodes.values():
            node.updateAuthority()
        for node in self.nodes.values():
            node.updateHub()
        self.normalize_nodes()


    def runPageRank(self, d):
        """Runs one iteration of Page Rank Algorithm."""
        for node in self.nodes.values():
            node.updatePageRank(d, len(self.nodes))


    def normalize_nodes(self) -> None:
        """Nodmalizes authority and hub scores of all nodes in graph for HITS Algorithm"""
        authority_sum = sum(node.authority for node in self.nodes.values())
        hub_sum = sum(node.hub for node in self.nodes.values())
        for node in self.nodes.values():
            if authority_sum:
                node.authority /= authority_sum
            if hub_sum:
                node.hub /= hub_sum

    
    def save(self):
        """Saves converted version of self to file."""
        self.convertGraphForSaving()
        with open(f'webGraph/webGraph.pickle', 'wb') as web_graph_file:
            pickle.dump(self.nodes, web_graph_file, pickle.HIGHEST_PROTOCOL)
        print("\tSaved Web Graph")
    


class Node:
    """Node representation of document in corupus."""
    def __init__(self, url: str, docID: int) -> None:
        self.url = url
        self.children_urls = set()
        self.children = set()
        self.parents = set()
        self.authority = 1.0
        self.hub = 1.0
        self.page_rank = 1.0
        self.docID = docID


    def updateAuthority(self):
        """Updates authority of Node for HITS Algorithm"""
        self.authority = sum(node.hub for node in self.parents)


    def updateHub(self):
        """Updates hub of Node for HITS Algorithm"""
        self.hub = sum(node.authority for node in self.children)


    def updatePageRank(self, damping_factor, n):
        """Update's page_rank value of Node for Page Rank Algorithm."""
        host_name = urlparse(self.url).hostname
        pagerank_sum = sum((node.page_rank / len(node.children))*NON_HOST_NAME_PAGE_RANK_FACTOR for node in self.parents if urlparse(node.url).hostname != host_name)
        pagerank_sum += sum((node.page_rank / len(node.children)) for node in self.parents)
        random_walk = damping_factor / n
        self.page_rank = random_walk + (1 - damping_factor) * pagerank_sum


###################################################################
#                          Functions                              #
###################################################################

def runHits(documents, web_graph):
    """Creates a graph with given documents and the documents that point to
        those documents.  Then runs hits on the graph 5 times.  Then returns
        the graph."""
    
    # Initialize Empty Grpah
    hits_graph = Graph()
    # Add node to graph for each document
    for document in documents:
        hits_graph.nodes[document] = web_graph.nodes[document]
    # Adds to the graph the nodes pointing to each node already in the graph
    for node in [node for node in hits_graph.nodes.values()]:
        for parent in sorted([parent for parent in node.parents], key = lambda node: node.page_rank, reverse=True)[:10]:
            hits_graph.nodes[parent.docID] = parent

    # Run's Hits for 5 Iterations
    for i in range(5):
        hits_graph.runHits()


    return hits_graph