import networkit as nk
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator  # native operators
import mpld3
from joblib import Parallel, delayed

import matplotlib.cm as cm
from matplotlib.colors import Normalize

# for saving tikz
import os
import sys
from network2tikz import plot

import copy
from scipy.special import comb
import pandas as pd
from scipy.spatial import cKDTree

class GlobalCascadeNetwork:

    def __init__(self,
                 network_generator=nk.generators.ErdosRenyiGenerator,
                 phis=None,
                 phi_distribution=np.random.uniform,
                 seeds=None,
                 num_seeds=1,
                 rule=operator.ge,
                 **kwargs):
        '''
        '''

        # handle default params for two standard network types
        if network_generator is None or network_generator == 'empty':
            options = {
                'N': 1,
                'p': 0}
            options.update(kwargs)


        elif network_generator == 'layered':

            options = {
                'layer_sizes': [5,5,5],
                'p': .1,
                #'phis': None,
                'directed': False}

            options.update(kwargs)

        elif network_generator == 'layered_random':

            options = {
                'num_layers': 5,
                'p': .1,
                #'phis': None,
                'max_layer_size':10,
                'min_layer_size':1,
                'input_layer_size':0,
                'output_layer_size':0,
                'directed': False
                }

            options.update(kwargs)

        elif network_generator is nk.generators.ErdosRenyiGenerator:
            # some default options for the ErdosRenyiGenerator
            options = {
                'N': 100,
                'p': .1}

            options.update(kwargs)

        elif network_generator is nk.generators.BarabasiAlbertGenerator:
            # some default options for the BarabasiAlbertGenerator
            options = {'k': 1,
                       'nMax': 100,
                       'n0': 1}

            options.update(kwargs)

        elif network_generator.__name__ == 'networkx.algorithms.bipartite':
            options = {'n': 3,
                       'm': 1}

            options.update(kwargs)

        self.network_generator = network_generator

        # the layout
        self.pos = None

        # create Networkit graph
        self.create_network(**options)

        # set phi values
        self.set_phi_values(phis, phi_distribution)

        # set the cascade rule operator
        self.rule = rule

        # set labels
        self.set_seeds(seeds, num_seeds)

        # set all nodes as unvisited
        self.set_unvisited()

        # for convenience, find the vulnerable nodes and vulnerable cluster
        #self.synchronous_find_vulnerable_nodes()
        #self.synchronous_find_vulnerable_cluster()

    # high level setup functions

    def create_network(self,
                       **kwargs):
        '''
        '''

        self.directed = False

        if self.network_generator is None or self.network_generator == 'empty':
            if 'N' in kwargs:
                N = kwargs['N']
            if 'p' in kwargs:
                p = 0

            self.network_generator = nk.generators.ErdosRenyiGenerator
            self.G = self.network_generator(N, p).generate()
            self.nxG = nk.nxadapter.nk2nx(self.G)
            self.pos = nx.random_layout(self.nxG)

        elif self.network_generator == 'layered':
            if 'layer_sizes' in kwargs:
                layer_sizes = kwargs['layer_sizes']
            if 'p' in kwargs:
                p = kwargs['p']
            if 'directed' in kwargs:
                directed = kwargs['directed']

            self.directed = directed
            self.create_layered_graph(layer_sizes, p)

        elif self.network_generator == 'layered_random':
            if 'num_layers' in kwargs:
                num_layers = kwargs['num_layers']
            if 'p' in kwargs:
                p = kwargs['p']
            if 'min_layer_size' in kwargs:
                min_layer_size = kwargs['min_layer_size']
            if 'max_layer_size' in kwargs:
                max_layer_size = kwargs['max_layer_size']
            if 'input_layer_size' in kwargs:
                input_layer_size = kwargs['input_layer_size']
            if 'output_layer_size' in kwargs:
                output_layer_size = kwargs['output_layer_size']
            if 'directed' in kwargs:
                directed = kwargs['directed']

            self.directed = directed

            #self.create_layered_graph(layer_sizes, p)
            self.create_random_layered_network(num_layers=num_layers,
                                               p=p,
                                               min_layer_size=min_layer_size,
                                               max_layer_size=max_layer_size,
                                               input_layer_size=input_layer_size,
                                               output_layer_size=output_layer_size)


        elif self.network_generator is nk.generators.ErdosRenyiGenerator:
            if 'N' in kwargs:
                N = kwargs['N']
            if 'p' in kwargs:
                p = kwargs['p']

            self.G = self.network_generator(N, p).generate()
            self.initialize_available_edges()
            self.nxG = nk.nxadapter.nk2nx(self.G)
            self.pos = nx.random_layout(self.nxG)

        elif self.network_generator is nk.generators.BarabasiAlbertGenerator:
            if 'k' in kwargs:
                k = kwargs['k']
            if 'nMax' in kwargs:
                nMax = kwargs['nMax']
            if 'n0' in kwargs:
                n0 = kwargs['n0']

            self.G = self.network_generator(k, nMax, n0).generate()
            self.nxG = nk.nxadapter.nk2nx(self.G)

        elif self.network_generator.__name__ == 'networkx.algorithms.bipartite':
            if 'n' in kwargs:
                n = kwargs['n']
            if 'm' in kwargs:
                m = kwargs['m']

            self.nxG = self.network_generator.complete_bipartite_graph(n, m)
            self.G = nk.nxadapter.nx2nk(self.nxG)
            self.pos = nx.bipartite_layout(self.nxG, nodes=[0, 1, 2], aspect_ratio=.05)
            self.pos[3][1] = self.pos[1][1]

        # for convenience, get a few values
        self.N = self.G.numberOfNodes()
        self.E = self.G.numberOfEdges()

    def create_random_layered_network(self,
                                      min_layer_size=1,
                                      max_layer_size=10,
                                      num_layers=5,
                                      input_layer_size = 0,
                                      output_layer_size = 0,
                                      p=.5):

        layer_sizes = self.random_layer_sizes(min_layer_size=min_layer_size,
                                              max_layer_size=max_layer_size,
                                              num_layers=num_layers,
                                              m = input_layer_size,
                                              n = output_layer_size)

        self.create_layered_graph(layer_sizes, p)

        self.G = nk.nxadapter.nx2nk(self.nxG)

    def random_layer_sizes(self, min_layer_size=1, max_layer_size=10, num_layers=5, p=.5, m=0, n=0):

        layer_sizes = (np.random.randint((max_layer_size + 1) - min_layer_size,
                                         size=[1, num_layers]) + min_layer_size).flatten()

        if m > 0:
            layer_sizes[0] = m

        if n > 0:
            layer_sizes[-1] = n

        return (layer_sizes)

    def create_layered_graph(self, layer_sizes, p):
        '''
        create a layered graph, also calculating the node positions
        '''

        if self.directed:
            self.nxG = nx.empty_graph(layer_sizes[0], create_using=nx.DiGraph())
        else:
            self.nxG = nx.empty_graph(layer_sizes[0])

        num_layers = len(layer_sizes)
        self.num_layers = num_layers

        pos = {}
        x_values = self.calc_layered_x_values(num_layers)

        self.add_layers(layer_sizes, num_layers, p, pos, x_values)

        self.pos = pos
        self.x_values = x_values
        self.G = nk.nxadapter.nx2nk(self.nxG)
        self.layers = nx.get_node_attributes(self.nxG, 'layer')
        self.layer_sizes = layer_sizes

        self.get_complete_edges_layered()
        self.initialize_available_edges()

    def calc_layered_x_values(self, num_layers):
        x_values = np.arange(0, 1, 1 / num_layers)
        return x_values

    def add_layers(self, layer_sizes, num_layers, p, pos, x_values):

        new_nodes = None

        for i in range(num_layers):

            num_nodes_in_layer = layer_sizes[i]
            num_nodes_in_graph = self.nxG.number_of_nodes()

            new_nodes = self.add_layer(i, num_nodes_in_graph, num_nodes_in_layer, p, pos, x_values, new_nodes)

    def add_layer(self, i, num_nodes_in_graph, num_nodes_in_layer, p, pos, x_values, new_nodes = None):
        # set node ids according to number of nodes already in graph
        if i > 0:
            last_layer = new_nodes
            new_nodes = list(range(num_nodes_in_graph, num_nodes_in_graph + num_nodes_in_layer))
        else:
            last_layer = None
            new_nodes = list(range(num_nodes_in_layer))
        self.nxG.add_nodes_from(new_nodes, layer=i)

        # add edges the easy way
        if last_layer is not None:
            if i > 0:
                for m in last_layer:
                    for n in new_nodes:
                        if np.random.random() < p:
                            self.nxG.add_edge(m, n)

        y_values = self.calc_layered_y_values(num_nodes_in_layer)
        
        k = 0
        for j in new_nodes:
            pos[j] = [x_values[i], y_values[k]]
            k += 1

        return(new_nodes)

    def calc_layered_y_values(self, num_nodes_in_layer):
        # set position according to layer and number of nodes in layer
        y_values = np.arange(0, 1, 1 / num_nodes_in_layer) + 1 / num_nodes_in_layer * .5
        return y_values

    def add_extra_layer(self, new_layer_size, p):
        '''

        :param new_layer_size:
        :param p:
        :param pos:
        :param x_values:
        :return:
        '''

        last_layer = np.array(list(self.layers.values())).max()
        next_layer = last_layer + 1

        # should really re-calculate positions
        self.x_values = np.append(self.x_values, 1.0)

        #self.layer_sizes.append(new_layer_size)
        self.add_layer(next_layer, self.G.numberOfNodes(), new_layer_size, p, self.pos, self.x_values)

        #self.pos = pos
        #self.x_values = x_values
        self.G = nk.nxadapter.nx2nk(self.nxG)
        self.layers = nx.get_node_attributes(self.nxG, 'layer')
        #self.layer_sizes = layer_sizes

    def get_complete_edges_layered(self):
        '''
        return a list of all possible edges in a layered network
        '''
        edges = []

        uniq_layers = list(set(self.layers.values()))

        for i in uniq_layers[0:-1]:
            for start_node in [k for (k, v) in self.layers.items() if v == i]:
                for j in [k for (k, v) in self.layers.items() if v == i + 1]:
                    edges.append((start_node, j))

        self.complete_possble_edges_layered = edges

    def add_random_edge_layered(self):
        '''
        add one edge randomly in a layered network

        if these two nodes already have an edge, do not add it
        do not add self-edges (u - u)

        :return:
        '''

        # get a list of all possible edges
        if not hasattr(self, 'complete_possble_edges_layered'):
            self.get_complete_edges_layered()

        if(self.is_complete_graph_layered()):
            return(None)

        # subtract the set of existing edges from the complete graph to get edge candidates
        # choose one


        candidate_edges = list(set(self.complete_possble_edges_layered) - set(self.G.edges()))
        i = np.random.choice(len(candidate_edges), 1).item()
        edge_to_add = candidate_edges[i]

        u = edge_to_add[0]
        v = edge_to_add[1]

        self.G.addEdge(u, v)
        #self.nxG = nk.nxadapter.nk2nx(self.G)
        return ((u, v))

    def get_layered_edges_within_radius(self, radius=1):
        edges_within_radius = []

        N = self.G.numberOfNodes()
        for nodeA in range(N):

            layerA = self.layers[nodeA]
            for nodeB in range(nodeA, N):
                layerB = self.layers[nodeB]

                if (abs(layerA - layerB) <= radius) and (layerA != layerB):
                    edges_within_radius.append((nodeA, nodeB))
        return (edges_within_radius)

    def add_random_edge_layered_within_radius(self, radius = 1):
        '''
        add one edge randomly in a layered network

        if these two nodes already have an edge, do not add it
        do not add self-edges (u - u)

        :return:
        '''
        edges_within_radius = self.get_layered_edges_within_radius(radius)
        candidate_edges = list(set(edges_within_radius) - set(self.G.edges()))
        i = np.random.choice(len(candidate_edges), 1).item()
        edge_to_add = candidate_edges[i]

        u = edge_to_add[0]
        v = edge_to_add[1]

        self.G.addEdge(u, v)
        # self.nxG = nk.nxadapter.nk2nx(self.G)
        return ((u, v))


    def get_nodes_by_attribute(self, attribute, value):
        '''
        return a list of nodes having attribute = value

        Note, this may have been overwritten when converting nk graph to nx graph during draw() method
        '''
        l = dict(self.nxG[attribute == value])
        nodes = list(l.keys())
        return(nodes)

    def get_nodes_in_layer(self, layer_num):
        return([k for (k, v) in self.layers.items() if v == layer_num])

    def num_activated_in_layer(self, layer_num):
        activated_nodes = self.get_nodes_in_layer(layer_num)
        return(np.array(self.labels[activated_nodes]).sum())

    def num_activated_last_layer(self):
        last_layer = self.num_layers - 1
        return(self.num_activated_in_layer(last_layer))

    def fraction_activated_in_layer(self, layer_num):
        num_activated = self.num_activated_in_layer(layer_num)
        num_in_layer = len(self.get_nodes_in_layer(layer_num))
        fract_active = float(num_activated)/num_in_layer
        return(fract_active)

    def fraction_activated_last_layer(self):
        last_layer = self.num_layers - 1
        return(self.fraction_activated_in_layer(last_layer))

    def fraction_activated_subset_last_layer(self, num_sets, set_id):

        last_layer = self.num_layers - 1
        nodes_last_layer = self.get_nodes_in_layer(last_layer)
        num_nodes_last_layer = len(nodes_last_layer)

        subset_last_layer = nodes_last_layer[int(set_id * num_nodes_last_layer / num_sets):
                                             int((set_id + 1) * num_nodes_last_layer / num_sets)]

        num_activated = np.array(self.labels[subset_last_layer]).sum()
        fraction_activated = num_activated / len(subset_last_layer)

        return(fraction_activated)

    def get_nodes_by_distance_to_seed(self):
        '''
        for BFS distance to seed, return a list of nodes at each distance

        :return: list of arrays of nodes at each unique distance, array of unique distances
        '''

        nodes_at_dist = []

        self.bfs()
        dists_from_seed = self.bfs_dists

        unique_dists = np.unique(dists_from_seed)

        for dist in unique_dists:
            nodes_at_dist.append(np.where(dists_from_seed == dist)[0])

        return (nodes_at_dist, unique_dists)

    def fraction_activated_all_layers(self):

        fractions_activated = []
        unique_dists = self.get_layers_by_bfs_dist()

        for dist in unique_dists:
            fractions_activated.append(self.fraction_activated_in_layer(dist))

        return(unique_dists, fractions_activated)

    def get_layers_by_bfs_dist(self):
        '''
        create layers dictionary for any non-layered graph (Erdos-Renyi, etc.)
        Assumes seed is only 1 node

        :return: sets self.layers
        '''

        dist_nodes, unique_dists = self.get_nodes_by_distance_to_seed()

        # create layers dictionary
        layers = {}

        for i in range(len(unique_dists)):
            dist = unique_dists[i]
            for j in range(len(dist_nodes[i])):
                node = dist_nodes[i][j]
                layers[node] = dist

        self.layers = layers

        return(unique_dists)


    def is_complete_graph(self):
        complete_graph = self.G.numberOfEdges() >= self.N * (self.N - 1) / 2
        return(complete_graph)

    def is_complete_graph_layered(self):
        '''
        determine if this is a complete layered graph
        :return:
        '''
        if not hasattr(self, 'complete_possble_edges_layered'):
            self.get_complete_edges_layered()

        return(self.G.numberOfEdges() >= len(self.complete_possble_edges_layered))


    def add_edge(self, new_edge):
        '''
        add specified edge to member networkit graph G
        also convert resulting graph to networkx graph
        (for now, do not check if edge already in graph)

        :param new_edge:
        :return:
        '''

        u = new_edge[0]
        v = new_edge[1]
        if not self.has_edge(new_edge):
            self.G.addEdge(u, v)
        self.nxG = nk.nxadapter.nk2nx(self.G)


    def remove_edge(self, edge_to_remove):
        '''
        remove specified edge from member networkit graph G
        also convert resulting graph to networkx graph
        (for now, do not check if edge already in graph)

        :param edge_to_remove:
        :return:
        '''

        self.G.removeEdge(edge_to_remove[0], edge_to_remove[1])
        self.nxG = nk.nxadapter.nk2nx(self.G)

    def has_edge(self, edge):
        # check that aren't already neighbors

        u = edge[0]
        v = edge[1]
        already_neighbors = v in self.G.neighbors(u)

        return(already_neighbors)

    def initialize_available_edges(self):
        G = nx.complete_graph(self.G. numberOfNodes())
        self.available_edges = np.array(G.edges())
        del (G)

    # def initialize_available_edges_layered(self):
    #     G = nx.complete_graph(self.G.numberOfNodes())
    #     self.available_edges_layered = list(G.edges())
    #     del (G)

    def add_random_edge_fast(self):
        index = np.random.randint(len(self.available_edges))
        random_edge = self.available_edges[index]
        #del (self.available_edges[index])
        np.delete(self.available_edges, index)

        u = random_edge[0]
        v = random_edge[1]
        self.G.addEdge(u, v)

    def get_nodes_within_distance_ckdtree(self, radius, which_dim = 0):
        '''
        find all node pairs located within radius of each other
        :param radius:
        :param which_dim: 0, 1, or 'both', consistent with numpy axis
        :return:
        '''

        nodes = np.array(list(self.pos.keys()))
        locs = np.array(list(self.pos.values()))

        if which_dim == 'both':
            ckdtree = cKDTree(locs)
        else: # 1-dimension (either x or y)
            locs[:,1-which_dim] = 0  #cancel out the other dimension
            ckdtree = cKDTree(locs)

        pairs_within_radius = ckdtree.query_pairs(radius)

        return (pairs_within_radius)

    def get_available_edges_within_radius(self, radius, which_dim = 0):
        '''
        intersection of 'nearby' node pairs and available edges
        :param radius:
        :param which_dim: 0, 1, or 'both', consistent with numpy axis
        :return:
        '''
        '''
        

        :param radius:
        :return:
        '''

        nearby_nodes = self.get_nodes_within_distance_ckdtree(radius, which_dim)
        available_edges = set(tuple(row) for row in self.available_edges)
        nearby_available_edges = nearby_nodes.intersection(available_edges)

        return (nearby_available_edges)


    def add_random_edge_layered_within_radius_fast(self, radius, which_dim):
        '''
        add one edge randomly in a layered network

        :return:
        '''

        #which_dim = 0 # we only use the 1-D horizontal position to find the distance

        # the available edges within radius
        candidate_edges = list(self.get_available_edges_within_radius(radius,
                                                                      which_dim))
        num_candidates = len(candidate_edges)
        if num_candidates == 0:
            return (None, 0)

        i = np.random.choice(len(candidate_edges), 1).item()
        edge_to_add = candidate_edges[i]

        u = edge_to_add[0]
        v = edge_to_add[1]

        self.G.addEdge(u, v)

        #update available edges list
        new_available_edges = list(tuple(row) for row in self.available_edges)
        new_available_edges.remove(edge_to_add)
        self.available_edges = np.array(new_available_edges)

        # self.nxG = nk.nxadapter.nk2nx(self.G)
        return ((u, v), num_candidates)

    def add_random_edge_ER_within_radius_fast(self, radius, which_dim):
        '''
        add one edge randomly in a layered network

        :return:
        '''

        #which_dim = 'both' # we only use the 1-D horizontal position to find the distance

        # the available edges within radius
        candidate_edges = list(self.get_available_edges_within_radius(radius,
                                                                      which_dim))
        num_candidates = len(candidate_edges)
        if num_candidates == 0:
            return (None, 0)

        i = np.random.choice(len(candidate_edges), 1).item()
        edge_to_add = candidate_edges[i]

        u = edge_to_add[0]
        v = edge_to_add[1]

        self.G.addEdge(u, v)

        #update available edges list
        # self.available_edges_layered.remove(edge_to_add)

        # update available edges list
        new_available_edges = list(tuple(row) for row in self.available_edges)
        new_available_edges.remove(edge_to_add)
        self.available_edges = np.array(new_available_edges)

        # self.nxG = nk.nxadapter.nk2nx(self.G)
        return ((u, v), num_candidates)

    def add_random_edge(self):
        '''
        add a random edge between nodes
        if these two nodes already have an edge, do not add it
        do not add self-edges (u - u)

        :return: the new edge
        '''

        already_neighbors = False
        complete_graph = self.is_complete_graph()

        nodes = np.random.choice(self.G.numberOfNodes(), 2, replace=False)

        # check that aren't already neighbors
        u = nodes[0]
        v = nodes[1]
        already_neighbors = self.G.hasEdge(u,v) #v in self.G.neighbors(u)


        # if needed, keep seeking a valid edge to add
        while already_neighbors and not self.is_complete_graph():
            # choose two different nodes
            nodes = np.random.choice(self.G.numberOfNodes(), 2, replace=False)

            # check that aren't already neighbors
            u = nodes[0]
            v = nodes[1]
            already_neighbors = v in self.G.neighbors(u)

        self.G.addEdge(u, v)
        #self.nxG = nk.nxadapter.nk2nx(self.G)

        return((u,v))

    def remove_random_edge(self):
        '''
        remove one edge randomly from graph
        '''

        edge_list = self.G.edges()
        index = np.random.choice(range(len(edge_list)))
        edge_to_remove = edge_list[index]
        self.G.removeEdge(edge_to_remove[0], edge_to_remove[1])
        self.nxG = nk.nxadapter.nk2nx(self.G)
        return(edge_to_remove)

    def remove_random_edge_between_unlabeled_nodes(self, verbose = False):
        '''
        remove a random edge between two unlabeled nodes

        :return:
        '''

        unlabeled_ids = np.where(self.labels == False)[0]
        subG = nk.Graph.subgraphFromNodes(self.G, unlabeled_ids)

        unlabeled_edges = subG.edges()
        num_unlabeled_edges = subG.numberOfEdges()
        del(subG)

        # if no unlabeled edges, get out
        if(num_unlabeled_edges == 0):
            return(None)

        edge_index = np.random.randint(num_unlabeled_edges)
        edge_to_remove = unlabeled_edges[edge_index]

        self.G = self.G.removeEdge(edge_to_remove[0], edge_to_remove[1])
        self.nxG = nk.nxadapter.nk2nx(self.G)

        if verbose:
            print('removing', edge_to_remove)

        return(edge_to_remove)




    def make_complete_graph(self,
                            N):

        self.nxG = nx.complete_graph(N)
        self.G = nk.nxadapter.nx2nk(self.nxG)

    def set_phi_values(self,
                       phis,
                       phi_distribution=np.random.uniform):
        '''
        '''
        if phis is None or phis == 'rand':  # set random phi values
            self.set_random_phi_values(phi_distribution)

        elif type(phis) is np.ndarray or type(phis) is list:  # phis is np array or list
            self.phi_vals = np.array(phis)  # cast list or array to array

        elif type(phis) is float:
            self.set_const_phi_values(phis)

    def set_seeds(self, seeds, num_seeds):
        '''
        either set random seeds or seeds passed as parameters
        '''
        if seeds is None:
            self.set_random_seeds(num_seeds)
        else:
            self.set_seed_nodes(seeds)

        self.count_unlabeled()

        #self.synchronous_find_vulnerable_nodes()
        #self.synchronous_find_vulnerable_cluster()

    def set_unvisited(self):
        '''
            set all nodes to unvisited
        :return:
        '''

        self.visited = np.zeros(self.N).astype(bool)

    # phi functions

    def set_random_phi_values(self,
                              phi_distribution=np.random.uniform):
        '''
        '''
        self.phi_vals = phi_distribution(size=self.N)

    def set_const_phi_values(self, const_phi):
        '''
        '''
        self.phi_vals = np.full(self.N, const_phi)

    # label functions

    def reset_labels(self):
        '''
        set all labels to False
        '''

        self.labels = np.zeros(self.N).astype(bool)
        self.seed_ids = None

    def reset_all_labels_except_seeds(self):

        self.labels = np.zeros(self.N).astype(bool)

        np.put(self.labels, self.seed_ids, True)

        #self.synchronous_find_vulnerable_nodes()

    def set_random_seeds(self, num_seeds):
        '''
        '''
        self.reset_labels()

        # if self.network_generator.__name__ == 'networkx.algorithms.bipartite':
        #     self.seed_ids = np.random.choice(list(range(self.N - 1)), num_seeds, replace=False)
        #
        # else:
        self.seed_ids = np.random.choice(list(range(self.N)),
                                         num_seeds,
                                         replace=False)

        np.put(self.labels, self.seed_ids, True)

    def set_seed_nodes(self, seed_id_list):
        '''
        sets a list of seeds
        '''

        self.reset_labels()

        self.seed_ids = seed_id_list

        np.put(self.labels, self.seed_ids, True)

    # plotting

    def plot_phi_hist(self):
        '''
        '''

        plt.hist(self.phi_vals, bins=50);
        plt.title('phi, mean: ' + str(self.phi_vals.mean()));

    def draw_graph(self,
                   label_seeds=True,
                   draw_phi_vals=True,
                   color_vulnerable_cluster=True,
                   graph_to_draw='nxG',
                   **kwargs):
        '''
        '''

        self.nxG = nk.nxadapter.nk2nx(self.G)
        self.get_bfs_tree()

        if graph_to_draw == 'nxG':
            G_to_draw = self.nxG
        elif graph_to_draw == 'bfs_tree':
            G_to_draw = self.bfs_tree

        if (self.pos is None):
            self.pos = nx.spring_layout(self.nxG)

        fig = plt.figure(**kwargs)
        ax = fig.gca()
        ax.grid(False)

        if graph_to_draw == 'nxG':

            font_colors = 'dark_red'

            if label_seeds:

                # draw the seed node ids
                keys = np.array(range(self.N))
                values = np.array([''] * self.N)
                values[self.labels == True] = keys[self.labels == True]
                node_labels = dict(zip(keys, values))

                # set node colors based on labels
                node_colors = self.labels.astype(object)

                node_colors[self.labels == False] = 'white'

                if color_vulnerable_cluster and len(self.vulnerable_cluster > 0):
                    node_colors[self.vulnerable_cluster] = 'green'

                node_colors[self.labels == True] = 'red'

                # node_colors = self.phi_vals

                # set node border colors based on labels
                #             edgecolors = self.labels.astype(object)
                #             edgecolors[self.labels == True] = 'lightgrey'
                #             edgecolors[self.labels == False] = 'grey'

                # reversed gray colormap
                cmap = cm.gray_r
                norm = Normalize(vmin=0.1, vmax=1)
                edgecolors = cmap(norm(self.phi_vals))

            #             edgecolors = 'grey'

            else:
                # draw the seed node ids
                keys = np.array(range(self.N))
                values = np.array([''] * self.N)
                values[self.labels == True] = keys[self.labels == True]
                node_labels = dict(zip(keys, values))

                # set node colors based on labels
                node_colors = self.labels.astype(object)
                node_colors[self.labels == True] = 'red'
                node_colors[self.labels == False] = 'white'

                # set node border colors based on labels
                #             edgecolors = self.labels.astype(object)
                #             edgecolors[self.labels == True] = 'grey'
                #             edgecolors[self.labels == False] = 'grey'

                edgecolors = 'grey'

        else:
            node_colors = 'white'
            edgecolors = 'grey'
            node_labels = None
            font_colors = 'black'

            # draw the seed node ids
            keys = np.array(range(self.N))
            values = np.array([''] * self.N)
            values[self.labels == True] = keys[self.labels == True]
            node_labels = dict(zip(keys, values))

            # set node colors based on labels
        #             node_colors = self.labels.astype(object)
        #             node_colors[self.labels == False] = 'white'
        #             node_colors[self.labels == True] = 'red'

        # set node sizes based on labels
        # node_sizes = 3000/self.N
        node_sizes = np.full(self.N, 2000 / self.N)
        #         node_sizes[self.labels == True] = 8000/self.N

        # node_sizes = self.phi_vals * 200

        colors = range(20)
        cmap = cm.gray
        # vmin = min(colors)
        # vmax = max(colors)
        vmin = 0
        vmax = 1.0
        #         nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
        #                    with_labels=False, vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []

        # some default plotting options to pass to networkx
        options = {
            'node_color': node_colors,
            'cmap': cmap,
            # 'vmin'      : vmin,
            # 'vmax'      : vmax,
            'edge_color': 'lightgrey',
            'node_size': node_sizes,
            'edge_size': .1,
            'edgecolors': edgecolors,
            'alpha': .7,
            'labels': node_labels,
            'font_color': font_colors,
            'font_size': 8,
            'font_weight': 'bold',
            'label': 'nodes',
            'pos': self.pos,
            'ax': ax}

        options.update(kwargs)

        # uses networkx.drawing.nx_pylab.draw_networkx
        # nk.viztasks.drawGraph(self.G, **options)

        nx.draw_networkx(G_to_draw, **options)

        if draw_phi_vals:

            # offset the labels
            pos_higher = {}
            x_off = 0
            y_off = .2 / np.sqrt(self.N)  # offset on the y axis
            for k, v in self.pos.items():
                pos_higher[k] = (v[0] + x_off, v[1] + y_off)

            # label the phi vals
            keys = np.array(range(self.N))
            values = np.round(self.phi_vals, 2)
            phi_labels = dict(zip(keys, values))

            nx.draw_networkx_labels(G=G_to_draw,
                                    labels=phi_labels,
                                    pos=pos_higher,
                                    font_color='grey',
                                    font_size=7)

        #         #legend
        #         cbar = plt.colorbar(sm, ticks=range(0,1, .1), pad=0.2)
        #         #cbar.ax.set_yticklabels(np.arange(0, 1,.1))
        #         cbar.set_label('$\phi$ value', rotation=270, labelpad= 15)

        #         leg = plt.legend();

        plt.axis('off')
        plt.tight_layout()

        return (fig)

    def draw_graph2(self, pos=None, draw_phi_vals=True, node_size = 50,
                    labels = True, title = '', node_label_color = 'green',
                    phi_label_color = 'grey'):

        # convert nk graph to nx graph
        self.nxG = nk.nxadapter.nk2nx(self.G)

        if (self.pos is None):
            self.pos = nx.spring_layout(self.nxG)

        fig = plt.figure()
        ax = fig.gca()
        ax.grid(False)

        # set node colors based on labels
        node_colors = self.labels.astype(object)
        node_colors[self.labels == False] = 'white'
        node_colors[self.labels == True] = 'black'
        edgecolor = 'lightgrey'

        nx.draw(self.nxG,
                self.pos,
                edge_color=edgecolor,
                node_color=node_colors,
                edgecolors='grey',
                node_size = node_size)

        if labels:
            # offset the labels
            pos_lower = {}
            x_off = 0
            y_off = .17 / np.sqrt(self.N)  # offset on the y axis
            for k, v in self.pos.items():
                pos_lower[k] = (v[0] + x_off, v[1] - y_off)

            # keys = np.array(range(self.N))
            # values = np.array(range(self.N))
            # my_labels = []
            # for value in values:
            #     my_labels.append('ϕ= ' + str(value))
            # my_labels = np.array(my_labels)

            #phi_labels = dict(zip(keys, my_labels))

            nx.draw_networkx_labels(G=self.nxG,
                                    #labels=phi_labels,
                                    pos=pos_lower,
                                    font_color=node_label_color,
                                    font_size=10)
            # nx.draw_networkx_labels(self.nxG, self.pos, font_color = node_label_color)

        if draw_phi_vals:
            # offset the labels
            pos_higher = {}
            x_off = 0
            y_off = .1 / np.sqrt(self.N)  # offset on the y axis
            for k, v in self.pos.items():
                pos_higher[k] = (v[0] + x_off, v[1] + y_off)

            # label the phi vals
            keys = np.array(range(self.N))
            values = np.round(self.phi_vals, 3)
            my_labels = []
            for value in values:
                my_labels.append('ϕ= '+ str(value))
            my_labels = np.array(my_labels)

            phi_labels = dict(zip(keys, my_labels))

            nx.draw_networkx_labels(G=self.nxG,
                                    labels=phi_labels,
                                    pos=pos_higher,
                                    font_color=phi_label_color,
                                    font_size=10)

        plt.title(title)
        plt.tight_layout(pad=0, w_pad=10, h_pad=10)

    def save_tifz_format(self, filename):
        '''
        save the network for latex

        credit to:
        2018 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
        http://www.ibi.ethz.ch
        '''

        # Network and attributes
        net = self.nxG
        layout = self.pos

        # Network dicts
        # -------------
        #     color_dict = {"m": "blue", "f": "red"}
        #     shape_dict = {"m": "circle", "f": "rectangle"}
        #     style_dict = {"m": "{shading=ball}", "f": None}
        #     layout = {'a': (4.3191, -3.5352), 'b': (0.5292, -0.5292),
        #               'c': (8.6559, -3.8008), 'd': (12.4117, -7.5239),
        #               'e': (12.7, -1.7069), 'f': (6.0022, -9.0323),
        #               'g': (9.7608, -12.7)}

        # Visual style dict
        # -----------------
        visual_style = {}

        # node styles
        # -----------
        visual_style['vertex_size'] = 1
        visual_style[
            'vertex_color'] = 'white'  # {n:color_dict[g] for n,g in nx.get_node_attributes(net,'gender').items()}
        visual_style['vertex_opacity'] = .7
        visual_style['vertex_label'] = nx.get_node_attributes(net, 'name')
        visual_style['vertex_label_position'] = 'below'
        visual_style['vertex_label_distance'] = 15
        visual_style['vertex_label_color'] = 'gray'
        visual_style['vertex_label_size'] = 3
        visual_style['vertex_shape'] = {n: shape_dict[g] for n, g in nx.get_node_attributes(net, 'gender').items()}
        visual_style['vertex_style'] = {n: style_dict[g] for n, g in nx.get_node_attributes(net, 'gender').items()}
        visual_style['vertex_label_off'] = {'e': True}
        visual_style['vertex_math_mode'] = {'a': True}
        visual_style['vertex_label_as_id'] = {'f': True}
        visual_style['vertex_pseudo'] = {'d': True}

        # edge styles
        # -----------
        visual_style[
            'edge_width'] = .1  # {e:.3 + .3 * int(f) for e,f in nx.get_edge_attributes(net,'is_formal').items()}
        visual_style['edge_color'] = 'gray'
        visual_style['edge_opacity'] = .8
        visual_style['edge_curved'] = 0
        # visual_style['edge_label'] = {e:e[0]+e[1] for e in net.edges}
        visual_style['edge_label_position'] = 'above'
        visual_style['edge_label_distance'] = .6
        visual_style['edge_label_color'] = 'gray'
        visual_style['edge_label_size'] = {('a', 'c'): 5}
        visual_style['edge_style'] = 'solid'  # 'dashed'
        visual_style['edge_arrow_size'] = .2
        visual_style['edge_arrow_width'] = .2
        visual_style['edge_loop_size'] = 15
        visual_style['edge_loop_position'] = 90
        visual_style['edge_loop_shape'] = 45
        visual_style['edge_directed'] = {('a', 'b'): True, ('a', 'c'): True,
                                         ('c', 'd'): False, ('d', 'e'): True,
                                         ('e', 'c'): True, ('c', 'f'): False,
                                         ('f', 'a'): True, ('f', 'g'): True,
                                         ('g', 'g'): True}
        # visual_style['edge_label'][('a','c')] = '\\frac{\\alpha}{\\beta}'
        visual_style['edge_math_mode'] = {('a', 'c'): True}
        visual_style['edge_not_in_bg'] = {('f', 'a'): True}

        # general options
        # ---------------
        visual_style['unit'] = 'mm'
        visual_style['layout'] = layout
        visual_style["margin"] = {'top': 5, 'bottom': 8, 'left': 5, 'right': 5}
        visual_style["canvas"] = (100, 60)
        visual_style['keep_aspect_ratio'] = False

        # Create a latex file
        # plot(net, filename, **visual_style)

        # Create a node and edge list used by tikz-network

        # plot(net,'network.csv',**visual_style)

        # Create pdf figure of the network
        # ONLY POSSIBLE IF tikz-network IS INSTALLED
        # AND (for Widows OS) COMPLETER HAS TO BE SET RIGHT

        # plot(net,'network.pdf',**visual_style)

        # Create temp pdf and show the output
        # ONLY POSSIBLE IF tikz-network IS INSTALLED
        # AND (for Widows OS) COMPLETER HAS TO BE SET RIGHT

        # plot(net,**visual_style)

    # cascade functions

    def get_unlabeled_ids(self):
        '''
        '''

        unlabeled_ids = np.arange(self.N)[self.labels == False]

        return (unlabeled_ids)

    def get_labeled_ids(self):
        '''
        '''

        labeled_ids = np.arange(self.N)[self.labels == True]

        return (labeled_ids)


    def number_labeled_neighbors(self, neighbor_ids):
        '''
        return the number of neighbor_ids that are labeled
        '''

        num_neighbors_labeled = (self.labels[neighbor_ids] == True).sum()

        return (num_neighbors_labeled)

    def fraction_neighbors_labeled(self, node_id):  # num_labeled, num_neighbors):
        '''
        returns the fraction of node_id's neighbors that are labeled

        if node_id has degree 0, returns -1
        '''

        # if directed, only look at in-neighbors
        if self.directed:
            neighbor_ids = list(self.nxG.predecessors(node_id))
        else:
            neighbor_ids = self.G.neighbors(node_id)

        degree = len(neighbor_ids)

        if degree > 0:
            num_labeled = self.number_labeled_neighbors(neighbor_ids)
            fract_labeled = num_labeled / degree
            if fract_labeled >= 0:
                return (fract_labeled)
        if degree == 0:
            return 0

        return (None)

    def cascade_rule_function(self, value, threshold):
        '''
        checks whether value, threshold satisfy the cascade rule function
        '''

        return (self.rule(value, threshold))

    def count_unlabeled(self):
        '''
        keep track of number of unlabelled nodes
        '''

        unlabeled_nodes = self.get_unlabeled_ids()

        self.num_unlabeled = len(unlabeled_nodes)

        return (self.num_unlabeled)

    def synchronous_find_vulnerable_nodes(self):
        '''
        These are the 'empirical' one-step vulnerable cells, calculated based on current labeled nodes.
        '''

        unlabeled_nodes = self.get_unlabeled_ids()
        phis = self.phi_vals[unlabeled_nodes]

        vulnerable_nodes = []
        for node in unlabeled_nodes:
            rho = self.fraction_neighbors_labeled(node)

            if rho is not None:
                phi = self.phi_vals[node]

                if self.cascade_rule_function(rho, phi):
                    vulnerable_nodes.append(node)

        self.vulnerable_nodes = np.array(vulnerable_nodes)

    def synchronous_find_vulnerable_cluster(self):
        '''
        These are the 'vulnerable cluster' from 1/k >= phi (see Watts Global Cascade paper)

        NOTE: This is different from the above vulnerable nodes.
        '''

        unlabeled_nodes = self.get_unlabeled_ids()
        phis = self.phi_vals[unlabeled_nodes]

        vulnerable_cluster = []
        for node in unlabeled_nodes:

            k = self.G.degree(node)

            if k != 0:
                rho = 1.0 / k
                phi = self.phi_vals[node]

                if self.cascade_rule_function(rho, phi):
                    #print('node ', node, k, rho, phi, rho >= phi)
                    vulnerable_cluster.append(node)

        self.vulnerable_cluster = np.array(vulnerable_cluster)

    # def synchronous_cascade(self, stop_time=5):
    #
    #     unlabeled_nodes = self.get_unlabeled_ids()
    #
    #     # update our number unlabeled
    #     self.count_unlabeled()
    #
    #     # check that there are unlabeled nodes
    #     if (self.num_unlabeled == 0):
    #         return
    #
    #     phis = self.phi_vals[unlabeled_nodes]
    #
    #     vulnerable_nodes = []
    #
    #     i = 0
    #     last_change = 0
    #
    #     while (i - last_change) <= stop_time and self.num_unlabeled > 0:
    #
    #         for node in unlabeled_nodes:
    #             rho = self.fraction_neighbors_labeled(node)
    #
    #             if rho is not None:
    #                 phi = self.phi_vals[node]
    #
    #                 if self.cascade_rule_function(rho, phi):
    #                     self.labels[node] = True
    #                     last_change = i
    #
    #         i += 1


    def synchronous_cascade(self):
        '''
        for each iteration, check all nodes.  stop when there is no change

        :return:
        '''

        # update our number unlabeled
        self.count_unlabeled()

        change = True

        while change and self.num_unlabeled > 0:
            change = False

            unlabeled_nodes = self.get_unlabeled_ids()

            # update our number unlabeled
            self.count_unlabeled()


            for node in unlabeled_nodes:
                rho = self.fraction_neighbors_labeled(node)

                if rho is not None:
                    phi = self.phi_vals[node]

                    if self.cascade_rule_function(rho, phi):
                        self.labels[node] = True
                        change = True



    def asynchronous_cascade_step(self, node_to_check = None):
        '''
        run one step of an asynchronous cascade
        '''

        unlabeled_nodes = self.get_unlabeled_ids()

        # update our number unlabeled
        self.count_unlabeled()

        # check that there are unlabeled nodes
        if (self.num_unlabeled == 0):
            return

        # choose 1 node

        if node_to_check is None:
            node_to_check = np.random.choice(unlabeled_nodes, size=1)

        phi = self.phi_vals[node_to_check]

        rho = self.fraction_neighbors_labeled(node_to_check)

        if rho is not None and self.cascade_rule_function(rho, phi):
            self.labels[node_to_check] = True

            return (True)

        return (False)

    def run_asynchronous_cascade(self, stop_time=None, verbose=False):
        '''
        '''
        i = 0
        last_change = 0
        self.count_unlabeled()

        if stop_time is None:
            stop_time = 5 * self.N

        while (i - last_change) < stop_time and self.num_unlabeled > 0:
            changed = self.asynchronous_cascade_step()

            if changed:
                last_change = i
                changed = False
                self.count_unlabeled()

                if verbose:
                    if i % 100 == 0:
                        print(i)

            i += 1

    def set_visited_node(self, node_to_check):
        self.visited[node_to_check] = True

    def get_unvisited_ids(self):
        '''
        speeds up cascade
        :return:
        '''

        unvisited_ids = np.arange(self.N)[self.visited == False]

        return(unvisited_ids)




    def asynchronous_cascade_step2(self, node_to_check = None):
        '''
        run one step of an asynchronous cascade on one node
        '''

        unlabeled_nodes = self.get_unlabeled_ids()

        # update our number unlabeled
        self.count_unlabeled()

        # check that there are unlabeled nodes
        if (self.num_unlabeled == 0):
            return

        # choose 1 node

        if node_to_check is None:
            node_to_check = np.random.choice(unlabeled_nodes, size=1)[0]

        phi = self.phi_vals[node_to_check]

        rho = self.fraction_neighbors_labeled(node_to_check)

        self.set_visited_node(node_to_check)

        # return true if the labelling rule is satisfied
        if rho is not None and self.cascade_rule_function(rho, phi):
            self.labels[node_to_check] = True

            return (True)

        return (False)



    def run_asynchronous_cascade2(self, verbose = False):
        '''
        better asynchronous cascade:
        keep track of which nodes have been examined
        and stop when all unlabeled nodes examined
        :param verbose:
        :return:
        '''

        self.nxG = nk.nxadapter.nk2nx(self.G)

        all_visited = False
        self.count_unlabeled()
        self.set_unvisited()

        # cascade until all unlabelled have been visited once
        while (not all_visited and self.num_unlabeled > 0):

            # run cascade and determine if some node became labelled
            changed = self.asynchronous_cascade_step2()

            # if one node has become labelled,
            # reset all to unvisited and count the unlabelled nodes
            if changed:
                self.set_unvisited()
                self.count_unlabeled()

            else:
                visited_ids = np.arange(self.N)[self.visited == True]
                num_visited = len(visited_ids)
                all_visited = self.num_unlabeled == num_visited

            if verbose:
                print(all_visited)

    def efficient_cascade_asynchronous_step(self, node_to_check):
        '''
        run one step of an asynchronous cascade on one node
        '''

        phi = self.phi_vals[node_to_check]
        rho = self.fraction_neighbors_labeled(node_to_check)

        # return true if the labelling rule is satisfied
        if rho is not None and self.cascade_rule_function(rho, phi):
            self.labels[node_to_check] = True

            return (True)

        return (False)

    def efficient_cascade_asynchronous(self, verbose = False):
        '''
        Run cascade on all unlabelled nodes until no change
        :param verbose:
        :return:
        '''
        changed = True

        while(changed == True):

            changed = False
            unlabeled_nodes = self.get_unlabeled_ids()

            changed_this_node = np.zeros((len(unlabeled_nodes)), dtype=bool)

            # randomize the order
            random_unlabeled = np.random.choice(unlabeled_nodes, len(unlabeled_nodes), replace=False)
            num_unlabeled = len(random_unlabeled)

            for i in range(num_unlabeled):

                if self.efficient_cascade_asynchronous_step(random_unlabeled[i]):
                    changed_this_node[i] = True

            changed = np.any(changed_this_node)





    def asynchronous_cascade_step3(self):
        '''
        cascade: searching only unvisited nodes - faster

        set all unlabeled nodes to unvisited

        while len unvisited > 0:
            cascade step
            if a node become labeled
            set all unlabeled nodes to unvisited

        '''

        pass


    def cascade_size(self):
        '''
        '''

        return ((self.labels == True).sum())

    def cascade_fraction(self):
        '''
        '''

        casc_size = self.cascade_size()

        return casc_size / self.N

    def get_seed_component(self, seedID = 0):
        '''
        get list of nodes in seed's connected component
        :return:
        '''
        cc = nk.components.ConnectedComponents(self.G).run()
        seed_component_id = cc.componentOfNode(seedID)
        seed_component = cc.getComponents()[seed_component_id]

        return(seed_component)


    def bfs(self):
        '''
        run a breadth first search from the seed node

        **NOTE, at the moment, assumes one seed**
        '''

        self.bfs_obj = nk.distance.BFS(self.G, self.seed_ids[0])
        self.bfs_obj.run()
        dists = self.bfs_obj.getDistances()
        self.bfs_dists = np.array(dists)

    def get_bfs_tree(self):
        '''
        use networkx to get the bfs tree from the seed

        **NOTE, at the moment, assumes one seed**
        '''

        self.bfs_tree = nx.bfs_tree(self.nxG, self.seed_ids[0])

    def mean_degree(self):
        '''
        calculate z, the mean degree
        :return: z
        '''

        result = sum(dict(self.nxG.degree()).values()) / self.nxG.number_of_nodes()

        return(result)

    def set_label_adj_sparse(self):
        '''
        set the labelling adjacency matrix

        from the sparse adjacency matrix
        '''


        self.label_adj = self.adj.multiply(self.labels)


    def get_num_neighbors_labelled(self):
        return(np.asarray(self.label_adj.sum(axis=1)).flatten())

    def get_num_neighbors(self):
        return(np.asarray(self.adj.sum(axis=1)).flatten())

    def fraction_neighbors_labelled(self):
        num_neighbors_labelled = self.get_num_neighbors_labelled()
        num_neighbors = self.get_num_neighbors()
        nu = num_neighbors_labelled/num_neighbors
        return(nu)

    def cascade_vect(self):
        '''
        '''

        nu = fraction_neighbors_labelled(self.adj, self.label_adj)
        result = self.rule(nu, self.phi_vals)  # .astype(int)

        return (result)

    def adjust_labels(self, casc):
        '''
        adjust labels according to cascade results
        '''
        elts_to_one = np.where(casc)[0]
        new_labels = self.labels.copy()
        new_labels[elts_to_one] = True  # 1
        return (new_labels)

    def cascade_vectorized(self, verbose=False):

        num_seeds = len(self.seed_ids)
        num_changed = num_seeds

        while num_changed > 0:

            adj = nk.algebraic.adjacencyMatrix(self.G)

            set_label_adj_sparse()

            phi_vector = cascadeNet.phi_vals

            casc = cascade_vect()

            new_labels = adjust_labels(casc)
            num_changed = np.subtract(new_labels.astype(int), self.labels.astype(int)).sum()

            cascadeNet.labels = new_labels

            if verbose:
                print(num_changed)

        return (cascadeNet)





def run_many_synchronous_cascades(num_iterations=10,
                                   N=10,
                                   p=.5,
                                   num_seeds=1,
                                   phi_values=None,
                                   biggest_cascade_threshold=0.5):
    cascade_fractions = []

    biggest_cascade_fract = -1
    biggest_cascade_net = None

    for i in range(num_iterations):

        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values)

        cascadeNet.synchronous_cascade()

        casc_fraction = cascadeNet.cascade_fraction()

        # keep the maximum cascade greater than the threshold
        if casc_fraction > biggest_cascade_fract and casc_fraction > biggest_cascade_threshold:
            biggest_cascade_net = copy.deepcopy(cascadeNet)

        cascade_fractions.append(casc_fraction)

        cascadeNet.synchronous_find_vulnerable_nodes()

    #         print(cascadeNet.get_unlabeled_ids())

    return (cascade_fractions, biggest_cascade_net)


def run_parallel_synchronous_cascades(num_iterations=10,
                                       N=10,
                                       p=.5,
                                       num_seeds=1,
                                       phi_values=None,
                                       biggest_cascade_threshold=0.5):
    cascade_fractions = []

    biggest_cascade_fract = -1
    biggest_cascade_net = None

    for i in range(num_iterations):

        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values)

        cascadeNet.synchronous_cascade()

        casc_fraction = cascadeNet.cascade_fraction()

        # # keep the maximum cascade greater than the threshold
        # if casc_fraction > biggest_cascade_fract and casc_fraction > biggest_cascade_threshold:
        #     biggest_cascade_net = copy.deepcopy(cascadeNet)

        cascade_fractions.append(casc_fraction)

        #cascadeNet.synchronous_find_vulnerable_nodes()

    #         print(cascadeNet.get_unlabeled_ids())

    return (cascade_fractions, biggest_cascade_net)


def run_many_asynchronous_cascades(num_iterations=10,
                                   N=10,
                                   p=.5,
                                   num_seeds=1,
                                   phi_values=None,
                                   biggest_cascade_threshold=0.5):
    cascade_fractions = []

    biggest_cascade_fract = -1
    biggest_cascade_net = None

    for i in range(num_iterations):

        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values)

        cascadeNet.run_asynchronous_cascade()

        casc_fraction = cascadeNet.cascade_fraction()

        # keep the maximum cascade greater than the threshold
        if casc_fraction > biggest_cascade_fract and casc_fraction > biggest_cascade_threshold:
            biggest_cascade_net = copy.deepcopy(cascadeNet)

        cascade_fractions.append(casc_fraction)

        cascadeNet.synchronous_find_vulnerable_nodes()

    #         print(cascadeNet.get_unlabeled_ids())

    return (cascade_fractions, biggest_cascade_net)


def sample_many_cascades_by_p_val(p_values=np.arange(0.1, 0.7, 0.1),
                                  num_iterations=100,
                                  N=20,
                                  num_seeds=1,
                                  phi_values=None,
                                  biggest_cascade_threshold=.001):
    '''
    get a sample of cascade sizes for each p value

    also, return the maximum cascade network

    '''

    # each row is a p-value, each col is a cascade size for one iteration
    fracts_by_ps = np.zeros([len(p_values), num_iterations])

    max_cascade_size = -1
    max_cascade_p_val = -1
    biggest_cascade_size = -1
    for i in range(len(p_values)):

        p_val = p_values[i]

        cascade_fracts, biggest_cascade_net = run_many_asynchronous_cascades(num_iterations=num_iterations,
                                                                             N=N,
                                                                             p=p_val,
                                                                             num_seeds=num_seeds,
                                                                             phi_values=phi_values,
                                                                             biggest_cascade_threshold=biggest_cascade_threshold)

        if biggest_cascade_net is not None:
            biggest_cascade_size = biggest_cascade_net.cascade_size()

        if biggest_cascade_size > max_cascade_size:
            max_cascade_size = biggest_cascade_size
            max_cascade_net = biggest_cascade_net
            max_cascade_p_val = p_val

        fracts_by_ps[i, :] = np.array(cascade_fracts)

    return (fracts_by_ps, max_cascade_net, max_cascade_p_val)


def mean_cascade_fract_by_pval(fracts_by_ps):
    mean_cascade_fract = fracts_by_ps.mean(axis=1)

    return (mean_cascade_fract)


def max_cascade_fract_by_pval(fracts_by_ps):
    max_cascade_fract = fracts_by_ps.max(axis=1)

    return (max_cascade_fract)


def create_run_synchronous_cascade(N=1000,
                                   p=.5,
                                   num_seeds=1,
                                   phi_values=.5):
    '''
    function for creating and running a **synchronous** cascade
    '''

    cascadeNet = GlobalCascadeNetwork(N=N,
                                      p=p,
                                      num_seeds=num_seeds,
                                      phi_values=phi_values)

    cascadeNet.synchronous_cascade()

    return (cascadeNet.cascade_fraction())


def parallelize_synchronous_cascade(num_iterations=100,
                                    N=1000,
                                    p=.5,
                                    num_seeds=1,
                                    phi_values=.5):
    '''
    function to use joblib for parallelism

    I DON'T TRUST THIS, GIVES DIFFERENT RESULTS FROM SERIAL METHOD!!
    '''
    result = Parallel(n_jobs=-1)(delayed(create_run_synchronous_cascade)(N=N,
                                                                         p=p,
                                                                         num_seeds=num_seeds,
                                                                         phi_values=phi_values)
                                 for i in range(num_iterations))

    return (result)


def create_run_asynchronous_cascade(N=1000,
                                    p=.5,
                                    num_seeds=1,
                                    phi_values=.5,
                                    rule = operator.ge,
                                    stop_time=None,
                                    cascadeNet = None):
    '''
    function for creating and running an **asynchronous** cascade
    '''

    if cascadeNet is None:
        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values,
                                          rule = rule)

    cascadeNet.run_asynchronous_cascade(stop_time=stop_time)

    return (cascadeNet.cascade_fraction())


def parallelize_asynchronous_cascade(num_iterations=100,
                                     N=1000,
                                     p=.5,
                                     num_seeds=1,
                                     phi_values=.5,
                                     rule = operator.ge,
                                     use_same_graph = False):
    '''
    function to use joblib for parallel running of many cascades

    if use_same_graph == True, create one graph and run it many times

    PROBLEM: CAN'T PICKLE NETWORKIT GRAPH FOR PARALLELIZATION
    SOLUTION: EXTRACT NETWORKX GRAPH, RE-CREATE NETWORKIT GRAPH
    '''

    cascadeNet = None

    if use_same_graph:
        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values,
                                          rule=rule)

    result = Parallel(n_jobs=-1)(delayed(create_run_asynchronous_cascade)(N=N,
                                                                         p=p,
                                                                         num_seeds=num_seeds,
                                                                         phi_values=phi_values,
                                                                         rule = rule,
                                                                         cascadeNet = cascadeNet)
                                 for i in range(num_iterations))

    if use_same_graph:
        return(result, cascadeNet)

    return (result)




def create_run_efficient_cascade(N=1000,
                                 p=.5,
                                 num_seeds=1,
                                 phi_values=.5,
                                 rule = operator.ge,
                                 cascadeNet = None):
    '''
    function for creating and running an **asynchronous** cascade
    '''

    if cascadeNet is None:
        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values,
                                          rule = rule)

    cascadeNet.efficient_cascade_asynchronous()

    return (cascadeNet.cascade_fraction())


def parallelize_efficient_cascade(num_iterations=100,
                                  N=1000,
                                  p=.5,
                                  num_seeds=1,
                                  phi_values=.5,
                                  rule = operator.ge,
                                  use_same_graph = False):
    '''
    function to use joblib for parallel running of many cascades

    if use_same_graph == True, create one graph and run it many times

    PROBLEM: CAN'T PICKLE NETWORKIT GRAPH FOR PARALLELIZATION
    SOLUTION: EXTRACT NETWORKX GRAPH, RE-CREATE NETWORKIT GRAPH
    '''

    cascadeNet = None

    if use_same_graph:
        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p=p,
                                          num_seeds=num_seeds,
                                          phi_values=phi_values,
                                          rule=rule)

    result = Parallel(n_jobs=-1)(delayed(create_run_efficient_cascade)(N=N,
                                                                       p=p,
                                                                       num_seeds=num_seeds,
                                                                       phi_values=phi_values,
                                                                       rule = rule,
                                                                       cascadeNet = cascadeNet)
                                 for i in range(num_iterations))

    if use_same_graph:
        return(result, cascadeNet)

    return (result)












def get_vulnerable_subgraph(cascadeNet):
    '''
    return a subgraph containing all nodes that are 'vulnerable'
    '''

    cascadeNet.synchronous_find_vulnerable_cluster()
    vulnerable_nodes = cascadeNet.vulnerable_cluster
    vulnerable_subgraph = cascadeNet.G.subgraphFromNodes(vulnerable_nodes)
    vulnerable_nxG = nk.nxadapter.nk2nx(vulnerable_subgraph)
    
    return(vulnerable_subgraph, vulnerable_nxG)

def get_vulnerable_component_sizes_from_subgraph(vulnerable_subgraph):
    '''
    return a list of vulnerable connected component sizes
    '''

    nkcc = nk.components.ConnectedComponents(vulnerable_subgraph)
    nkcc.run()
    component_sizes_dict = nkcc.getComponentSizes()
    component_sizes = list(component_sizes_dict.values())
    
    return(component_sizes)

def get_vulnerable_component_sizes(cascadeNet):
    vulnerable_subgraph, vulnerable_nxG = get_vulnerable_subgraph(cascadeNet)
    component_sizes = get_vulnerable_component_sizes_from_subgraph(vulnerable_subgraph)
    return(component_sizes)

def test_cluster_cascade_sizes(N = 500,
                               phi = .1,
                               num_seeds = 1,
                               min_z = 1,
                               max_z = 10,
                               num_z_steps = 100,
                               verbose = False):
    '''
    get vulnerable cluster, cascade size statistics across zvals

    foreach z value:
        create a linear threshold model
        get its cluster size statistics
        get its vulnerable cluster size statistics
        cascade size statistics

    zvals, 
    component_sizes_list, 
    mean_vuln_sizes, 
    median_vuln_sizes, 
    std_vulnerable_cluster_sizes, 
    num_components_list, 
    largest_component_list, 
    cascade_sizes, 
    phis
    '''
    
    z_step = (max_z - min_z) / num_z_steps

    zvals = np.arange(min_z, max_z, z_step)

    # vulnerable cluster information
    component_sizes_list = []
    mean_vuln_sizes = []
    median_vuln_sizes = []
    std_vulnerable_cluster_sizes = []

    num_components_list = []

    largest_component_list = []

    # cascade information
    cascade_sizes = []


    for z in zvals:
        if verbose:
            print('mean degree:', z)
        p = z/(N-1)
        #print('z=', z, 'p=', p)

        cascadeNet = GlobalCascadeNetwork(N=N,
                                          p = p, 
                                          phis = phi,
                                          num_seeds = num_seeds)

        component_sizes = get_vulnerable_component_sizes(cascadeNet)
        component_sizes_list.append(component_sizes)

        num_components = len(component_sizes)
        num_components_list.append(num_components)

        try:
            giant_cluster_size = np.max(component_sizes)
        except:
            giant_cluster_size = 0

        largest_component_list.append(giant_cluster_size)

        mean_vulnerable_cluster_size = np.mean(component_sizes)
        median_vulnerable_cluster_size = np.median(component_sizes)
        std_vulnerable_cluster_size = np.std(component_sizes)

        mean_vuln_sizes.append(mean_vulnerable_cluster_size)
        median_vuln_sizes.append(median_vulnerable_cluster_size)
        std_vulnerable_cluster_sizes.append(std_vulnerable_cluster_size)

        cascadeNet.run_asynchronous_cascade2()
        cascade_size = cascadeNet.cascade_fraction()
        cascade_sizes.append(cascade_size)
    
    return(zvals,
           component_sizes_list,
           mean_vuln_sizes,
           median_vuln_sizes,
           std_vulnerable_cluster_sizes,
           num_components_list,
           largest_component_list,
           cascade_sizes, 
           cascadeNet.phi_vals)


## perhaps parallelize above code for speed
def create_stats_run_synchronous_cascade(N=1000,
                                         p=.5,
                                         num_seeds=1,
                                         phi_values=.5):
    '''
    function for creating and running a **synchronous** cascade
    '''

    cascadeNet = GlobalCascadeNetwork(N=N,
                                      p=p,
                                      num_seeds=num_seeds,
                                      phi_values=phi_values)



    cascadeNet.synchronous_cascade()

    return (cascadeNet.cascade_fraction())


def parallelize_cluster_stats_synchronous_cascade(num_iterations=100,
                                                  N=1000,
                                                  p=.5,
                                                  num_seeds=1,
                                                  phi_values=.5):
    '''
    function to use joblib for parallelism
    '''
    result = Parallel(n_jobs=-1)(delayed(create_stats_run_synchronous_cascade)(N=N,
                                                                               p=p,
                                                                               num_seeds=num_seeds,
                                                                               phi_values=phi_values)
                                 for i in range(num_iterations))

    return (result)


def cprofile_capture_to_df(profile_results):
    '''
    create a dataframe sorted descending by totaltime percall

    run:
    %%capture profile_results
    in a cell
    then run:
    cprofile_capture_to_df(profile_results)
    '''

    ## Parse the stdout text and split it into a table
    data = []
    started = False

    for l in profile_results.stdout.split("\n"):
        if not started:
            if l == "   ncalls  tottime  percall  cumtime  percall filename:lineno(function)":
                started = True
                data.append(l)
        else:
            data.append(l)
    content = []
    for l in data:
        fs = l.find(" ", 8)
        content.append(
            tuple([l[0:fs], l[fs:fs + 9], l[fs + 9:fs + 18], l[fs + 18:fs + 27], l[fs + 27:fs + 46], l[fs + 36:]]))
    prof_df = pd.DataFrame(content[1:], columns=content[0])

    prof_df.rename(columns=lambda x: x.strip(), inplace=True)
    prof_df.columns = ['ncalls',
                       'tottime',
                       'percall1',
                       'cumtime',
                       'percall2',
                       'filename:lineno(function)']

    return prof_df.sort_values('percall1', ascending=False)


def log_distribute_radii(num_radii = 5,
                         maxval = 1.42,
                         minval = 0.1):
    logmax = np.log(maxval)
    logmin = np.log(minval)

    vals = np.arange(logmin, logmax, (logmax - logmin)/(num_radii-1))

    exp_vals = np.exp(vals)
    exp_vals = np.append(exp_vals, maxval)
    return(exp_vals)