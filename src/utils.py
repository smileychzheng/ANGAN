import numpy as np
import scipy.io as sio


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_uu_edges(filename):
    """read data from files
    Args:
        train_filename: training file name
        test_filename: test file name
    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """
    with open(filename, "r") as fin:
        firstline = fin.readline().strip().split()
        nodes = int(firstline[0])
        lines = fin.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]

    graph = {}
    for node in range(nodes):
        graph[node] = []

    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    return nodes, graph

def read_ua_edges(filename):
    with open(filename, "r") as fin:
        firstline = fin.readline().strip().split()
        nodes = int(firstline[0])
        lines = fin.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    graph = {}
    for node in range(nodes):
        graph[node] = []
    for edge in edges:
        graph[edge[0]].append(edge[1])

    return graph

def read_au_edges(filename):
    with open(filename, "r") as fin:
        firstline = fin.readline().strip().split()
        nodes = int(firstline[0])
        lines = fin.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]

    graph = {}
    for node in range(nodes):
        graph[node] = []

    for edge in edges:
        graph[edge[1]].append(edge[0])

    return nodes, graph


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def read_embeddings(filename):
    """read pretrained node embeddings
    """
    fin = sio.loadmat(filename)
    embedding = fin['embedding']

    return embedding


def reindex_node_id(edges):
    """reindex the original node ID to [0, node_num)

    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.add(node_set.index(edge[0]))
        new_nodes = new_nodes.add(node_set.index(edge[1]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes


def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors
    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(neighbors))])

    # for each edge in the test set, sample a negative edge
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        neg_nodes = list(nodes.difference(set(neighbors[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()
