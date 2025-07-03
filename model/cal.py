import torch
import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import random
import multiprocessing as mp
from scipy.stats import rankdata
import math

def build_graph(adj):
    # G = nx.Graph()
    # num_nodes = adj.shape[0]
    # G.add_nodes_from(range(num_nodes))
    # edges = adj.nonzero()  # Get non-zero elements in the adjacency matrix
    # edge_list = [(i, j) for i, j in zip(edges[0], edges[1]) if i != j]
    # G.add_edges_from(edge_list)
    adj_numpy = adj.cpu().numpy()

    # 使用 nx.from_numpy_array() 创建图
    G = nx.from_numpy_array(adj_numpy)
    return G

def compute_diameter(adjacency_matrix):
    """
    计算图的直径（最长最短路径）
    """
    adjacency_matrix = csr_matrix(adjacency_matrix)
    dist_matrix = shortest_path(adjacency_matrix, directed=False)

    return np.max(dist_matrix[dist_matrix != np.inf])

def compute_ricci_curvature(G):
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    ricci_dict = nx.get_edge_attributes(orc.G, 'ricciCurvature')
    return ricci_dict

def single_source_shortest_path_range(graph, node_range, cutoff):
    paths_dict = {}
    for node in node_range:
        paths_dict[node] = nx.single_source_shortest_path(graph, node, cutoff)   # unweighted
    return paths_dict

def all_pairs_shortest_path_parallel(graph, cutoff=None, num_workers=8):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    paths_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return paths_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def calculate_rc_and_sc_for_node(node, train_idx, G, ricci_dict, D_G):
    # Get shortest paths to labeled nodes (train_idx)
    shortest_path_lengths = []
    sc_sum = 0
    for labeled_node in train_idx.numpy():
        if node != labeled_node:  # Skip if source and target are the same
            try:
                path = nx.shortest_path(G, source=node, target=labeled_node)
                path_length = len(path) - 1  # Path length (minus 1 for edge count)
                shortest_path_lengths.append(path_length)
                
                # Sum up Ricci curvature for edges on the path
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    if edge in ricci_dict:
                        sc_sum += ricci_dict[edge]
            except nx.NetworkXNoPath:
                # If no path exists, we can't compute RC/SC for this labeled node
                pass
    
    # RC: Mean of (1 - log(path_length) / log(D_G)) for shortest path lengths to labeled nodes
    rc_value = np.mean([1 - np.log(length) / np.log(D_G) for length in shortest_path_lengths]) if shortest_path_lengths else 0
    
    # SC: Mean Ricci curvature of edges on shortest paths to labeled nodes
    sc_value = sc_sum / len(shortest_path_lengths) if shortest_path_lengths else 0
    
    return node, rc_value, sc_value

def calculate_rc_and_sc(adj, train_idx):
    num_nodes = adj.shape[0]
    D_G = np.max(np.linalg.norm(adj.numpy(), axis=1))  # Graph's diameter
    
    # Step 1: Compute the graph
    G = build_graph(adj)
    
    # Step 2: Compute Ricci curvature for edges
    ricci_dict = compute_ricci_curvature(G)
    
    # Step 3: Precompute all pairs shortest paths using parallel computation
    # all_paths = all_pairs_shortest_path_parallel(G, cutoff=None)
    
    # for id in range(num_nodes):
    #     calculate_rc_and_sc_for_node(id, train_idx, G, ricci_dict, D_G)
    
    # Step 4: Use multiprocessing to calculate RC and SC for all nodes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(calculate_rc_and_sc_for_node, [(node, train_idx, G, ricci_dict, D_G) for node in range(num_nodes)])
    
    # Step 5: Collect results and divide by the size of train_idx to get mean
    rc_dict = {node: rc for node, rc, _ in results}
    sc_dict = {node: sc for node, _, sc in results}
    
    # Ensure the results are ordered by node id
    rc_values = [rc_dict[node] for node in range(num_nodes)]
    sc_values = [sc_dict[node] for node in range(num_nodes)]
    
    rc_ranks = rankdata(rc_values)  # 对 RC 值进行排名
    sc_ranks = rankdata(sc_values)  # 对 SC 值进行排名
    
    # Step 6: Calculate the scores based on the formula
    scores = []
    for i in range(num_nodes):
        rc_rank = rc_ranks[i]
        sc_rank = sc_ranks[i]
        # 计算最终的分数
        score = (math.cos(rc_rank / len(train_idx) * math.pi) + 1) * (math.cos(sc_rank / len(train_idx) * math.pi) + 1)
        scores.append(score)
    
    return scores
    
    # Divide RC and SC values by the number of labeled nodes (train_idx size)
    # train_size = len(train_idx)
    # rc_values = [rc / train_size for rc in rc_values]
    # sc_values = [sc / train_size for sc in sc_values]
    
    # return rc_values, sc_values