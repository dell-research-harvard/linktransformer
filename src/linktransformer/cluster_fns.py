from itertools import combinations


def cluster(cluster_type, cluster_params, corpus_embeddings, corpus_ids=None):
    # Define default values for each cluster type
    default_params = {
        "agglomerative": {
            "threshold": 0.5,
            "clustering linkage": "ward",  # You can choose a default linkage method
            "metric": "euclidean",  # You can choose a default metric
        },
        "HDBScan": {
            "min cluster size": 5,
            "min samples": 1,
            "metric": "cosine"

        },
        "SLINK": {
            "min cluster size": 2,
            "threshold": 0.1,
            "metric": "cosine"
        },
    }

    if cluster_type not in ["agglomerative", "HDBScan", "SLINK"]:
        raise ValueError('cluster_type must be "agglomerative", "HDBScan", or "SLINK"')

    # Get the default parameters based on cluster_type
    default_params_for_type = default_params.get(cluster_type, {})

    # Fill in defaults on a copy so the caller's dict is never modified
    cluster_params = dict(cluster_params or {})
    for key, default_value in default_params_for_type.items():
        cluster_params.setdefault(key, default_value)

    # Validate the cluster_params
    if cluster_type == "agglomerative":
        required_params = ["threshold", "clustering linkage", "metric"]
    elif cluster_type == "HDBScan":
        required_params = ["min cluster size", "min samples"]
    elif cluster_type == "SLINK":
        required_params = ["min cluster size", "threshold"]
    else:
        raise ValueError('Invalid cluster_type')

    for param in required_params:
        if param not in cluster_params:
            raise ValueError(f'cluster_params must contain "{param}"')

    if cluster_type == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_params["threshold"],
            linkage=cluster_params["clustering linkage"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "SLINK":
        from sklearn.cluster import DBSCAN

        clustering_model = DBSCAN(
            eps=cluster_params["threshold"],
            min_samples=cluster_params["min cluster size"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "HDBScan":
        from linktransformer.utils import _require

        hdbscan = _require("hdbscan")
        clustering_model = hdbscan.HDBSCAN(
            min_cluster_size=cluster_params["min cluster size"],
            min_samples=cluster_params["min samples"],
            gen_min_span_tree=True
        )

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_


    return cluster_assignment


def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def edges_from_clusters(cluster_dict):
    """
    Convert every pair in a cluster into an edge
    """
    cluster_edges = []
    for cluster_id in list(cluster_dict.keys()):
        art_ids_list = cluster_dict[cluster_id]
        edge_list = [list(comb) for comb in combinations(art_ids_list, 2)]
        cluster_edges.extend(edge_list)

    return cluster_edges
