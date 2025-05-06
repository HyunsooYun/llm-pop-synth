# BN.py
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import logging
import sys

import plotly.graph_objects as go
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
from pgmpy.models import BayesianModel

def learn_bn_structure(data: pd.DataFrame, scoring_method: str = 'bic', max_indegree: int = None, verbose: bool = False):
    """
    Learns the structure of a Bayesian Network using the given data.

    Parameters:
      - data: pandas DataFrame (must be converted to categorical variables)
      - scoring_method: 'bic' or 'k2' (score-based structure learning)
      - max_indegree: optional limit on the maximum number of parents per node
      - verbose: if True, prints logs during the learning process

    Returns:
      - bn_model: learned pgmpy.models.BayesianModel (DAG)
    """
    if scoring_method.lower() == 'bic':
        scorer = BicScore(data)
    elif scoring_method.lower() == 'k2':
        scorer = K2Score(data)
    else:
        raise ValueError("scoring_method must be either 'bic' or 'k2'")
    
    hc = HillClimbSearch(data)
    if max_indegree is not None:
        bn_model = hc.estimate(scoring_method=scorer, max_indegree=max_indegree)
    else:
        bn_model = hc.estimate(scoring_method=scorer)
    
    if verbose:
        logging.info("Learned BN structure (edges): %s", str(bn_model.edges()))
    return bn_model


def get_topological_order(bn_model, randomize: bool = False):
    """
    Performs topological sorting on the DAG of a trained Bayesian Network to determine the feature order.
    If randomize=True, introduces randomness among nodes with in-degree 0.

    Parameters:
      - bn_model: pgmpy.models.BayesianModel
      - randomize: Boolean, whether to introduce randomness

    Returns:
      - order: list of features in topologically sorted order
    """
    import networkx as nx
    import random
    
    # BN 모델의 에지들을 사용하여 NetworkX DiGraph 생성
    G = nx.DiGraph(bn_model.edges())
    
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Learned BN structure is not a DAG!")
    
    order = []
    in_degree = {node: G.in_degree(node) for node in G.nodes()}
    zero_in_degree = [node for node in G.nodes() if in_degree[node] == 0]
    if randomize:
        random.shuffle(zero_in_degree)
    
    while zero_in_degree:
        node = zero_in_degree.pop(0)
        order.append(node)
        for neighbor in list(G.successors(node)):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)
        if randomize:
            random.shuffle(zero_in_degree)
    
    return order

def visualize_bn_plotly_dot(bn_model, title="Learned Bayesian Network (Plotly with DOT Layout)"):
    G = nx.DiGraph(bn_model.edges())
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        print("pygraphviz is not installed. Using default spring_layout instead.")
        pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[len(list(G.adj[node])) for node in G.nodes()],
            size=20,
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Connections', side='right'),
                xanchor='left'
            ),
            line_width=2)
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

def learn_bn_from_csv(csv_file: str, expected_columns: list, fill_missing: bool = True) -> pd.DataFrame:
    """
    Reads data from a CSV file, selects only the expected columns, handles missing values,
    and returns a DataFrame with categorical (string) variables.

    Parameters:
      - csv_file: path to the CSV file
      - expected_columns: list of columns to use
      - fill_missing: if True, replaces missing values with the string "None"

    Returns:
      - data: pandas DataFrame (categorical)
    """
    data = pd.read_csv(csv_file)
    missing = set(expected_columns) - set(data.columns)
    if missing:
        raise ValueError("The following columns are missing: " + ", ".join(missing))
    data = data[expected_columns]
    if fill_missing:
        for col in expected_columns:
            data[col] = data[col].fillna("None")
    # Convert all values to strings to treat them as categorical variables
    data = data.astype(str)
    return data

def sample_topological_order(bn_model):
    """
    Randomly samples one valid linear extension from the DAG of the BN model.
    Returns one of the many linear extensions that satisfy the partial order of the BN.
    """

    # Create a NetworkX DiGraph using the edges of the BN model.
    G = nx.DiGraph(bn_model.edges())
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Learned BN structure is not a DAG!")
    
    order = []
    G_copy = G.copy()
    # # Randomly select one node with in-degree 0 from G_copy to build the linear extension.
    while G_copy.nodes():
        zero_in_degree = [node for node in G_copy.nodes() if G_copy.in_degree(node) == 0]
        selected = random.choice(zero_in_degree)
        order.append(selected)
        G_copy.remove_node(selected)
    return order

def sample_topological_order_random_path(bn_model):
    """
    Generates a topological order by following a random path in the DAG of the BN model.
    At each step, one of the current node's outgoing edges is randomly selected,
    and all other unselected edges are ignored.

    Parameters:
      - bn_model: pgmpy.models.BayesianModel

    Returns:
      - order: list of nodes in topologically sorted order
    """
    import random
    import networkx as nx

    # Create a NetworkX DiGraph using the edges from the BN model
    G = nx.DiGraph(bn_model.edges())
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Learned BN structure is not a DAG!")

    # Reset the random seed
    random.seed()

    # List to store the resulting topological order
    order = []

    # Copy of the graph for modification
    G_work = G.copy()

    # Repeat until all nodes are processed
    while G_work.nodes():
        # Find nodes with in-degree 0
        zero_in_degree = [node for node in G_work.nodes() if G_work.in_degree(node) == 0]

        if not zero_in_degree:
            # Should not happen in a DAG, but added as a safeguard
            break

        # Randomly select one node from the zero in-degree nodes
        current = random.choice(zero_in_degree)
        order.append(current)

        # Pointer to follow the path
        current_node = current

        # Follow the path
        path_continues = True
        while path_continues:
            # Get successors (outgoing neighbors) of the current node
            neighbors = list(G_work.successors(current_node))

            if neighbors:
                # Randomly select one of the outgoing neighbors
                next_node = random.choice(neighbors)

                # Remove current node to avoid revisiting it
                G_work.remove_node(current_node)

                # Check if the selected next node now has in-degree 0
                if G_work.in_degree(next_node) == 0:
                    # Continue along the path
                    order.append(next_node)
                    current_node = next_node
                else:
                    # Cannot proceed; next node still has other parents
                    path_continues = False
            else:
                # No more outgoing edges; remove current node and stop path
                G_work.remove_node(current_node)
                path_continues = False

    return order
