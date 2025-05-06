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
    주어진 데이터(data)를 사용하여 Bayesian Network 구조를 학습합니다.
    
    Parameters:
      - data: pandas DataFrame (범주형 변수로 변환되어 있어야 함)
      - scoring_method: 'bic' 또는 'k2' (점수 기반 구조 학습)
      - max_indegree: 각 노드의 최대 부모 수 제한 (optional)
      - verbose: True이면 학습 과정의 로그 출력
    
    Returns:
      - bn_model: 학습된 pgmpy.models.BayesianModel (DAG)
    """
    if scoring_method.lower() == 'bic':
        scorer = BicScore(data)
    elif scoring_method.lower() == 'k2':
        scorer = K2Score(data)
    else:
        raise ValueError("scoring_method must be either 'bic' or 'k2'")
    
    # HillClimbSearch 생성 시 scoring_method 인자는 제거합니다.
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
    학습된 BN의 DAG에 대해 토폴로지 정렬을 수행하여 feature 순서를 도출합니다.
    randomize=True이면, in-degree 0 노드들 사이에 무작위성을 부여합니다.
    
    Parameters:
      - bn_model: pgmpy.models.BayesianModel
      - randomize: Boolean, 무작위성을 부여할지 여부
    
    Returns:
      - order: 토폴로지 정렬된 feature 리스트
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
        print("pygraphviz가 설치되어 있지 않습니다. 기본 spring_layout 사용.")
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
    CSV 파일에서 데이터를 읽어, 기대하는 컬럼들만 선택하고 결측치 처리를 수행한 후,
    범주형 변수(문자열)로 변환한 DataFrame을 반환합니다.
    
    Parameters:
      - csv_file: CSV 파일 경로
      - expected_columns: 사용할 컬럼 리스트
      - fill_missing: True이면, 결측치를 "None" 문자열로 대체
      
    Returns:
      - data: pandas DataFrame (범주형)
    """
    data = pd.read_csv(csv_file)
    missing = set(expected_columns) - set(data.columns)
    if missing:
        raise ValueError("다음 컬럼들이 누락되었습니다: " + ", ".join(missing))
    data = data[expected_columns]
    if fill_missing:
        for col in expected_columns:
            data[col] = data[col].fillna("None")
    # 모든 값을 문자열로 변환하여 범주형 변수로 처리
    data = data.astype(str)
    return data

def sample_topological_order(bn_model):
    """
    BN 모델의 DAG에서 가능한 valid한 선형 확장 중 하나를 무작위로 샘플링합니다.
    BN의 부분 순서를 만족하는 여러 선형 확장 중 하나를 반환합니다.
    """
    # BN 모델의 에지들을 사용해 NetworkX DiGraph를 생성합니다.
    G = nx.DiGraph(bn_model.edges())
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Learned BN structure is not a DAG!")
    
    order = []
    G_copy = G.copy()
    # G_copy에서 in-degree가 0인 노드들 중 하나를 무작위로 선택해 선형 확장을 구성합니다.
    while G_copy.nodes():
        zero_in_degree = [node for node in G_copy.nodes() if G_copy.in_degree(node) == 0]
        selected = random.choice(zero_in_degree)
        order.append(selected)
        G_copy.remove_node(selected)
    return order

def sample_topological_order_random_path(bn_model):
    """
    BN 모델의 DAG에서 무작위 경로를 따라가며 토폴로지 순서를 생성합니다.
    각 단계에서 현재 노드의 outgoing 엣지 중 하나를 무작위로 선택하고,
    선택되지 않은 다른 엣지들은 무시합니다.
    
    Parameters:
      - bn_model: pgmpy.models.BayesianModel
    
    Returns:
      - order: 토폴로지 정렬된 노드 리스트
    """
    import random
    import networkx as nx
    
    # BN 모델의 엣지들을 사용해 NetworkX DiGraph 생성
    G = nx.DiGraph(bn_model.edges())
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Learned BN structure is not a DAG!")
    
    # 난수 생성기 재설정
    random.seed()
    
    # 결과 순서를 저장할 리스트
    order = []
    
    # 작업용 그래프 복사
    G_work = G.copy()
    
    # 모든 노드가 처리될 때까지 반복
    while G_work.nodes():
        # 현재 in-degree=0인 노드들 찾기
        zero_in_degree = [node for node in G_work.nodes() if G_work.in_degree(node) == 0]
        
        if not zero_in_degree:
            # 이론적으로 DAG에서는 발생하지 않아야 하지만, 안전장치로 추가
            break
        
        # in-degree=0인 노드들 중 하나를 무작위로 선택
        current = random.choice(zero_in_degree)
        order.append(current)
        
        # 현재 경로를 따라갈 포인터
        current_node = current
        
        # 경로 따라가기
        path_continues = True
        while path_continues:
            # 현재 노드에서 나가는 엣지(다음으로 갈 수 있는 노드들)
            neighbors = list(G_work.successors(current_node))
            
            # 나가는 엣지가 있으면 하나를 무작위로 선택
            if neighbors:
                # 무작위로 하나의 이웃 노드 선택
                next_node = random.choice(neighbors)
                
                # 현재 노드 제거 (이전 선택으로 돌아가지 않도록)
                G_work.remove_node(current_node)
                
                # 선택된 다음 노드가 이제 진입차수가 0인지 확인
                if G_work.in_degree(next_node) == 0:
                    # 경로 계속 따라가기
                    order.append(next_node)
                    current_node = next_node
                else:
                    # 다음 노드의 진입차수가 아직 0이 아님 -> 경로 종료
                    path_continues = False
            else:
                # 더 이상 나가는 엣지가 없음 -> 현재 노드 제거 후 경로 종료
                G_work.remove_node(current_node)
                path_continues = False
    
    return order