import os
import numpy as np
import networkx as nx
import json
import plotly.graph_objects as go
import math
from . import config

def calculate_forces(positions, G, node_map, nodes_list):
    """
    各粒子にかかる引力と斥力を計算します。
    Calculates attraction and repulsion forces on each particle.
    """
    num_particles = positions.shape[0]
    forces = np.zeros_like(positions)

    for i in range(num_particles):
        node_i = nodes_list[i]
        
        for j in range(num_particles):
            if i == j: continue

            node_j = nodes_list[j]

            delta = positions[j] - positions[i]
            distance = np.linalg.norm(delta)
            if distance == 0: distance = 1e-6

            if distance > config.DISTANCE_PARAM * 5:
                continue

            repulsion = config.REPULSION_STRENGTH * (config.DISTANCE_PARAM / distance)**13 * (delta / distance)
            forces[i] -= repulsion

            # 引力 (リンクの重みに比例)
            link_weight = 0
            if G.has_edge(node_i, node_j):
                link_weight += G[node_i][node_j].get('weight', 1)
            if G.has_edge(node_j, node_i):
                link_weight += G[node_j][node_i].get('weight', 1)

            if link_weight > 0:
                attraction = link_weight * config.ATTRACTION_STRENGTH * np.exp(distance/config.DISTANCE_PARAM)*(delta / distance)
                forces[i] += attraction

    return forces

def run_simulation(G, nodes_to_visualize, node_map, nodes_list, simulation_output_path):
    """
    力指向レイアウトシミュレーションを実行し、最終的なノード位置を返します。

    Args:
        G (nx.Graph): 対象のグラフ。
        nodes_to_visualize (list): 可視化対象のノードリスト。
        node_map (dict): ノード名をインデックスにマッピングする辞書。
        nodes_list (list): インデックスをノード名にマッピングするリスト。
        simulation_output_path (str): シミュレーション結果を保存するNPYファイルへのパス。

    Returns:
        np.array: 各ノードの最終的な2D座標の配列。
    """
    num_particles = len(nodes_to_visualize)

    # --- 動的な境界サイズの計算 ---
    # 粒子数とconfig設定に基づき、シミュレーション空間のサイズを動的に決定する
    # 粒子1つあたりの面積を計算し、全体の必要面積から境界サイズを算出
    area_per_particle = (2 * config.DISTANCE_PARAM * config.BOUNDARY_SCALE_FACTOR) ** 2
    total_required_area = area_per_particle * num_particles / config.PACKING_DENSITY
    simulation_boundary_size = math.sqrt(total_required_area)
    print(f"粒子数: {num_particles}, 動的境界サイズ: {simulation_boundary_size:.2f}")
    
    grid_size = math.ceil(math.sqrt(num_particles))
    cell_size = simulation_boundary_size / grid_size
    
    positions = np.zeros((num_particles, 2))
    for i in range(num_particles):
        row = i // grid_size
        col = i % grid_size
        
        x = (col + 0.5) * cell_size - (simulation_boundary_size / 2)
        y = (row + 0.5) * cell_size - (simulation_boundary_size / 2)
        positions[i] = [x, y]

    initial_velocity_scale = math.sqrt(2 * config.INITIAL_TEMPERATURE)
    velocities = np.random.randn(num_particles, 2) * initial_velocity_scale

    masses = np.ones((num_particles, 1))

    min_bound = -simulation_boundary_size / 2
    max_bound = simulation_boundary_size / 2

    print(f"シミュレーション開始 ({config.ITERATIONS} イテレーション)...")
    for iteration in range(config.ITERATIONS):
        old_positions = positions.copy()

        forces = calculate_forces(positions, G, node_map, nodes_list)
        
        velocities *= config.DAMPING_FACTOR 
        velocities += (forces / masses) * config.DT 
        positions += velocities * config.DT 

        for i in range(num_particles):
            for dim in range(2):
                if positions[i, dim] < min_bound:
                    positions[i, dim] = min_bound
                    velocities[i, dim] *= -1.0
                elif positions[i, dim] > max_bound:
                    positions[i, dim] = max_bound
                    velocities[i, dim] *= -1.0

        kinetic_energy = 0.5 * np.sum(masses * velocities**2)
        current_sim_temperature = kinetic_energy / num_particles

        target_temperature = config.INITIAL_TEMPERATURE * (config.COOLING_RATE ** iteration)
        
        if current_sim_temperature > 1e-6:
            scale_factor = math.sqrt(target_temperature / current_sim_temperature)
            velocities *= scale_factor

        max_movement = np.max(np.linalg.norm(positions - old_positions, axis=1))
        if max_movement < config.MOVEMENT_THRESHOLD:
            print(f"シミュレーションが収束しました (イテレーション {iteration}/{config.ITERATIONS})。")
            break

        if iteration % (config.ITERATIONS // 10) == 0:
            print(f"イテレーション {iteration}/{config.ITERATIONS}, 系の温度/目標温度: {current_sim_temperature:.3f}/{target_temperature:.3f}, 最大移動量: {max_movement:.4f}")

    print("シミュレーション完了。")
    np.save(simulation_output_path, positions)
    print(f"シミュレーション結果を '{simulation_output_path}' に保存しました。")
    
    return positions

def visualize_concepts(graph_path, depth_path, titles_to_visualize, output_html_path, output_png_path, simulation_input_path):
    """
    グラフデータと深さデータに基づき、概念の可視化を行います。

    Args:
        graph_path (str): グラフデータ (.json) へのパス。
        depth_path (str): 深さデータ (.json) へのパス。
        titles_to_visualize (list[str]): 可視化するWikipediaタイトルのリスト。
        output_html_path (str): 出力するインタラクティブHTMLファイルへのパス。
        output_png_path (str): 出力するPNG画像ファイルへのパス。
        simulation_input_path (str): シミュレーション結果をロードするパス (.npy)。
    """
    print(f"グラフ '{graph_path}' と深さデータ '{depth_path}' をロード中...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data, edges="links")
    with open(depth_path, 'r', encoding='utf-8') as f:
        depth_data = json.load(f)
    print("データロード完了。")

    nodes_to_visualize = [title for title in titles_to_visualize if title in G]
    if not nodes_to_visualize:
        print("エラー: 可視化できる有効なノードがありません。")
        return

    node_map = {node: i for i, node in enumerate(nodes_to_visualize)}
    nodes_list = nodes_to_visualize

    if os.path.exists(simulation_input_path):
        print(f"既存のシミュレーション結果 '{simulation_input_path}' をロード中...")
        positions = np.load(simulation_input_path)
        print("シミュレーション結果ロード完了。")
    else:
        print("シミュレーション結果が見つからないため、新たに実行します。")
        positions = run_simulation(G, nodes_to_visualize, node_map, nodes_list, simulation_input_path)

    print("可視化を生成中...")

    marker_sizes = [10 for _ in nodes_to_visualize]
    node_colors = ['blue' if G.nodes[node]['type'] == 'article' else 'red' for node in nodes_to_visualize]

    fig = go.Figure(data=go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+text',
        text=nodes_to_visualize,
        textposition="bottom center",
        marker=dict(size=marker_sizes, color=node_colors),
        hoverinfo='text'
    ))

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    fig.update_layout(
        title='概念マップ',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[x_min - x_padding, x_max + x_padding]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[y_min - y_padding, y_max + y_padding]),
        showlegend=False,
        width=2000,
        height=2000
    )

    fig.write_html(output_html_path)
    print(f"可視化結果を '{output_html_path}' に保存しました。")
    
    import matplotlib.pyplot as plt
    import japanize_matplotlib

    # Matplotlibで静的画像を生成
    fig_mpl, ax = plt.subplots(figsize=(20, 20))
    
    # Plotlyの色と合わせる
    colors = ['#1f77b4' if G.nodes[node]['type'] == 'article' else '#d62728' for node in nodes_to_visualize]
    
    ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=100)

    # ラベルを追加
    for i, node in enumerate(nodes_to_visualize):
        ax.text(positions[i, 0], positions[i, 1], node, fontsize=8)

    ax.set_title('概念マップ')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    try:
        fig_mpl.savefig(output_png_path, format='png', dpi=150)
        print(f"PNG画像を '{output_png_path}' に保存しました。")
    except Exception as e:
        print(f"PNG画像の保存に失敗しました: {e}")
    plt.close(fig_mpl) # メモリを解放
