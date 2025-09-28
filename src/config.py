"""
シミュレーションと可視化に関する設定値を定義します。
Defines configuration values for simulation and visualization.
"""

# --- Data Creation Parameters ---
# テストデータ作成に関するパラメータ
PAGE_LIMIT_FOR_TEST = 20000


# --- Visualization Parameters ---
# 可視化に関するパラメータ

# 可視化したい記事のタイトルリスト
# このリストが空の場合、MIN_LINKS_FOR_VISUALIZATION の設定が使用されます。
# List of article titles to visualize.
# If this list is empty, the MIN_LINKS_FOR_VISUALIZATION setting will be used.
TITLES_TO_VISUALIZE = [
    # "人工知能", "機械学習", "深層学習", "ニューラルネットワーク", 
    # "自然言語処理", "コンピュータビジョン", "強化学習", "Python",
    # "数学", "物理学", "化学", "生物学",
    # "プログラミング", "アルゴリズム", "データ構造",
    # "第二次世界大戦", "冷戦", "グローバリゼーション",
    # "哲学", "歴史", "経済学", "心理学"
]

# TITLES_TO_VISUALIZE が空の場合に、可視化対象とするノードの最小リンク数
# Minimum number of links for a node to be visualized when TITLES_TO_VISUALIZE is empty.
MIN_LINKS_FOR_VISUALIZATION = 300


# --- Simulation Physics Parameters ---
# 力指向レイアウトシミュレーションの物理パラメータ

# 引力の強さの係数
ATTRACTION_STRENGTH = 2.0

# 斥力の強さの係数
REPULSION_STRENGTH = 100.0

# 斥力の距離パラメータ (Lennard-Jonesポテンシャルのsigmaに相当)
DISTANCE_PARAM = 1.0

# シミュレーテッドアニーリングの初期温度
INITIAL_TEMPERATURE = 10.0

# シミュレーテッドアニーリングの冷却率
COOLING_RATE = 0.95

# シミュレーションの最大イテレーション数
ITERATIONS = 100

# 時間ステップ
DT = 0.01

# 速度の減衰係数
DAMPING_FACTOR = 1

# 収束判定のための移動量の閾値
MOVEMENT_THRESHOLD = 0.1

# 動的なシミュレーション境界計算のためのパラメータ
# パラメータは visualize.py の run_simulation 内で使用される。
# Note: より正確な平衡距離xはランベルトW関数を用いて x/distance_param = 13*W((repulsion_strength/attraction_strength)^(1/13)/13) で近似できる。
# The following parameters are for dynamic simulation boundary calculation.
# They are used within run_simulation in visualize.py.
# Note: A more precise equilibrium distance 'x' can be approximated using the Lambert W function:
# x/distance_param = 13*W((repulsion_strength/attraction_strength)^(1/13)/13)

# シミュレーション空間のパッキング密度（充填率）
PACKING_DENSITY = 0.4

# 動的境界計算のためのスケール係数
BOUNDARY_SCALE_FACTOR = 1.4


# --- File Paths ---
# 各種ファイルのパス設定

# プロジェクトのルートディレクトリからの相対パスで定義
GRAPH_PATH_TEST = "data/wiki_graph_test.json"
DEPTH_PATH_TEST = "data/depth_test.json"
SIMULATION_RESULT_PATH_TEST = "data/simulation_results/test_positions.npy"
OUTPUT_HTML_PATH_TEST = "data/simulation_results/test_visualization.html"
OUTPUT_PNG_PATH_TEST = "data/simulation_results/test_visualization.png"

GRAPH_PATH_FULL = "data/wiki_graph_full.json"
DEPTH_PATH_FULL = "data/depth_full.json"
SIMULATION_RESULT_PATH_FULL = "data/simulation_results/full_positions.npy"
OUTPUT_HTML_PATH_FULL = "data/simulation_results/full_visualization.html"
OUTPUT_PNG_PATH_FULL = "data/simulation_results/full_visualization.png"
