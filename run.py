import argparse
import os
import json
import networkx as nx
from src import create_data, graph_extractor, visualize, config

def main():
    """
    メインの実行関数。
    コマンドライン引数を解釈し、データ生成から可視化までのパイプラインを制御します。
    各ステップはデフォルトで実行され、--skip-*系の引数でスキップできます。
    """
    parser = argparse.ArgumentParser(description='Wikipediaデータから概念マップを生成・可視化します。')
    
    # モード設定
    parser.add_argument('--test', action='store_true', help='テストデータセットで実行します。')

    # スキップフラグ
    parser.add_argument('--skip-create-test-data', action='store_true', help='テストデータXMLの作成をスキップします。')
    parser.add_argument('--skip-graph-extraction', action='store_true', help='グラフデータの抽出をスキップします。')
    parser.add_argument('--skip-simulation', action='store_true', help='物理シミュレーションをスキップします。')
    parser.add_argument('--skip-visualization', action='store_true', help='HTML/PNGの可視化ファイル生成をスキップします。')

    args = parser.parse_args()

    # --- パス設定 ---
    if args.test:
        wiki_xml_path = 'test_data/intermediate_wiki.xml'
        graph_path = config.GRAPH_PATH_TEST
        depth_path = config.DEPTH_PATH_TEST
        simulation_path = config.SIMULATION_RESULT_PATH_TEST
        output_html_path = config.OUTPUT_HTML_PATH_TEST
        output_png_path = config.OUTPUT_PNG_PATH_TEST
    else:
        wiki_xml_path = 'files/jawiki-latest-pages-articles.xml'
        graph_path = config.GRAPH_PATH_FULL
        depth_path = config.DEPTH_PATH_FULL
        simulation_path = config.SIMULATION_RESULT_PATH_FULL
        output_html_path = config.OUTPUT_HTML_PATH_FULL
        output_png_path = config.OUTPUT_PNG_PATH_FULL

    # 1. テストデータ作成
    if args.test and not args.skip_create_test_data:
        print("--- ステージ1: テストデータ作成 ---")
        full_xml_path = 'files/jawiki-latest-pages-articles.xml'
        if not os.path.exists(full_xml_path):
            print(f"エラー: 大元のWikipediaダンプが見つかりません: {full_xml_path}")
            print("フルデータセットをダウンロードし、配置してください。")
            return
        create_data.create_small_wiki_dump(full_xml_path, wiki_xml_path, page_limit=config.PAGE_LIMIT_FOR_TEST)
    
    # 2. グラフ抽出
    if not args.skip_graph_extraction:
        print("--- ステージ2: グラフ抽出 ---")
        if not os.path.exists(wiki_xml_path):
            print(f"エラー: 入力となるWikipedia XMLが見つかりません: {wiki_xml_path}")
            return
        graph_extractor.extract_titles_and_links(wiki_xml_path, graph_path, depth_path)

    # --- 可視化対象ノードの選定 ---
    final_nodes_list = []
    if not args.skip_simulation or not args.skip_visualization:
        print("--- 可視化対象ノードの選定 ---")
        if not os.path.exists(graph_path):
            print(f"エラー: グラフファイルが見つかりません: {graph_path}")
            print("グラフ抽出ステップを先に実行してください。")
            return
        
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        G = nx.node_link_graph(graph_data)

        titles_to_visualize = config.TITLES_TO_VISUALIZE
        if titles_to_visualize:
            print(f"設定ファイルで指定されたタイトルを元にノードを選定: {titles_to_visualize}")
            nodes_for_visualization = set(titles_to_visualize)
            for title in titles_to_visualize:
                if title in G:
                    nodes_for_visualization.update(G.neighbors(title))
        else:
            min_links = config.MIN_LINKS_FOR_VISUALIZATION
            print(f"リンク数が{min_links}以上のノードを自動選定します。")
            nodes_for_visualization = {n for n, degree in G.degree() if degree >= min_links}

        final_nodes_list = list(nodes_for_visualization)
        if not final_nodes_list:
            print("警告: 可視化対象のノードが見つかりませんでした。処理を中断します。")
            return
        print(f"選定されたノード数: {len(final_nodes_list)}")

    # 3. シミュレーション
    if not args.skip_simulation:
        print("--- ステージ3: シミュレーション ---")
        visualize.run_simulation(
            G=G,
            nodes_to_visualize=final_nodes_list,
            node_map={node: i for i, node in enumerate(final_nodes_list)},
            nodes_list=final_nodes_list,
            simulation_output_path=simulation_path
        )

    # 4. 可視化
    if not args.skip_visualization:
        print("--- ステージ4: 可視化 ---")
        visualize.visualize_concepts(
            graph_path=graph_path,
            depth_path=depth_path,
            titles_to_visualize=final_nodes_list,
            output_html_path=output_html_path,
            output_png_path=output_png_path,
            simulation_input_path=simulation_path
        )

    print("--- 処理完了 ---")

if __name__ == '__main__':
    main()