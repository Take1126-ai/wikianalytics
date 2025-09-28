import os
import re
from lxml import etree
import networkx as nx
import json

# Regex patterns for extracting links and categories
INTERNAL_LINK_PATTERN = re.compile(r'\[\[(?!ファイル:|File:|Category:|カテゴリ:)([^|\]]+)(?:\|[^|\]]+)?\]\]')
CATEGORY_LINK_PATTERN = re.compile(r'\[\[(Category|カテゴリ):([^|\]]+)(?:\|[^|\]]+)?\]\]')

def extract_titles_and_links(wiki_xml_path, output_graph_path, output_depth_path):
    """
    WikipediaのXMLダンプを解析し、記事タイトル、リンク、カテゴリを抽出し、グラフを構築して深さを計算します。
    Parses a Wikipedia XML dump to extract article titles, links, and categories, builds a graph, and calculates depth.

    Args:
        wiki_xml_path (str): jawiki-latest-pages-articles.xml ファイルへのパス。
        output_graph_path (str): 構築したグラフを保存するパス (GML形式)。
        output_depth_path (str): 計算した深さ情報を保存するパス (JSON形式)。
    """
    print(f'"{wiki_xml_path}" からタイトル、リンク、カテゴリの抽出を開始します...')
    
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

    # --- First Pass: Collect all valid article titles ---
    print("--- 最初のパス: 有効な記事タイトルを収集中 ---")
    valid_titles = set()
    context_first_pass = etree.iterparse(wiki_xml_path, tag='{http://www.mediawiki.org/xml/export-0.11/}page', events=('end',))
    for event, elem in context_first_pass:
        title_elem = elem.find('mw:title', namespaces=ns)
        text_elem = elem.find('mw:revision/mw:text', namespaces=ns)
        
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            
            # 特別ページやリダイレクトはスキップ
            # 名前空間プレフィックスを持つページはスキップ (例: Help:, User:, Template:)
            # ただし、Category: はノードとして扱うためスキップしない
            if ((':' in title and not (title.startswith('Category:') or title.startswith('カテゴリ:')))
               or (text_elem is not None and text_elem.text and text_elem.text.strip().startswith('#REDIRECT'))):
                pass # スキップ対象だが、ここでは何もしない (valid_titlesに追加しない)
            else:
                valid_titles.add(title)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    print(f"--- 最初のパス完了。有効な記事タイトル数: {len(valid_titles)} ---")

    # --- Second Pass: Build graph using valid titles ---
    print("--- 2番目のパス: グラフを構築中 ---")
    G = nx.DiGraph() # 有向グラフ
    
    page_count = 0
    context_second_pass = etree.iterparse(wiki_xml_path, tag='{http://www.mediawiki.org/xml/export-0.11/}page', events=('end',))

    for event, elem in context_second_pass:
        title_elem = elem.find('mw:title', namespaces=ns)
        text_elem = elem.find('mw:revision/mw:text', namespaces=ns)
        
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            
            # 有効なタイトルのみを処理
            if title not in valid_titles:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                continue

            # ノードとして追加
            node_type = 'article'
            if title.startswith('Category:') or title.startswith('カテゴリ:'):
                node_type = 'category'
            G.add_node(title, type=node_type)

            if text_elem is not None and text_elem.text:
                text = text_elem.text

                # 内部リンクの抽出と重み付け
                for match in INTERNAL_LINK_PATTERN.finditer(text):
                    target_title = match.group(1).strip()
                    if target_title and target_title != title and target_title in valid_titles:
                        if G.has_edge(title, target_title):
                            G[title][target_title]['weight'] += 1
                        else:
                            G.add_edge(title, target_title, type='internal_link', weight=1)
                
                # カテゴリリンクの抽出（重みは1で固定）
                for match in CATEGORY_LINK_PATTERN.finditer(text):
                    category_name = (match.group(1) + ':' + match.group(2)).strip()
                    if category_name and category_name != title and category_name in valid_titles:
                        if not G.has_edge(title, category_name):
                             G.add_edge(title, category_name, type='is_member_of', weight=1)
                        # カテゴリから親カテゴリへのリンク (カテゴリページの場合)
                        if node_type == 'category':
                            if not G.has_edge(title, category_name):
                                G.add_edge(title, category_name, type='is_sub_category_of', weight=1)

            page_count += 1
            if page_count % 1000 == 0:
                print(f'... {page_count} ページを処理中。ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}')

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print(f"抽出完了。合計ページ数: {page_count}。最終ノード数: {G.number_of_nodes()}, 最終エッジ数: {G.number_of_edges()}")

    # 深さの計算
    print("グラフの深さを計算中...")
    depths = calculate_graph_depth(G)
    print("深さの計算が完了しました。")

    # グラフと深さの保存
    output_dir = os.path.dirname(output_graph_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # JSON形式でグラフを保存
    graph_data = nx.node_link_data(G, edges="links")
    with open(output_graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"グラフを '{output_graph_path}' にJSON形式で保存しました。")

    with open(output_depth_path, 'w', encoding='utf-8') as f:
        json.dump(depths, f, ensure_ascii=False, indent=2)
    print(f"深さデータを '{output_depth_path}' に保存しました。")

def calculate_graph_depth(G):
    """
    グラフ内の各ノードの深さを計算します。
    Calculates the depth of each node in the graph.

    Args:
        G (nx.DiGraph): 深さを計算する対象のグラフ。

    Returns:
        dict: 各ノードのタイトルをキーとし、深さを値とする辞書。
    """
    depths = {}
    
    # ルートカテゴリの特定 (親カテゴリを持たないカテゴリ)
    root_categories = [node for node, in_degree in G.in_degree() if in_degree == 0 and G.nodes[node]['type'] == 'category']
    
    # ルートカテゴリからの最短パスをBFSで計算
    # 複数のルートがある場合、最も浅い深さを採用
    for root in root_categories:
        for node, depth in nx.shortest_path_length(G, source=root).items():
            if node not in depths or depth < depths[node]:
                depths[node] = depth

    # 記事の深さは、その記事が属する最も浅いカテゴリの深さ + 1 とする
    # カテゴリに属さない記事は深さ0とするか、別途処理
    for node in G.nodes():
        if G.nodes[node]['type'] == 'article':
            article_depth = float('inf')
            # 記事が属するカテゴリを探す
            for _, target_node in G.out_edges(node):
                if G.nodes[target_node]['type'] == 'category' and target_node in depths:
                    article_depth = min(article_depth, depths[target_node] + 1)
            if article_depth == float('inf'): # どのカテゴリにも属さない記事
                depths[node] = 0 # または別のデフォルト値
            else:
                depths[node] = article_depth
        elif node not in depths: # どのルートカテゴリからも到達できないカテゴリ
            depths[node] = 0 # または別のデフォルト値

    return depths