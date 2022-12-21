

# forward関数を作成する関数(整列済みの nodes, links を input)
def make_forward(nodes, links, keyword):

    # ノード別リンク開始インデックスを格納するリスト
    # forward: {node: link_index}
    forward = dict(zip(nodes.index, [0 for i in range(len(nodes))]))
    k = 0
    now_node = nodes.index[k]
    for i in links.index:
        while links[keyword][i] > now_node:
            k += 1
            now_node = nodes.index[k]
            forward[now_node] = i
    # 終点ノードでは，仮想的なリンク番号(リンク数)を付与
    k += 1
    while k < len(nodes):
        now_node = nodes.index[k]
        forward[now_node] = len(links)
        k += 1

    return forward


# upstream orderを計算するメソッド(上流向きノード順)
def make_node_upstream_order(nodes, links):

    # ノードをインデックス順にソート
    nodes.sort_index()
    # リンクを始点ノード順にソート
    links.sort_values('init_node')
    # forward関数を取得
    forward = make_forward(nodes, links, 'init_node')

    # 結果格納リスト
    upstream_order = []

    # あるノードが探索済みかを示す辞書(0: 未探索，1: 探索済み)
    judged = dict(zip(nodes.index, [False for i in range(len(nodes))]))

    # 未探索のノードリスト
    non_search_index = list(nodes.index)
    # print(non_search_index)

    while len(non_search_index) > 0:
        remove_nodes = []
        for i in non_search_index:
            if i == nodes.index[-1]:
                link_set = links[forward[i]:len(links)]
            else:
                link_set = links[forward[i]:forward[i+1]]
            # print(link_set)

            ok = True
            for j in link_set.index:
                if judged[link_set['term_node'][j]] == False:
                    ok = False

            if ok:
                judged[i] = True
                upstream_order.append(i)
                remove_nodes.append(i)

        for i in remove_nodes:
            non_search_index.remove(i)

    return upstream_order


# downstream orderを計算するメソッド
def make_node_downstream_order(nodes, links, origin_node):

    # ノードをインデックス順にソート
    nodes.sort_index()
    # リンクを終点ノード順にソート
    links.sort_values('term_node')
    # forward関数を取得
    forward = make_forward(nodes, links, 'term_node')
    # print(forward)

    # 結果格納リスト
    downstream_order = [origin_node]

    # あるノードが探索済みかを示す辞書(0: 未探索，1: 探索済み)
    judged = dict(zip(nodes.index, [False for i in range(len(nodes))]))
    judged[origin_node] = True

    # 未探索のノードリスト
    non_search_index = list(nodes.index)
    non_search_index.remove(origin_node)
    # print(non_search_index)

    while len(non_search_index) > 0:
        remove_nodes = []
        for i in non_search_index:

            if i == nodes.index[-1]:
                link_set = links[forward[i]:len(links)]
            else:
                link_set = links[forward[i]:forward[i+1]]

            ok = True
            for j in link_set.index:
                if judged[link_set['init_node'][j]] == False:
                    ok = False

            if ok:
                judged[i] = True
                downstream_order.append(i)
                remove_nodes.append(i)

        for i in remove_nodes:
            non_search_index.remove(i)

    return downstream_order
