import sys
import pandas as pd
import math
sys.path.append('../Network/')
import readNetwork as rn

# NGEV配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# nodes: Pandas: index, theta_i^o, q_d^o(demand); 
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o;
# node_order: [[downstream order], [upstream order]]
def NGEV(nodes, links, node_order):

    max_dbl = sys.float_info.max
    nodes['exp_cost'] = max_dbl
    nodes['exp_cost'][node_order[0][0]] = 0.0
    nodes['NGEV_flow'] = 0.0
    links['percent'] = 0.0
    links['NGEV_flow'] = 0.0

    # リンクを終点ノード順にソート
    term_order_links = links.sort_values('term_node')
    # forward関数を取得
    term_forward = make_forward(nodes, links, 'term_node')


    # 起点ノードから順に期待最小費用を計算
    for i in node_order[0][1:]:

        exp_sum = 0
        if i == nodes.index[-1]:
            link_set = links[term_forward[i]:len(links)]
        else:
            link_set = links[term_forward[i]:term_forward[i+1]]

        for index, link in link_set.iterrows():
            exp_sum += link['alpha'] * math.exp(-nodes['theta'][i] * ( link['free_flow_time'] + nodes['exp_cost'][link['init_node']] ))
    
        # print(exp_sum)
        nodes['exp_cost'][i] = - math.log(exp_sum)/nodes['theta'][i]





    # リンクを起点ノード順にソート
    links.sort_values('init_node')
    init_forward = make_forward(nodes, links, 'init_node')
        
    # 終点ノードから順にフローを計算
    for i in node_order[1]:

        # 下流側からのflow
        sum_flow = 0.0
        if i == nodes.index[-1]:
            link_set = links[init_forward[i]:len(links)]
        else:
            link_set = links[init_forward[i]:init_forward[i+1]]
        for index, link in link_set.iterrows():
            sum_flow += link['NGEV_flow']

        # ノードフローを計算
        nodes['NGEV_flow'][i] = sum_flow + nodes['demand'][i]

        # 上流リンク条件付選択確率を計算
        if i == nodes.index[-1]:
            link_set = term_order_links[term_forward[i]:len(links)]
        else:
            link_set = term_order_links[term_forward[i]:term_forward[i+1]]
        for index, link in link_set.iterrows():
            links['percent'][index] = link['alpha'] * math.exp( - nodes['theta'][i] * ( link['free_flow_time'] + nodes['exp_cost'][link['init_node']] ) ) / math.exp(- nodes['theta'][i] * nodes['exp_cost'][i])
            links['NGEV_flow'][index] = nodes['NGEV_flow'][i] * links['percent'][index]

    return 0



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




if __name__ == '__main__':

    import os
    import numpy as np

    root = os.path.dirname(os.path.abspath('.'))
    root = root + '\..\_sampleData\Sample'

    # ノード情報を追加
    nodes = rn.read_node(root + '\Sample_node.tntp')
    keys = nodes.columns
    nodes.drop(keys, axis=1, inplace=True)
    nodes['theta'] = 5.0
    nodes['demand'] = 0.0
    nodes['demand'][4] = 10.0

    # リンク情報を追加
    links = rn.read_net(root + '\Sample_net.tntp')
    keys = links.columns
    for key in keys:
        if key == 'init_node' or key == 'term_node' or key == 'free_flow_time':
            continue
        else:
            links.drop(key, axis=1, inplace=True)
    links['alpha'] = 1.0


    down_order = make_node_downstream_order(nodes, links, 1)
    # print(down_order)
    up_order = make_node_upstream_order(nodes, links)
    # print(up_order)

    NGEV(nodes, links, [down_order, up_order])
    
    print(nodes)
    print('\n')
    print(links)