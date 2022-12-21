import sys
import pandas as pd
import math
import NGEV_sub as GEVsub
sys.path.append('../Network/')
import readNetwork as rn
sys.path.append('../optimizationProgram/')
import accelGradient as ag

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
    term_forward = GEVsub.make_forward(nodes, links, 'term_node')


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
        nodes.loc[i, 'exp_cost'] = - math.log(exp_sum)/nodes['theta'][i]


    # リンクを起点ノード順にソート
    links.sort_values('init_node')
    init_forward = GEVsub.make_forward(nodes, links, 'init_node')
        
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
        nodes.loc[i, 'NGEV_flow'] = sum_flow + nodes['demand'][i]

        # 上流リンク条件付選択確率を計算
        if i == nodes.index[-1]:
            link_set = term_order_links[term_forward[i]:len(links)]
        else:
            link_set = term_order_links[term_forward[i]:term_forward[i+1]]
        for index, link in link_set.iterrows():
            links.loc[index, 'percent'] = link['alpha'] * math.exp( - nodes['theta'][i] * ( link['free_flow_time'] + nodes['exp_cost'][link['init_node']] ) ) / math.exp(- nodes['theta'][i] * nodes['exp_cost'][i])
            links.loc[index, 'NGEV_flow'] = nodes['NGEV_flow'][i] * links['percent'][index]

    return 0


# NGEV-CC配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# https://drive.google.com/file/d/19siddz0k3gIzzLCfEeQHc9OIGSd3_gav/view：式(27)-(29)の(29)が等式制約ver.
# B: ndarray(|*|×|リンク|): 容量制約条件の係数行列
# b: ndarray(|*|): 容量制約条件の右辺
# nodes: Pandas: index, theta_i^o, q_d^o(demand);
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o;
# node_order: [[downstream order], [upstream order]]
def NGEV_CC_equal(B, b, nodes, links, node_order):

    max_dbl = sys.float_info.max
    nodes['exp_cost'] = max_dbl
    nodes['exp_cost'][node_order[0][0]] = 0.0
    nodes['NGEV-CC_flow'] = 0.0
    links['percent'] = 0.0
    links['NGEV-CC_flow'] = 0.0





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


    down_order = GEVsub.make_node_downstream_order(nodes, links, 1)
    # print(down_order)
    up_order = GEVsub.make_node_upstream_order(nodes, links)
    # print(up_order)

    NGEV(nodes, links, [down_order, up_order])
    
    print(nodes)
    print('\n')
    print(links)
