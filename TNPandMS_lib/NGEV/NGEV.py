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
# cost_name: links の何という変数を cost として代入するか(default = 'free_flow_time') 
def NGEV(nodes, links, node_order, cost_name = 'free_flow_time'):

    # print('Start NGEV')

    max_dbl = sys.float_info.max
    max_exp_dbl = math.log(sys.float_info.max)
    nodes['exp_cost'] = max_dbl
    nodes['exp_cost'][node_order[0][0]] = 0.0
    nodes['NGEV_flow'] = 0.0
    links['percent'] = 0.0
    links['NGEV_flow'] = 0.0


    # 起点ノードから順に期待最小費用を計算
    for i in node_order[0][1:]:

        exp_sum = 0

        link_set = links[links['term_node'] == i]

        # print('term_node = ', i)
        # print(link_set)


        if len(link_set) == 0:
            continue

        for index, link in link_set.iterrows():
            # print(nodes['exp_cost'][link['init_node']] + link[cost_name])
            # if -nodes['theta'][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]) > max_exp_dbl:
            #     exp_sum = max_dbl
            #     continue
            exp_sum += link['alpha'] * math.exp(-nodes['theta'][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]))
    
        if exp_sum == 0:
            nodes.loc[i, 'exp_cost'] = max_dbl
        else:
            nodes.loc[i, 'exp_cost'] = - math.log(exp_sum)/nodes['theta'][i]

        # print(nodes)
        # print('exp_sum = ', exp_sum)
        # print('exp_cost = ', nodes.loc[i, 'exp_cost'])

    # print(nodes)

        
    # 終点ノードから順にフローを計算
    for i in node_order[1]:

        # 下流側からのflow
        sum_flow = 0.0

        link_set = links[links['init_node'] == i]

        for index, link in link_set.iterrows():
            sum_flow += link['NGEV_flow']

        # ノードフローを計算
        nodes.loc[i, 'NGEV_flow'] = sum_flow + nodes['demand'][i]
        # print(nodes['demand'][i])
        # print('node', i, ': sum_flow = ', nodes.loc[i, 'NGEV_flow'])

        # 上流リンク条件付選択確率を計算
        link_set = links[links['term_node'] == i]
        if math.exp(- nodes['theta'][i] * nodes['exp_cost'][i]) == 0.0:
            min_cost = max_dbl
            k = 0
            for index, link in link_set.iterrows():
                temp_cost = link[cost_name] + nodes['exp_cost'][link['init_node']]
                # print(temp_cost, min_cost)
                if temp_cost < min_cost:
                    min_cost = temp_cost
                    k = index
            for index, link in link_set.iterrows():

                if k == 0:
                    links.loc[index, 'percent'] = 1.0/len(link_set)
                elif index == k:
                    links.loc[index, 'percent'] = 1.0
                else:
                    links.loc[index, 'percent'] = 0
        else:
            for index, link in link_set.iterrows():
                links.loc[index, 'percent'] = link['alpha'] * math.exp( - nodes['theta'][i] * ( link[cost_name] + nodes['exp_cost'][link['init_node']] ) ) / math.exp(- nodes['theta'][i] * nodes['exp_cost'][i])
        for index, link in link_set.iterrows():
            links.loc[index, 'NGEV_flow'] = nodes['NGEV_flow'][i] * links['percent'][index]

    links.sort_index()

    return 0


# NGEV-CC配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# https://drive.google.com/file/d/19siddz0k3gIzzLCfEeQHc9OIGSd3_gav/view：式(27)-(29)の(29)が等式制約ver.
# B: ndarray(|*|×|リンク|): 容量制約条件の係数行列
# b: ndarray(|*|): 容量制約条件の右辺
# nodes: Pandas: index, theta_i^o, q_d^o(demand);
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o; (indexの順にソート済みのものを入れる)
# node_order: [[downstream order], [upstream order]]
def NGEV_CC_equal(B, b, nodes, links, node_order):


    # 勾配関数
    def nbl_func(now_sol):
        
        # 現在の各リンクコストを計算し，costとしてlinksに代入
        cost = np.array([list(links['free_flow_time'])]) + now_sol @ B
        for i in range(len(links)):
            links.loc[links.index[i], 'cost'] = cost[0][i]

        # cost を基に，NGEV配分を計算
        NGEV(nodes, links, node_order, cost_name='cost')
        # print(links)
        nodes.drop('exp_cost', axis=1, inplace=True)
        nodes.drop('NGEV_flow', axis=1, inplace=True)
        links.drop('percent', axis=1, inplace=True)
        # print(nodes)

        # 勾配を計算
        now_flow = np.array([list(links['NGEV_flow'])])
        nbl = (B @ now_flow.T).T[0] - b
        # minに合わせるために符号を逆に
        nbl = -nbl
        links.drop('NGEV_flow', axis=1, inplace=True)
        # print(nbl)

        return nbl

    def obj_func(now_sol):

        # 現在の各リンクコストを計算し，costとしてlinksに代入
        cost = np.array([list(links['free_flow_time'])]) + now_sol @ B
        for i in range(len(links)):
            links.loc[links.index[i], 'cost'] = cost[0][i]

        # cost を基に，NGEV配分を計算
        NGEV(nodes, links, node_order, cost_name='cost')
        nodes.drop('NGEV_flow', axis=1, inplace=True)
        links.drop('percent', axis=1, inplace=True)
        links.drop('NGEV_flow', axis=1, inplace=True)
        # print(links)
        # print(nodes)

        # 目的関数を計算
        exp_cost = np.array([list(nodes['exp_cost'])])
        demand = np.array([list(nodes['demand'])])
        obj = exp_cost @ demand.T - now_sol @ b
        # minに合わせるために符号を逆に
        obj = -obj
        nodes.drop('exp_cost', axis=1, inplace=True)
        # print(obj)

        return obj[0][0]


    # linksの準備
    links['now_sol'] = 0.0
    links['cost'] = list(links['free_flow_time'])

    # 初期解の設定
    sol_init = np.array(list(links['now_sol']))

    fista = ag.FISTA_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_lips_init(0.1)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.03)
    # fista.set_output_iter(1)
    fista.exect_FISTA_back()

    # print('\n\n')

    # print('sol: ', fista.sol)
    # print('sol_obj: ', fista.sol_obj)
    # print('iteration: ', fista.iter)
    # print('elapsed_time: ', fista.time)
    # print('num_call_nabla: ', fista.num_call_nbl)
    # print('num_call_obj: ', fista.num_call_obj)
    # def nbl_func(now_sol, nodes, links, node_order):

    # 現在の各リンクコストを計算し，costとしてlinksに代入
    links['price'] = fista.sol
    cost = np.array([list(links['free_flow_time'])]) + fista.sol @ B
    for i in range(len(links)):
        links.loc[links.index[i], 'cost'] = cost[0][i]
    NGEV(nodes, links, node_order, cost_name='cost')
    # print(nodes)
    # print(links)


    return 0
        





if __name__ == '__main__':

    import os
    import numpy as np
    from scipy import sparse

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample', 'virtual_net', 'user', '0')
    # print(root)

    # ノード情報を追加
    nodes = rn.read_node(root + '\Sample_vir_node.tntp')
    keys = nodes.columns
    # print(keys)
    nodes.drop(keys, axis=1, inplace=True)
    nodes['theta'] = 1.0
    nodes['demand'] = 0.0
    nodes['demand'][8] = 10.0

    # リンク情報を追加
    links = rn.read_net(root + '\Sample_vir_net.tntp')
    # print(links)
    keys = links.columns
    for key in keys:
        if key == 'init_node' or key == 'term_node' or key == 'free_flow_time' or key == 'capacity':
            continue
        else:
            links.drop(key, axis=1, inplace=True)
    links['alpha'] = 1.0

    # print(nodes)
    # print(links.loc[100:])

    down_order = GEVsub.make_node_downstream_order(nodes, links, 1)
    # print(down_order)
    up_order = GEVsub.make_node_upstream_order(nodes, links)
    # print(up_order)

    # NGEV(nodes, links, [down_order, up_order])

    B = sparse.lil_matrix((len(links), len(links)))
    for i in range(len(links)):
        B[i, i] = 1.0
    b = np.ones(len(links))*5.0
    # NGEV_CC_equal(B, b, nodes, links, [down_order, up_order])

    NGEV(nodes, links, [down_order, up_order], cost_name='free_flow_time')
    
    print(nodes)
    print('\n')
    print(links[:50])
    print(links[50:100])
    print(links[100:])
