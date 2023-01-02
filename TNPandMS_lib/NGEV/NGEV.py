import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
import pandas as pd
import numpy as np
import time
import math
from scipy import sparse
import NGEV_sub as GEVsub
import readNetwork as rn
import accelGradient as ag
import MSA as msa
import readSparseMat as rsm
import linprog as lp



# NGEV配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# nodes: Pandas: index, theta_i^o, q_d^o(demand); 
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o;
# node_order: [[downstream order], [upstream order]]
# alpha_name: links の何という変数を alpha として用いるか(default = 'alpha')
# theta_name: nodes の何という変数を theta として用いるか(default = 'theta')
# cost_name: links の何という変数を cost として代入するか(default = 'free_flow_time') 
def NGEV(nodes, links, node_order, alpha_name = 'alpha', theta_name = 'theta', cost_name = 'free_flow_time'):

    max_dbl = sys.float_info.max
    max_exp_dbl = math.log(sys.float_info.max)
    nodes['exp_cost'] = max_dbl
    nodes['exp_cost'][node_order[0][0]] = 0.0
    nodes['NGEV_flow'] = 0.0
    links['percent'] = 0.0
    links['NGEV_flow'] = 0.0

    start_time = time.process_time()


    # 起点ノードから順に期待最小費用を計算
    for i in node_order[0][1:]:

        exp_sum = 0

        link_set = links[links['term_node'] == i]

        if len(link_set) == 0:
            continue

        for index, link in link_set.iterrows():
            if -nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]) > max_exp_dbl:
                exp_sum = max_dbl
                break
            elif link[alpha_name] * math.exp(-nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']])) > max_dbl - exp_sum:
                exp_sum = max_dbl
                break
            exp_sum += link[alpha_name] * math.exp(-nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]))
    
        if exp_sum == 0:
            nodes.loc[i, 'exp_cost'] = max_dbl
        else:
            nodes.loc[i, 'exp_cost'] = - math.log(exp_sum)/nodes[theta_name][i]

        
    # 終点ノードから順にフローを計算
    for i in node_order[1]:

        # 下流側からのflow
        sum_flow = 0.0

        link_set = links[links['init_node'] == i]

        for index, link in link_set.iterrows():
            sum_flow += link['NGEV_flow']

        # ノードフローを計算
        nodes.loc[i, 'NGEV_flow'] = sum_flow + nodes['demand'][i]

        # 上流リンク条件付選択確率を計算
        link_set = links[links['term_node'] == i]
        if math.exp(- nodes[theta_name][i] * nodes['exp_cost'][i]) == 0.0:
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
            sum_per = 0.0
            for index, link in link_set.iterrows():
                max_index = []
                max_cost = 0.0
                if  - nodes[theta_name][i] * ( link[cost_name] + nodes['exp_cost'][link['init_node']] ) > max_exp_dbl:
                    if max_cost < - nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]):
                        max_index.append(index)
                        max_cost = - nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']])
                else:
                    links.loc[index, 'percent'] = link[alpha_name] * math.exp( - nodes[theta_name][i] * ( link[cost_name] + nodes['exp_cost'][link['init_node']] ) ) / math.exp(- nodes[theta_name][i] * nodes['exp_cost'][i])
                    sum_per += links.loc[index, 'percent']
            if len(max_index) > 0:
                links.loc[max_index[-1], 'percent'] = (1.0 - sum_per)
        for index, link in link_set.iterrows():
            links.loc[index, 'NGEV_flow'] = nodes['NGEV_flow'][i] * links['percent'][index]

    end_time = time.process_time()

    return end_time - start_time


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
    fista.set_conv_judge(0.1)
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
        

# NGEV-CC配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# https://drive.google.com/file/d/19siddz0k3gIzzLCfEeQHc9OIGSd3_gav/view：式(27)-(29)の(29)が等式制約ver.
# B: ndarray(|*|×|リンク|): 容量制約条件の係数行列
# b: ndarray(|*|): 容量制約条件の右辺
# nodes: Pandas: index, theta_i^o, q_d^o(demand);
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o; (indexの順にソート済みのものを入れる)
# node_order: [[downstream order], [upstream order]]
def NGEV_CC(B, b, nodes, links, node_order):

    # 勾配関数
    def nbl_func(now_sol):

        # 現在の各リンクコストを計算し，costとしてlinksに代入
        cost = np.array([list(links['free_flow_time'])]) + now_sol @ B
        links['cost'] = cost[0]

        # cost を基に，NGEV配分を計算
        temp_time = NGEV(nodes, links, node_order, cost_name='cost')
        # nodes.drop('exp_cost', axis=1, inplace=True)
        # nodes.drop('NGEV_flow', axis=1, inplace=True)
        # links.drop('percent', axis=1, inplace=True)

        # 勾配を計算
        now_flow = np.array([list(links['NGEV_flow'])])
        nbl = (B @ now_flow.T).T[0] - b
        # minに合わせるために符号を逆に
        nbl = -nbl
        # links.drop('NGEV_flow', axis=1, inplace=True)
        # print(nbl)

        return nbl, temp_time, temp_time

    def obj_func(now_sol):

        # 現在の各リンクコストを計算し，costとしてlinksに代入
        cost = np.array([list(links['free_flow_time'])]) + now_sol @ B
        links['cost'] = cost[0]

        # cost を基に，NGEV配分を計算
        temp_time = NGEV(nodes, links, node_order, cost_name='cost')
        # nodes.drop('NGEV_flow', axis=1, inplace=True)
        # links.drop('percent', axis=1, inplace=True)
        # links.drop('NGEV_flow', axis=1, inplace=True)
        # print(links)
        # print(nodes)

        # 目的関数を計算
        exp_cost = np.array([list(nodes['exp_cost'])])
        demand = np.array([list(nodes['demand'])])
        obj = exp_cost @ demand.T - now_sol @ b
        # minに合わせるために符号を逆に
        obj = -obj
        # nodes.drop('exp_cost', axis=1, inplace=True)
        # print(obj)

        return obj[0][0], temp_time, temp_time

    def proj_func(now_sol):

        start_time = time.process_time()

        for i in range(len(now_sol)):
            if now_sol[i] < 0.0:
                now_sol[i] = 0.0

        end_time = time.process_time()

        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):

        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -min(now_nbl)
        end_time = time.process_time()

        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)
    

    # linksの準備
    links['now_sol'] = 0.0
    links['cost'] = list(links['free_flow_time'])

    # 初期解の設定
    sol_init = np.zeros(B.shape[0])

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(0.1)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.01)
    fista.set_output_iter(1)
    fista.exect_FISTA_proj_back()

    # print('\n\n')

    # print('sol: ', fista.sol)
    # print('sol_obj: ', fista.sol_obj)
    # print('iteration: ', fista.iter)
    # print('elapsed_time: ', fista.time)
    # print('num_call_nabla: ', fista.num_call_nbl)
    # print('num_call_obj: ', fista.num_call_obj)
    # def nbl_func(now_sol, nodes, links, node_order):

    # 現在の各リンクコストを計算し，costとしてlinksに代入
    cost = np.array([list(links['free_flow_time'])]) + fista.sol @ B
    links['cost'] = cost[0]
    NGEV(nodes, links, node_order, cost_name='cost')
    # print(nodes)
    # print(links)

    return fista.sol, fista.para_time, fista.total_time


# NGEV-CC配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# https://drive.google.com/file/d/19siddz0k3gIzzLCfEeQHc9OIGSd3_gav/view：式(27)-(29)の(29)が等式制約ver.
# B: ndarray(|*|×|リンク|): 容量制約条件の係数行列
# b: ndarray(|*|): 容量制約条件の右辺
# nodes: Pandas: index, theta_i^o, q_d^o(demand);
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o; (indexの順にソート済みのものを入れる)
# node_order: [[downstream order], [upstream order]]
def NGEV_CC_TNP(TNP_constMat, capacity, veh_nodes, veh_links, veh_trips, fft_name = 'free_flow_time'):

    # 勾配関数
    def nbl_func(now_sol_TNP):        

        para_time = 0.0
        total_time = 0.0

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]

        temp_para_time = []
        start_time = time.process_time()

        nbl_TNP = -capacity

        end_time = time.process_time()
        para_time += end_time - start_time
        total_time += end_time - start_time
        

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num][fft_name])]) + now_sol_TNP @ TNP_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time


            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.array([list(veh_links[veh_num]['NGEV_flow'])])
                nbl_TNP += (TNP_constMat[veh_num] @ now_flow.T).T[0]

                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        para_time += max(temp_para_time)

        # min に合わせるために符号を逆に
        nbl_TNP = -nbl_TNP

        return nbl_TNP, para_time, total_time


    def obj_func(now_sol_TNP):

        para_time = 0.0
        total_time = 0.0

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]

        temp_para_time = []

        obj = 0.0

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num][fft_name])]) + now_sol_TNP @ TNP_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                exp_cost = np.array([list(veh_nodes[veh_num]['exp_cost'])])
                demand = np.array([list(veh_nodes[veh_num]['demand'])])
                start_time = time.process_time()
                obj += (exp_cost @ demand.T)[0][0]
                end_time = time.process_time()
                para_time += end_time - start_time 
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        # 目的関数を計算
        obj -= now_sol_TNP @ capacity
        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    def proj_func(now_sol):

        start_time = time.process_time()

        for i in range(len(now_sol)):
            if now_sol[i] < 0.0:
                now_sol[i] = 0.0

        end_time = time.process_time()

        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):

        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -min(now_nbl)
        # conv = - (now_sol @ now_nbl)
        end_time = time.process_time()

        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)

    print('start NGEV_CC_TNP')

    # 初期解の設定
    num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
    sol_init = np.zeros(num_TNPconst)

    total_flow = sum([sum(list(veh_trips[list(veh_trips.keys())[0]][origin_node].values())) for origin_node in veh_trips[list(veh_trips.keys())[0]].keys()])
    max_cost = max(list(veh_links[list(veh_links.keys())[0]]['free_flow_time']))

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(total_flow/num_TNPconst*max_cost)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.1)
    fista.set_output_iter(10)
    # fista.set_output_root(output_root)
    fista.exect_FISTA_proj_back()

    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', fista.sol_obj)
    print('iteration: ', fista.iter)
    print('para_time: ', fista.para_time)
    print('total_time: ', fista.total_time)
    # print('num_call_nabla: ', fista.num_call_nbl)
    # print('num_call_obj: ', fista.num_call_obj)
    # print('num_call_proj: ', fista.num_call_proj)
    # print('num_call_conv: ', fista.num_call_conv)
    # print('output_data: ')
    # print(fista.output_data)

    return fista




# NGEV-CC配分アルゴリズム (no cycle)　※起点ノードはdownstream orderの最初のノード
# https://drive.google.com/file/d/19siddz0k3gIzzLCfEeQHc9OIGSd3_gav/view：式(27)-(29)の(29)が等式制約ver.
# B: ndarray(|*|×|リンク|): 容量制約条件の係数行列
# b: ndarray(|*|): 容量制約条件の右辺
# nodes: Pandas: index, theta_i^o, q_d^o(demand);
# links: Pandas: index, init_node, term_node, free_flow_time, alpha_{ij}^o; (indexの順にソート済みのものを入れる)
# node_order: [[downstream order], [upstream order]]
def NGEV_CC_MS(MSU_constMat, MS_capacity, user_nodes, user_links, user_trips, fft_name = 'free_flow_time'):

    # 勾配関数
    def nbl_func(now_sol_MS):

        para_time = 0.0
        total_time = 0.0

        num_MSconst = MSU_constMat[list(user_nodes.keys())[0]].shape[0]

        temp_para_time = []

        nbl_MS = -MS_capacity

        
        for user_num in user_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(user_links[user_num][fft_name])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in user_trips[user_num].keys():
                
                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.array([list(user_links[user_num]['NGEV_flow'])])
                nbl_MS += (MSU_constMat[user_num] @ now_flow.T).T[0]

                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)

        para_time += max(temp_para_time)

        # min に合わせるために符号を逆に
        nbl_MS = -nbl_MS

        return nbl_MS, para_time, total_time
        

    def obj_func(now_sol_MS):

        
        para_time = 0.0
        total_time = 0.0

        num_MSconst = MSU_constMat[list(user_nodes.keys())[0]].shape[0]

        temp_para_time = []

        obj = 0.0

        for user_num in user_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(user_links[user_num][fft_name])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in user_trips[user_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                exp_cost = np.array([list(user_nodes[user_num]['exp_cost'])])
                demand = np.array([list(user_nodes[user_num]['demand'])])
                start_time = time.process_time()
                obj += (exp_cost @ demand.T)[0][0]
                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)


        # 目的関数を計算
        obj -= now_sol_MS @ MS_capacity
        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    def proj_func(now_sol):

        start_time = time.process_time()

        for i in range(len(now_sol)):
            if now_sol[i] < 0.0:
                now_sol[i] = 0.0

        end_time = time.process_time()

        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):

        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -min(now_nbl)
        # conv = - (now_sol @ now_nbl)
        end_time = time.process_time()

        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)

    print('start NGEV_CC_MS')

    # 初期解の設定
    num_MSconst = MSU_constMat[list(user_nodes.keys())[0]].shape[0]
    sol_init = np.zeros(num_MSconst)
    
    total_flow = sum([sum(list(user_trips[list(user_trips.keys())[0]][origin_node].values())) for origin_node in user_trips[list(user_trips.keys())[0]].keys()])
    max_cost = max(list(user_links[list(user_links.keys())[0]]['free_flow_time']))

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(total_flow/num_MSconst*max_cost)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.1)
    fista.set_output_iter(10)
    # fista.set_output_root(output_root)
    fista.exect_FISTA_proj_back()

    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', fista.sol_obj)
    print('iteration: ', fista.iter)
    print('pararel_time: ', fista.para_time)
    print('total_time: ', fista.total_time)
    # print('num_call_nabla: ', fista.num_call_nbl)
    # print('num_call_obj: ', fista.num_call_obj)
    # print('num_call_proj: ', fista.num_call_proj)
    # print('num_call_conv: ', fista.num_call_conv)
    # print('output_data: ')
    # print(fista.output_data)

    return fista


def NGEV_TNPandMS_MSA(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root):

    def TNP_price_to_MS_capacity(TNP_price):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        MS_capacity = np.zeros(MSV_constMat[list(MSV_constMat.keys())[0]].shape[0])


        for veh_num in veh_nodes.keys():

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num]['free_flow_time'])]) + TNP_price @ TNP_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.array([list(veh_links[veh_num]['NGEV_flow'])])
                MS_capacity += (MSV_constMat[veh_num] @ now_flow.T).T[0]

                end_time = time.process_time()
                temp_para_time.append(end_time - start_time)
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        para_time += max(temp_para_time)

        return MS_capacity, para_time, total_time


    def TNP_price_to_sol(TNP_price, fft_name):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        now_flow = np.array([])
        

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num][fft_name])]) + TNP_price @ TNP_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time


            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.hstack([now_flow, np.array(list(veh_links[veh_num]['NGEV_flow']))])

                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        return now_flow, para_time, total_time



    def veh_sol_to_MS_capacity(veh_sol):

        para_time = 0.0
        total_time = 0.0

        MS_capacity = np.zeros(MSV_constMat[list(MSV_constMat.keys())[0]].shape[0])
        start_index = 0

        start_time = time.process_time()

        for veh_num in veh_nodes.keys():

            for origin_node in veh_trips[veh_num].keys():

                now_flow = np.array([list(veh_sol[start_index:start_index + len(veh_links[veh_num])])])
                # print(MSV_constMat[veh_num].shape)
                # print(now_flow.shape)
                MS_capacity += (MSV_constMat[veh_num] @ now_flow.T).T[0]
                start_index += len(veh_links[veh_num])

        end_time = time.process_time()
        para_time += end_time - start_time
        total_time += end_time - start_time

        return MS_capacity, para_time, total_time



    def MS_price_to_veh_fft(MS_price):
        
        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            fft_ms = np.array([list(veh_links[veh_num]['free_flow_time'])]) - MS_price @ MSV_constMat[veh_num]
            veh_links[veh_num]['fft_ms'] = fft_ms[0]

            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        para_time += max(temp_para_time)

        return para_time, total_time



    def dir_func(now_sol):

        para_time = 0.0
        total_time = 0.0

        [MS_capacity, temp_para_time, temp_total_time] = veh_sol_to_MS_capacity(now_sol)
        para_time += temp_para_time
        total_time += temp_total_time

        MS_fista = NGEV_CC_MS(MSU_constMat, MS_capacity, user_nodes, user_links, user_trips)
        MS_price = MS_fista.sol
        # para_time += MS_fista.para_time
        # total_time += MS_fista.total_time

        [temp_para_time, temp_total_time] = MS_price_to_veh_fft(MS_price)
        para_time += temp_para_time
        total_time += temp_total_time

        TNP_fista = NGEV_CC_TNP(TNP_constMat, np.array(list(TS_links['capacity'])), veh_nodes, veh_links, veh_trips, fft_name='fft_ms')
        [temp_sol, temp_para_time, temp_total_time] = TNP_price_to_sol(TNP_fista.sol, 'fft_ms')
        para_time += TNP_fista.para_time + temp_para_time
        total_time += TNP_fista.total_time + temp_total_time

        dir_vec = temp_sol - now_sol

        
        # print('temp_sol: ', temp_sol)
        # print('now_sol: ', now_sol)
        # print('dir_vec: ', dir_vec)

        # return dir_vec, para_time, total_time
        return dir_vec, para_time, total_time

    # 初期解を作成する関数
    def make_init_sol():

        init_sol = np.array([])

        for veh_num in veh_trips.keys():

            for origin_node in veh_trips[veh_num].keys():

                veh_links[veh_num]['now_flow'] = 0.0

                for dest_node in veh_trips[veh_num][origin_node].keys():

                    # print(origin_node, dest_node, veh_trips[veh_num][origin_node][dest_node])
                    link_set = veh_links[veh_num][(veh_links[veh_num]['init_node']==origin_node) & (veh_links[veh_num]['term_node']==dest_node)]
                    for index, link in link_set.iterrows():
                        veh_links[veh_num].loc[index, 'now_flow'] = veh_trips[veh_num][origin_node][dest_node]

                add_sol = np.array(list(veh_links[veh_num]['now_flow']))
                init_sol = np.hstack([init_sol, add_sol])
                veh_links[veh_num].drop('now_flow', axis=1, inplace=True)

        return init_sol

    # 目的関数
    def obj_func(now_sol):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        obj = 0.0

        # 車両側の目的関数値を計算
        for veh_num in veh_trips.keys():
            for origin_node in veh_trips[veh_num].keys():

                start_time = time.process_time()

                # 起点別リンクフローを設定
                num_links = len(veh_links[veh_num].index)
                veh_links[veh_num]['link_flow'] = now_sol[start_index:start_index + num_links]
                start_index += num_links

                # print(veh_links[veh_num][(veh_links[veh_num]['link_flow'] != 0.0) & (veh_links[veh_num]['term_node'] == 7)])

                # 起点別ノードフローを計算
                veh_nodes[veh_num]['node_flow'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    in_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]['link_flow']))
                    # out_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['init_node'] == node_num]['link_flow']))
                    veh_nodes[veh_num].loc[node_num, 'node_flow'] = in_flow
                # print(veh_nodes[veh_num])

                # print(veh_nodes[veh_num][veh_nodes[veh_num]['node_flow'] != 0.0])

                # 線形項を計算
                obj += np.array(list(veh_links[veh_num]['link_flow'])) @ np.array(list(veh_links[veh_num]['free_flow_time']))
                # print(obj)

                # エントロピー項を計算
                veh_nodes[veh_num]['log_term'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    if veh_nodes[veh_num]['node_flow'][node_num] == 0.0:
                        continue
                    link_set = veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]
                    # print(link_set)
                    for index, link in link_set.iterrows():
                        if link['link_flow'] == 0.0:
                            continue
                        # print(link['link_flow'])
                        # print(veh_nodes[veh_num]['node_flow'][node_num])
                        # print(link['link_flow'] * math.log(link['link_flow'] / veh_nodes[veh_num]['node_flow'][node_num]))
                        veh_nodes[veh_num].loc[node_num, 'log_term'] += link['link_flow'] * math.log(link['link_flow'] / (veh_nodes[veh_num]['node_flow'][node_num] * link['alpha']))
                    
                    # print(np.array(veh_nodes[veh_num]['log_term']))
                    # print(1.0/np.array(veh_nodes[veh_num]['theta']))

                # print(np.array(veh_nodes[veh_num]['log_term']) @ (1.0 / np.array(veh_nodes[veh_num]['theta'])))

                obj += np.array(veh_nodes[veh_num]['log_term']) @ (1.0 / np.array(veh_nodes[veh_num]['theta']))

                # print(veh_nodes[veh_num][veh_nodes[veh_num]['log_term'] != 0.0])
                # print(obj)

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                veh_links[veh_num].drop('link_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('node_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('log_term', axis=1, inplace=True)

        para_time += max(temp_para_time)

        # 利用者側の目的関数値を計算
        [MS_capacity, temp_para_time, temp_total_time] = veh_sol_to_MS_capacity(now_sol)
        para_time += temp_para_time 
        total_time += temp_total_time
        # print(MS_capacity)
        MSU_fista = NGEV_CC_MS(MSU_constMat, MS_capacity, user_nodes, user_links, user_trips)
        obj -= MSU_fista.sol_obj
        para_time += MSU_fista.para_time
        total_time += MSU_fista.total_time

        # print(obj)

        return obj, para_time, total_time

    # init_MS_price = np.zeros(MSV_constMat[list(veh_trips.keys())[0]].shape[0])
    # MS_price_to_veh_fft(init_MS_price)

    # TNP_price = np.zeros(TNP_constMat[0].shape[0])
    # [temp_sol, para_time, total_time] = TNP_price_to_sol(TNP_price, 'fft_ms')
    # print(temp_sol)

    # print(obj_func(temp_sol))
    

    init_sol = make_init_sol()
    # print(init_sol)

    veh_msa = msa.MSA()
    veh_msa.set_x_init(init_sol)
    veh_msa.set_obj_func(obj_func)
    veh_msa.set_dir_func(dir_func)
    veh_msa.set_conv_judge(0.1)
    veh_msa.set_output_iter(1)
    veh_msa.set_output_root(output_root)
    veh_msa.exect_MSA()


    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', veh_msa.sol_obj)
    print('iteration: ', veh_msa.iter)
    print('pararel_time: ', veh_msa.para_time)
    print('total_time: ', veh_msa.total_time)
    print('num_call_obj: ', veh_msa.num_call_obj)
    print('num_call_dir: ', veh_msa.num_call_dir)
    # print('output_data: ')
    # print(fista.output_data)

    return veh_msa



def NGEV_TNPandMS_FrankWolf(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, veh_incMat, user_nodes, user_links, user_trips, MSU_constMat, user_incMat, TS_links, output_root):

    # 勾配関数
    def nbl_func(now_sol):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        nbl = np.array([])
        

        # 車両側の勾配を計算
        for veh_num in veh_trips.keys():
            for origin_node in veh_trips[veh_num].keys():

                start_time = time.process_time()

                temp_nbl = np.array(list(veh_links[veh_num]['free_flow_time']))

                # 起点別リンクフローを設定
                num_links = len(veh_links[veh_num].index)
                veh_links[veh_num]['link_flow'] = now_sol[start_index:start_index + num_links]
                start_index += num_links

                # 起点別ノードフローを計算
                veh_nodes[veh_num]['node_flow'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    in_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]['link_flow']))
                    # out_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['init_node'] == node_num]['link_flow']))
                    veh_nodes[veh_num].loc[node_num, 'node_flow'] = in_flow

                # エントロピー項を計算
                veh_links[veh_num]['log_term'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    if veh_nodes[veh_num]['node_flow'][node_num] == 0.0:
                        continue
                    link_set = veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]
                    for index, link in link_set.iterrows():
                        if link['link_flow'] == 0.0:
                            continue
                        veh_links[veh_num].loc[index, 'log_term'] += math.log(link['link_flow'] / (veh_nodes[veh_num]['node_flow'][node_num] * link['alpha'])) / veh_nodes[veh_num]['theta'][node_num]
                    
                temp_nbl += np.array(list(veh_links[veh_num]['log_term']))
                nbl = np.hstack([nbl, temp_nbl])

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                veh_links[veh_num].drop('link_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('node_flow', axis=1, inplace=True)
                veh_links[veh_num].drop('log_term', axis=1, inplace=True)

                

        # 利用者側の勾配を計算
        for user_num in user_trips.keys():
            for origin_node in user_trips[user_num].keys():

                start_time = time.process_time()

                temp_nbl = np.array(list(user_links[user_num]['free_flow_time']))

                # 起点別リンクフローを設定
                num_links = len(user_links[user_num].index)
                user_links[user_num]['link_flow'] = now_sol[start_index:start_index + num_links]
                start_index += num_links

                # 起点別ノードフローを計算
                user_nodes[user_num]['node_flow'] = 0.0
                for node_num in user_nodes[user_num].index:
                    in_flow = sum(list(user_links[user_num][user_links[user_num]['term_node'] == node_num]['link_flow']))
                    user_nodes[user_num].loc[node_num, 'node_flow'] = in_flow

                # エントロピー項を計算
                user_links[user_num]['log_term'] = 0.0
                for node_num in user_nodes[user_num].index:
                    if user_nodes[user_num]['node_flow'][node_num] == 0.0:
                        continue
                    link_set = user_links[user_num][user_links[user_num]['term_node'] == node_num]
                    for index, link in link_set.iterrows():
                        if link['link_flow'] == 0.0:
                            continue
                        user_links[user_num].loc[index, 'log_term'] += math.log(link['link_flow'] / (user_nodes[user_num]['node_flow'][node_num] * link['alpha'])) / user_nodes[user_num]['theta'][node_num]
                    
                temp_nbl += np.array(list(user_links[user_num]['log_term']))
                nbl = np.hstack([nbl, temp_nbl])

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                user_links[user_num].drop('link_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('node_flow', axis=1, inplace=True)
                user_links[user_num].drop('log_term', axis=1, inplace=True)

        para_time += max(temp_para_time)

        return nbl, para_time, total_time




    def make_B_eq():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        num_var = 0
        num_const = 0
        for veh_num in veh_trips.keys():
            num_var += len(veh_trips[veh_num]) * veh_incMat[veh_num].shape[1]
            num_const += len(veh_trips[veh_num]) * veh_incMat[veh_num].shape[0]
        for user_num in user_trips.keys():
            num_var += len(user_trips[user_num]) * user_incMat[user_num].shape[1]
            num_const += len(user_trips[user_num]) * user_incMat[user_num].shape[0]

        start_row_index = 0
        start_col_index = 0

        row = np.array([])
        col = np.array([])
        data = np.array([])

        # 車両側のフロー保存則の行列を作成
        for veh_num in veh_incMat.keys():
            for origin_node in veh_trips[veh_num].keys():
                row = np.hstack([row, np.array(veh_incMat[veh_num].row + start_row_index)])
                col = np.hstack([col, np.array(veh_incMat[veh_num].col + start_col_index)])
                data = np.hstack([data, np.array(veh_incMat[veh_num].data)])

                start_row_index += veh_incMat[veh_num].shape[0]
                start_col_index += veh_incMat[veh_num].shape[1]

        # 利用者側のフロー保存則の行列を作成
        for user_num in user_incMat.keys():
            for origin_node in user_trips[user_num].keys():
                row = np.hstack([row, np.array(user_incMat[user_num].row + start_row_index)])
                col = np.hstack([col, np.array(user_incMat[user_num].col + start_col_index)])
                data = np.hstack([data, np.array(user_incMat[user_num].data)])

                start_row_index += user_incMat[user_num].shape[0]
                start_col_index += user_incMat[user_num].shape[1]

        # print(len(row))
        # print(len(col))
        # print(len(data))
        # print(num_var)
        # print(num_const)
        B_eq = sparse.csr_matrix((data, (row, col)), shape=(num_const, num_var))
        # print(B_eq.shape)

        return B_eq, para_time, total_time

    def make_b_eq():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        b_eq = np.array([])
        
        # 車両側のOD需要を追加
        for veh_num in veh_trips.keys():

            for origin_node in veh_trips[veh_num].keys():

                temp_b_eq = dict(zip(list(veh_nodes[veh_num].index), [0.0 for i in range(veh_incMat[veh_num].shape[0])]))
                # print(temp_b_eq)
                
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    temp_b_eq[origin_node] -= veh_trips[veh_num][origin_node][dest_node]
                    temp_b_eq[dest_node] = veh_trips[veh_num][origin_node][dest_node]

                temp_b_eq = np.array(list(temp_b_eq.values()))
                # print(temp_b_eq)

                b_eq = np.hstack([b_eq, temp_b_eq])

        
        # 利用者側のOD需要を追加
        for user_num in user_trips.keys():

            for origin_node in user_trips[user_num].keys():

                temp_b_eq = dict(zip(list(user_nodes[user_num].index), [0.0 for i in range(user_incMat[user_num].shape[0])]))
                # print(temp_b_eq)
                
                for dest_node in user_trips[user_num][origin_node].keys():
                    temp_b_eq[origin_node] -= user_trips[user_num][origin_node][dest_node]
                    temp_b_eq[dest_node] = user_trips[user_num][origin_node][dest_node]

                temp_b_eq = np.array(list(temp_b_eq.values()))
                # print(temp_b_eq)

                b_eq = np.hstack([b_eq, temp_b_eq])

        return b_eq, para_time, total_time


        
    def make_B():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        num_var = 0
        num_const = TNP_constMat[list(TNP_constMat.keys())[0]].shape[0] + MSV_constMat[list(MSV_constMat.keys())[0]].shape[0]
        for veh_num in veh_trips.keys():
            num_var += len(veh_trips[veh_num]) * veh_incMat[veh_num].shape[1]
            num_const += len(veh_trips[veh_num]) * veh_incMat[veh_num].shape[1]
        for user_num in user_trips.keys():
            num_var += len(user_trips[user_num]) * user_incMat[user_num].shape[1]
            num_const += len(user_trips[user_num]) * user_incMat[user_num].shape[1]
            

        # print(num_var)
        # print(TNP_constMat[list(TNP_constMat.keys())[0]].shape[0])
        # print(MSV_constMat[list(MSV_constMat.keys())[0]].shape[0])
        # print(num_const)

        start_TNProw_index = 0
        start_MSrow_index = TNP_constMat[list(TNP_constMat.keys())[0]].shape[0]
        start_nzero_index = TNP_constMat[list(TNP_constMat.keys())[0]].shape[0] + MSV_constMat[list(MSV_constMat.keys())[0]].shape[0]
        start_col_index = 0

        row = np.array([])
        col = np.array([])
        data = np.array([])

        # 車両側の制約条件の行列を作成
        for veh_num in veh_incMat.keys():
            for origin_node in veh_trips[veh_num].keys():
                # TNPの制約条件
                row = np.hstack([row, np.array(TNP_constMat[veh_num].row + start_TNProw_index)])
                col = np.hstack([col, np.array(TNP_constMat[veh_num].col + start_col_index)])
                data = np.hstack([data, np.array(TNP_constMat[veh_num].data)])
                # MSの制約条件
                row = np.hstack([row, np.array(MSV_constMat[veh_num].row + start_MSrow_index)])
                col = np.hstack([col, np.array(MSV_constMat[veh_num].col + start_col_index)])
                data = np.hstack([data, -np.array(MSV_constMat[veh_num].data)])

                start_col_index += veh_incMat[veh_num].shape[1]


        # 利用者側の制約条件の行列を作成
        for user_num in user_incMat.keys():
            for origin_node in user_trips[user_num].keys():
                # MSの制約条件
                row = np.hstack([row, np.array(MSU_constMat[user_num].row + start_MSrow_index)])
                col = np.hstack([col, np.array(MSU_constMat[user_num].col + start_col_index)])
                data = np.hstack([data, np.array(MSU_constMat[user_num].data)])

                start_col_index += MSU_constMat[user_num].shape[1]

        # 非負制約を追加
        row = np.hstack([row, np.array(list(range(num_var))) + start_nzero_index])
        col = np.hstack([col, np.array(list(range(num_var)))])
        data = np.hstack([data, np.ones(num_var)])

        # # print(len(row))
        # # print(len(col))
        # # print(len(data))
        # # print(num_var)
        # # print(num_const)
        B = sparse.csr_matrix((data, (row, col)), shape=(num_const, num_var))
        # print(B.shape)
        # print(B)

        return B, para_time, total_time

    def make_b():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        num_var = 0
        for veh_num in veh_trips.keys():
            num_var += len(veh_trips[veh_num]) * veh_incMat[veh_num].shape[1]
        for user_num in user_trips.keys():
            num_var += len(user_trips[user_num]) * user_incMat[user_num].shape[1]

        # TNP制約
        b = np.array(TS_links['capacity'])    
        # MS制約   
        b = np.hstack([b, np.zeros(MSV_constMat[list(MSV_constMat.keys())[0]].shape[0])])
        # 非負制約
        b = np.hstack([b, np.zeros(num_var)])

        # print(b.shape)

        return b, para_time, total_time



    def dir_func(now_sol):

        para_time = 0.0
        total_time = 0.0

        [now_nbl, temp_para_time, temp_total_time] = nbl_func(now_sol)

        [B_eq, temp_para_time, temp_total_time] = make_B_eq()
        [B, temp_para_time, temp_total_time] = make_B()
        [b_eq, temp_para_time, temp_total_time] = make_b_eq()
        [b, temp_para_time, temp_total_time] = make_b()

        [model, temp_sol] = lp.linprog(now_nbl, B, b, B_eq, b_eq)
        para_time += model.Runtime
        total_time += model.Runtime

        dir_vec = temp_sol - now_sol

        return dir_vec, para_time, total_time

    # 目的関数
    def obj_func(now_sol):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        obj = 0.0

        # 車両側の目的関数値を計算
        for veh_num in veh_trips.keys():
            for origin_node in veh_trips[veh_num].keys():

                start_time = time.process_time()

                # 起点別リンクフローを設定
                num_links = len(veh_links[veh_num].index)
                veh_links[veh_num]['link_flow'] = now_sol[start_index:start_index + num_links]
                start_index += num_links

                # 起点別ノードフローを計算
                veh_nodes[veh_num]['node_flow'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    in_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]['link_flow']))
                    # out_flow = sum(list(veh_links[veh_num][veh_links[veh_num]['init_node'] == node_num]['link_flow']))
                    veh_nodes[veh_num].loc[node_num, 'node_flow'] = in_flow

                # 線形項を計算
                obj += np.array(list(veh_links[veh_num]['link_flow'])) @ np.array(list(veh_links[veh_num]['free_flow_time']))
                # print(obj)

                # エントロピー項を計算
                veh_nodes[veh_num]['log_term'] = 0.0
                for node_num in veh_nodes[veh_num].index:
                    if veh_nodes[veh_num]['node_flow'][node_num] == 0.0:
                        continue
                    link_set = veh_links[veh_num][veh_links[veh_num]['term_node'] == node_num]
                    # print(link_set)
                    for index, link in link_set.iterrows():
                        if link['link_flow'] == 0.0:
                            continue
                        # print(link['link_flow'])
                        # print(veh_nodes[veh_num]['node_flow'][node_num])
                        # print(link['link_flow'] * math.log(link['link_flow'] / veh_nodes[veh_num]['node_flow'][node_num]))
                        veh_nodes[veh_num].loc[node_num, 'log_term'] += link['link_flow'] * math.log(link['link_flow'] / (veh_nodes[veh_num]['node_flow'][node_num] * link['alpha']))
                    
                    # print(np.array(veh_nodes[veh_num]['log_term']))
                    # print(1.0/np.array(veh_nodes[veh_num]['theta']))

                obj += np.array(veh_nodes[veh_num]['log_term']) @ (1.0 / np.array(veh_nodes[veh_num]['theta']))
                # print(obj)

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                veh_links[veh_num].drop('link_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('node_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('log_term', axis=1, inplace=True)

                

        # 利用者側の目的関数値を計算
        for user_num in user_trips.keys():
            for origin_node in user_trips[user_num].keys():

                start_time = time.process_time()

                # 起点別リンクフローを設定
                num_links = len(user_links[user_num].index)
                user_links[user_num]['link_flow'] = now_sol[start_index:start_index + num_links]
                start_index += num_links

                # 起点別ノードフローを計算
                user_nodes[user_num]['node_flow'] = 0.0
                for node_num in user_nodes[user_num].index:
                    in_flow = sum(list(user_links[user_num][user_links[user_num]['term_node'] == node_num]['link_flow']))
                    user_nodes[user_num].loc[node_num, 'node_flow'] = in_flow

                # 線形項を計算
                obj += np.array(list(user_links[user_num]['link_flow'])) @ np.array(list(user_links[user_num]['free_flow_time']))
                # print(obj)

                # エントロピー項を計算
                user_nodes[user_num]['log_term'] = 0.0
                for node_num in user_nodes[user_num].index:
                    if user_nodes[user_num]['node_flow'][node_num] == 0.0:
                        continue
                    link_set = user_links[user_num][user_links[user_num]['term_node'] == node_num]
                    # print(link_set)
                    for index, link in link_set.iterrows():
                        if link['link_flow'] == 0.0:
                            continue
                        user_nodes[user_num].loc[node_num, 'log_term'] += link['link_flow'] * math.log(link['link_flow'] / (user_nodes[user_num]['node_flow'][node_num] * link['alpha']))

                obj += np.array(user_nodes[user_num]['log_term']) @ (1.0 / np.array(user_nodes[user_num]['theta']))
                # print(obj)

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                user_links[user_num].drop('link_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('node_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('log_term', axis=1, inplace=True)

        para_time += max(temp_para_time)

        # print('obj: ', obj)

        return obj, para_time, total_time

        
    # 初期解を作成する関数
    def make_init_sol():

        init_sol = np.array([])

        # 車両側のフローを全てOD直結リンクに流す
        for veh_num in veh_trips.keys():

            for origin_node in veh_trips[veh_num].keys():

                veh_links[veh_num]['now_flow'] = 0.0

                for dest_node in veh_trips[veh_num][origin_node].keys():

                    # print(origin_node, dest_node, veh_trips[veh_num][origin_node][dest_node])
                    link_set = veh_links[veh_num][(veh_links[veh_num]['init_node']==origin_node) & (veh_links[veh_num]['term_node']==dest_node)]
                    for index, link in link_set.iterrows():
                        veh_links[veh_num].loc[index, 'now_flow'] = veh_trips[veh_num][origin_node][dest_node]

                add_sol = np.array(list(veh_links[veh_num]['now_flow']))
                init_sol = np.hstack([init_sol, add_sol])
                veh_links[veh_num].drop('now_flow', axis=1, inplace=True)

        # 利用者側のフローを全てOD直結リンクに流す
        for user_num in user_trips.keys():

            for origin_node in user_trips[user_num].keys():

                user_links[user_num]['now_flow'] = 0.0

                for dest_node in user_trips[user_num][origin_node].keys():
                    link_set = user_links[user_num][(user_links[user_num]['init_node']==origin_node) & (user_links[user_num]['term_node']==dest_node)]
                    for index, link in link_set.iterrows():
                        user_links[user_num].loc[index, 'now_flow'] = user_trips[user_num][origin_node][dest_node]

                add_sol = np.array(list(user_links[user_num]['now_flow']))
                init_sol = np.hstack([init_sol, add_sol])
                user_links[user_num].drop('now_flow', axis=1, inplace=True)

        return init_sol


    init_sol = make_init_sol()
    # print(len(init_sol))

    # [init_nbl, temp_para_time, temp_total_time] = nbl_func(init_sol)
    # print(len(init_nbl))
    # print(init_nbl)
    # [init_obj, temp_para_time, temp_total_time] = obj_func(init_sol)
    # print(init_obj)

    # [B_eq, temp_para_time, temp_total_time] = make_B_eq()
    # [B, temp_para_time, temp_total_time] = make_B()
    # [b_eq, temp_para_time, temp_total_time] = make_b_eq()
    # [b, temp_para_time, temp_total_time] = make_b()
    # print(B_eq.shape)
    # print(b_eq.shape)
    # print(B.shape)
    # print(b.shape)

    # lp.linprog(init_nbl, B, b, B_eq, b_eq)

    veh_msa = msa.MSA()
    veh_msa.set_x_init(init_sol)
    veh_msa.set_obj_func(obj_func)
    veh_msa.set_dir_func(dir_func)
    veh_msa.set_conv_judge(0.1)
    veh_msa.set_output_iter(1)
    veh_msa.set_output_root(output_root)
    veh_msa.exect_MSA()


    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', veh_msa.sol_obj)
    print('iteration: ', veh_msa.iter)
    print('pararel_time: ', veh_msa.para_time)
    print('total_time: ', veh_msa.total_time)
    print('num_call_obj: ', veh_msa.num_call_obj)
    print('num_call_dir: ', veh_msa.num_call_dir)
    # print('output_data: ')
    # print(fista.output_data)

    return veh_msa










def NGEV_TNPandMS(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root):

    # 勾配関数
    def nbl_func(now_sol):

        para_time = 0.0
        total_time = 0.0

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
        num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]

        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]


        temp_para_time = []
        start_time = time.process_time()

        nbl_TNP = -np.array(list(TS_links['capacity']))
        nbl_MS = np.zeros(num_MSconst)

        end_time = time.process_time()
        para_time += end_time - start_time
        total_time += end_time - start_time

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num]['free_flow_time'])]) + now_sol_TNP @ TNP_constMat[veh_num] - now_sol_MS @ MSV_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time


            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.array([list(veh_links[veh_num]['NGEV_flow'])])
                nbl_TNP += (TNP_constMat[veh_num] @ now_flow.T).T[0]
                nbl_MS -= (MSV_constMat[veh_num] @ now_flow.T).T[0]

                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)



        for user_num in user_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入

            cost = np.array([list(user_links[user_num]['free_flow_time'])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in user_trips[user_num].keys():
                
                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                start_time = time.process_time()

                now_flow = np.array([list(user_links[user_num]['NGEV_flow'])])
                nbl_MS += (MSU_constMat[user_num] @ now_flow.T).T[0]

                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)

        para_time += max(temp_para_time)

        nbl = np.concatenate([nbl_TNP, nbl_MS])
        # min に合わせるために符号を逆に
        nbl = -nbl

        return nbl, para_time, total_time



    def obj_func(now_sol):

        para_time = 0.0
        total_time = 0.0

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
        num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]

        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]

        temp_para_time = []

        obj = 0.0

        for veh_num in veh_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num]['free_flow_time'])]) + now_sol_TNP @ TNP_constMat[veh_num] - now_sol_MS @ MSV_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                exp_cost = np.array([list(veh_nodes[veh_num]['exp_cost'])])
                demand = np.array([list(veh_nodes[veh_num]['demand'])])
                start_time = time.process_time()
                obj += (exp_cost @ demand.T)[0][0]
                end_time = time.process_time()
                para_time += end_time - start_time 
                total_time += end_time - start_time

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        for user_num in user_nodes.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(user_links[user_num]['free_flow_time'])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            for origin_node in user_trips[user_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                temp_time = NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')
                temp_para_time.append(temp_time)
                total_time += temp_time

                exp_cost = np.array([list(user_nodes[user_num]['exp_cost'])])
                demand = np.array([list(user_nodes[user_num]['demand'])])
                start_time = time.process_time()
                obj += (exp_cost @ demand.T)[0][0]
                end_time = time.process_time()
                para_time += end_time - start_time
                total_time += end_time - start_time

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)


        # 目的関数を計算
        obj -= now_sol_TNP @ np.array(list(TS_links['capacity']))
        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    
    def proj_func(now_sol):

        start_time = time.process_time()

        for i in range(len(now_sol)):
            if now_sol[i] < 0.0:
                now_sol[i] = 0.0

        end_time = time.process_time()

        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):

        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -min(now_nbl)
        # conv = - (now_sol @ now_nbl)
        end_time = time.process_time()

        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)

    # 初期解の設定
    num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
    num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]
    sol_init = np.zeros(num_TNPconst + num_MSconst)

    total_flow = sum([sum(list(veh_trips[list(veh_trips.keys())[0]][origin_node].values())) for origin_node in veh_trips[list(veh_trips.keys())[0]].keys()])
    max_cost = max(list(veh_links[list(veh_links.keys())[0]]['free_flow_time']))

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(total_flow / num_MSconst * max_cost)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.1)
    fista.set_output_iter(1)
    fista.set_output_root(output_root)
    fista.exect_FISTA_proj_back()

    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', fista.sol_obj)
    print('iteration: ', fista.iter)
    print('pararel_time: ', fista.para_time)
    print('total_time: ', fista.total_time)
    print('num_call_nabla: ', fista.num_call_nbl)
    print('num_call_obj: ', fista.num_call_obj)
    print('num_call_proj: ', fista.num_call_proj)
    print('num_call_conv: ', fista.num_call_conv)
    # print('output_data: ')
    # print(fista.output_data)

    return fista



if __name__ == '__main__':

    import os

    net_name = 'Sample'
    scenarios = ['Scenario_2', 'Scenario_1', 'Scenario_3', 'Scenario_0']

    for scene in scenarios:

        root = os.path.dirname(os.path.abspath('.'))
        veh_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'user')
        user_files = os.listdir(user_root)


        # 時空間ネットワークを読み込む
        TS_links = rn.read_net(os.path.join(root, '..', '_sampleData', net_name, scene, 'TS_net', 'Sample_ts_net.tntp'))
        # print(TS_links)


        # -----------------車両側の仮想ネットワーク情報を追加-------------------------------------------------------------------
        veh_links = {}
        veh_nodes = {}
        veh_trips = {}
        for file in veh_files:
            veh_links[int(file)] = rn.read_net(veh_root + '\\' + file + '\Sample_vir_net.tntp')
            veh_nodes[int(file)] = rn.read_node(veh_root + '\\' + file + '\Sample_vir_node.tntp')
            veh_trips[int(file)] = rn.read_trips(veh_root + '\\' + file + '\Sample_vir_trips.tntp')


            # nodes の要らない情報を削除
            keys = veh_nodes[int(file)].columns
            veh_nodes[int(file)].drop(keys, axis=1, inplace=True)
            veh_nodes[int(file)]['theta'] = 1.0


            # links の要らない情報を削除
            keys = veh_links[int(file)].columns
            for key in keys:
                if key == 'init_node' or key == 'term_node' or key == 'free_flow_time':
                    continue
                else:
                    veh_links[int(file)].drop(key, axis=1, inplace=True)
            veh_links[int(file)]['alpha'] = 1.0

        # print(veh_links)
        # print(veh_nodes)
        # print(veh_trips)


        
        # -----------------利用者側の仮想ネットワーク情報を追加-------------------------------------------------------------------
        user_links = {}
        user_nodes = {}
        user_trips = {}
        for file in user_files:
            user_links[int(file)] = rn.read_net(user_root + '\\' + file + '\Sample_vir_net.tntp')
            user_nodes[int(file)] = rn.read_node(user_root + '\\' + file + '\Sample_vir_node.tntp')
            user_trips[int(file)] = rn.read_trips(user_root + '\\' + file + '\Sample_vir_trips.tntp')

            # nodes の要らない情報を削除
            keys = user_nodes[int(file)].columns
            user_nodes[int(file)].drop(keys, axis=1, inplace=True)
            user_nodes[int(file)]['theta'] = 1.0

            # links の要らない情報を削除
            keys = user_links[int(file)].columns
            for key in keys:
                if key == 'init_node' or key == 'term_node' or key == 'free_flow_time':
                    continue
                else:
                    user_links[int(file)].drop(key, axis=1, inplace=True)
            user_links[int(file)]['alpha'] = 1.0

        # print(veh_links)
        # print(veh_nodes)
        # print(veh_trips)


        
        
        
        
        
        # -----------------制約条件の係数行列を取得-------------------------------------------------------------------

        veh_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'constMat', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'constMat', 'user')
        user_files = os.listdir(user_root)

        # 車両側の行列を取得
        TNP_constMat = {}
        MSV_constMat = {}
        V_incMat = {}
        for file in veh_files:
            TNP_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\TNP_constMat')
            MSV_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\MSV_constMat')
            V_incMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\incidenceMat')

        # 利用者側の行列を取得
        MSU_constMat = {}
        U_incMat = {}
        for file in user_files:
            MSU_constMat[int(file)] = rsm.read_sparse_mat(user_root + '\\' + file + '\MSU_constMat')
            U_incMat[int(file)] = rsm.read_sparse_mat(user_root + '\\' + file + '\incidenceMat')
            
        
        # output_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'result', 'FISTA_D')
        # os.makedirs(output_root, exist_ok=True)
        # NGEV_TNPandMS(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root)

        # TNP_sol = np.zeros(len(TS_links))
        # capacity = np.array(TS_links['capacity'])
        # NGEV_CC_TNP(TNP_constMat, capacity, veh_nodes, veh_links, veh_trips, fft_name = 'free_flow_time')


        # output_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'result', 'MSA')
        # os.makedirs(output_root, exist_ok=True)
        # NGEV_TNPandMS_MSA(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root)



        output_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'result', 'Frank-wolf')
        os.makedirs(output_root, exist_ok=True)
        NGEV_TNPandMS_FrankWolf(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, V_incMat, user_nodes, user_links, user_trips, MSU_constMat, U_incMat, TS_links, output_root)


    # for orig_node in veh_trips[0].keys():

    #     print('origin_node = ', orig_node)

    #     # OD需要をセット
    #     veh_nodes[0]['demand'] = 0.0
    #     for dest_node in veh_trips[0][orig_node].keys():
    #         veh_nodes[0]['demand'][dest_node] = veh_trips[0][orig_node][dest_node]

    #     down_order = GEVsub.make_node_downstream_order(veh_nodes[0], veh_links[0], orig_node)
    #     up_order = GEVsub.make_node_upstream_order(veh_nodes[0], veh_links[0])

    #     NGEV_CC(TNP_constMat[0], np.array(TS_links['capacity']), veh_nodes[0], veh_links[0], [down_order, up_order])

    #     print(veh_links[0])
    #     print(veh_nodes[0])

    
    # print(nodes)
    # print('\n')
    # print(links[:50])
    # print(links[50:100])
    # print(links[100:])
