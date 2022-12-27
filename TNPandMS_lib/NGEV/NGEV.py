import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
import pandas as pd
import numpy as np
import math
import NGEV_sub as GEVsub
import readNetwork as rn
import accelGradient as ag
import readSparseMat as rsm



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


    # 起点ノードから順に期待最小費用を計算
    for i in node_order[0][1:]:

        exp_sum = 0

        link_set = links[links['term_node'] == i]

        # print('term_node = ', i)
        # print(link_set)


        if len(link_set) == 0:
            continue

        for index, link in link_set.iterrows():
            exp_sum += link[alpha_name] * math.exp(-nodes[theta_name][i] * (link[cost_name] + nodes['exp_cost'][link['init_node']]))
    
        if exp_sum == 0:
            nodes.loc[i, 'exp_cost'] = max_dbl
        else:
            nodes.loc[i, 'exp_cost'] = - math.log(exp_sum)/nodes[theta_name][i]

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
            for index, link in link_set.iterrows():
                links.loc[index, 'percent'] = link[alpha_name] * math.exp( - nodes[theta_name][i] * ( link[cost_name] + nodes['exp_cost'][link['init_node']] ) ) / math.exp(- nodes[theta_name][i] * nodes['exp_cost'][i])
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
        



def NGEV_TNPandMS(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root):

    # 勾配関数
    def nbl_func(now_sol):

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
        num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]

        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]

        nbl_TNP = -np.array(list(TS_links['capacity']))
        nbl_MS = np.zeros(num_MSconst)

        for veh_num in veh_nodes.keys():

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num]['free_flow_time'])]) + now_sol_TNP @ TNP_constMat[veh_num] - now_sol_MS @ MSV_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]
            # print(veh_links[veh_num])

            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]


                # cost を基に，NGEV配分を計算
                NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')
                # veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                # veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                # veh_links[veh_num].drop('percent', axis=1, inplace=True)

                now_flow = np.array([list(veh_links[veh_num]['NGEV_flow'])])
                # print((TNP_constMat[veh_num] @ now_flow.T).T[0])
                nbl_TNP += (TNP_constMat[veh_num] @ now_flow.T).T[0]
                nbl_MS -= (MSV_constMat[veh_num] @ now_flow.T).T[0]
                # print(nbl_MS)

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)



        for user_num in user_nodes.keys():

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(user_links[user_num]['free_flow_time'])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            for origin_node in user_trips[user_num].keys():
                
                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')

                now_flow = np.array([list(user_links[user_num]['NGEV_flow'])])
                nbl_MS += (MSU_constMat[user_num] @ now_flow.T).T[0]

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)


        # print(nbl_TNP)
        # print(nbl_MS)

        nbl = np.concatenate([nbl_TNP, nbl_MS])
        # min に合わせるために符号を逆に
        nbl = -nbl

        return nbl



    def obj_func(now_sol):

        num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
        num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]

        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]

        obj = 0.0

        for veh_num in veh_nodes.keys():

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(veh_links[veh_num]['free_flow_time'])]) + now_sol_TNP @ TNP_constMat[veh_num] - now_sol_MS @ MSV_constMat[veh_num]
            veh_links[veh_num]['cost'] = cost[0]
            # print(veh_links[veh_num])

            for origin_node in veh_trips[veh_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(veh_nodes[veh_num], veh_links[veh_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(veh_nodes[veh_num], veh_links[veh_num])

                # OD需要を設定
                veh_nodes[veh_num]['demand'] = 0.0
                for dest_node in veh_trips[veh_num][origin_node].keys():
                    veh_nodes[veh_num].loc[dest_node, 'demand'] = veh_trips[veh_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                NGEV(veh_nodes[veh_num], veh_links[veh_num], [down_order, up_order], cost_name='cost')

                exp_cost = np.array([list(veh_nodes[veh_num]['exp_cost'])])
                demand = np.array([list(veh_nodes[veh_num]['demand'])])
                obj += (exp_cost @ demand.T)[0][0]

                veh_nodes[veh_num].drop('NGEV_flow', axis=1, inplace=True)
                veh_nodes[veh_num].drop('exp_cost', axis=1, inplace=True)
                veh_nodes[veh_num].drop('demand', axis=1, inplace=True)
                veh_links[veh_num].drop('percent', axis=1, inplace=True)
                veh_links[veh_num].drop('NGEV_flow', axis=1, inplace=True)

        for user_num in user_nodes.keys():

            # 現在の各リンクコストを計算し，costとしてlinksに代入
            cost = np.array([list(user_links[user_num]['free_flow_time'])]) + now_sol_MS @ MSU_constMat[user_num]
            user_links[user_num]['cost'] = cost[0]

            for origin_node in user_trips[user_num].keys():

                # ノード順序を作成
                down_order = GEVsub.make_node_downstream_order(user_nodes[user_num], user_links[user_num], origin_node)
                up_order = GEVsub.make_node_upstream_order(user_nodes[user_num], user_links[user_num])

                # OD需要を設定
                user_nodes[user_num]['demand'] = 0.0
                for dest_node in user_trips[user_num][origin_node].keys():
                    user_nodes[user_num].loc[dest_node, 'demand'] = user_trips[user_num][origin_node][dest_node]

                # cost を基に，NGEV配分を計算
                NGEV(user_nodes[user_num], user_links[user_num], [down_order, up_order], cost_name='cost')

                exp_cost = np.array([list(user_nodes[user_num]['exp_cost'])])
                demand = np.array([list(user_nodes[user_num]['demand'])])
                obj += (exp_cost @ demand.T)[0][0]

                user_nodes[user_num].drop('NGEV_flow', axis=1, inplace=True)
                user_nodes[user_num].drop('exp_cost', axis=1, inplace=True)
                user_nodes[user_num].drop('demand', axis=1, inplace=True)
                user_links[user_num].drop('percent', axis=1, inplace=True)
                user_links[user_num].drop('NGEV_flow', axis=1, inplace=True)


        # 目的関数を計算
        obj -= now_sol_TNP @ np.array(list(TS_links['capacity']))
        # minに合わせるために符号を逆に
        obj = -obj
 
        return obj

    
    def proj_func(now_sol):

        for i in range(len(now_sol)):
            if now_sol[i] < 0.0:
                now_sol[i] = 0.0

        return now_sol

    def conv_func(now_sol):

        now_nbl = nbl_func(now_sol)
        if min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -min(now_nbl)

        return conv

    # 初期解の設定
    num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
    num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]
    sol_init = np.zeros(num_TNPconst + num_MSconst)


    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(0.1)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.03)
    fista.set_output_iter(1)
    fista.set_output_root(output_root)
    fista.exect_FISTA_proj_back()

    # print('\n\n')

    print('sol: ', fista.sol)
    print('sol_obj: ', fista.sol_obj)
    print('iteration: ', fista.iter)
    print('elapsed_time: ', fista.time)
    print('num_call_nabla: ', fista.num_call_nbl)
    print('num_call_obj: ', fista.num_call_obj)
    print('num_call_proj: ', fista.num_call_proj)
    print('num_call_conv: ', fista.num_call_conv)
    print('output_data: ')
    print(fista.output_data)

    return fista



if __name__ == '__main__':

    import os
    from scipy import sparse

    root = os.path.dirname(os.path.abspath('.'))
    veh_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'virtual_net', 'vehicle')
    veh_files = os.listdir(veh_root)
    user_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'virtual_net', 'user')
    user_files = os.listdir(user_root)


    # 時空間ネットワークを読み込む
    TS_links = rn.read_net(os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'TS_net', 'Sample_ts_net.tntp'))
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

    veh_root = os.path.join(root, '..', '_sampleData','Sample', 'Scenario_0', 'constMat', 'vehicle')
    veh_files = os.listdir(veh_root)
    user_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'constMat', 'user')
    user_files = os.listdir(user_root)

    # 車両側の行列を取得
    TNP_constMat = {}
    MSV_constMat = {}
    for file in veh_files:
        TNP_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\TNP_constMat')
        MSV_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\MSV_constMat')

    # 利用者側の行列を取得
    MSU_constMat = {}
    for file in user_files:
        MSU_constMat[int(file)] = rsm.read_sparse_mat(user_root + '\\' + file + '\MSU_constMat')
        
    output_root = os.path.join(root, '..', '_sampleData', 'Sample', 'Scenario_0', 'result')
    os.makedirs(output_root, exist_ok=True)
    
    NGEV_TNPandMS(veh_nodes, veh_links, veh_trips, TNP_constMat, MSV_constMat, user_nodes, user_links, user_trips, MSU_constMat, TS_links, output_root+'\\result.csv')




    # for orig_node in veh_trips[0].keys():

    #     print('origin_node = ', orig_node)

    #     # OD需要をセット
    #     veh_nodes[0]['demand'] = 0.0
    #     for dest_node in veh_trips[0][orig_node].keys():
    #         veh_nodes[0]['demand'][dest_node] = veh_trips[0][orig_node][dest_node]

    #     down_order = GEVsub.make_node_downstream_order(veh_nodes[0], veh_links[0], orig_node)
    #     up_order = GEVsub.make_node_upstream_order(veh_nodes[0], veh_links[0])

    #     NGEV(veh_nodes[0], veh_links[0], [down_order, up_order])

    #     print(veh_links[0])
    #     print(veh_nodes[0])

    # B = sparse.lil_matrix((len(links), len(links)))
    # for i in range(len(links)):
    #     B[i, i] = 1.0
    # b = np.ones(len(links))*5.0
    # # NGEV_CC_equal(B, b, nodes, links, [down_order, up_order])

    # NGEV(nodes, links, [down_order, up_order], cost_name='free_flow_time')
    
    # print(nodes)
    # print('\n')
    # print(links[:50])
    # print(links[50:100])
    # print(links[100:])
