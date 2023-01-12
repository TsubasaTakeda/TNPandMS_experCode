import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
import pandas as pd
import numpy as np
import time
from scipy import sparse
import readNetwork as rn

# tripsデータ（OD需要）を行列（起点ノード×全ノード）形式に変換する関数
def make_tripsMat(trips, num_init_node, num_nodes):
    tripsMat = np.zeros(shape=(num_init_node, num_nodes))
    for orig_node in trips.keys():
        for dest_node in trips[orig_node].keys():
            tripsMat[orig_node-1, dest_node-1] = trips[orig_node][dest_node]
    return tripsMat


# 起点ノード×リンクの接続行列を作成する関数
def make_init_incMat(links, num_nodes):

    init_incMat = np.zeros((num_nodes, len(links)), dtype=int)
    for index, link in links.iterrows():
        init_incMat[int(link['init_node'])-1, index] = 1 

    init_incMat = sparse.csr_matrix(init_incMat)

    return init_incMat

# 終点ノード×リンクの接続行列を作成する関数
def make_term_incMat(links, num_nodes):

    term_incMat = np.zeros((num_nodes, len(links)), dtype=int)
    for index, link in links.iterrows():
        term_incMat[int(link['term_node'])-1, index] = 1

    term_incMat = sparse.csr_matrix(term_incMat)

    return term_incMat

# 行列形式のリンク情報をベクトル形式に変換する関数
def trans_linkMat_to_linkVec(linkMat, init_incMat, term_incMat):

    temp_linkMat = init_incMat.T @ linkMat @ term_incMat
    linkVec = np.diag(temp_linkMat.toarray())

    return linkVec

# ベクトル形式のリンク情報を行列形式に変換する関数
def trans_linkVec_to_linkMat(linkVec, init_incMat, term_incMat):

    num_vec = linkVec.shape[0]
    row = np.arange(num_vec)
    col = np.arange(num_vec)

    temp_linkMat = sparse.csr_matrix((linkVec, (row, col)))

    linkMat = init_incMat @ temp_linkMat @ term_incMat.T

    return linkMat

# 重み行列（exp(-theta*link_cost)）作成
def make_link_weight(cost_vec, theta):

    link_weight = np.exp(-theta * cost_vec)

    return link_weight

# 期待最小費用行列を作成する関数(起点×目的のノード)
def calc_expected_minCost_mat(weight_mat):

    num_vec = weight_mat.shape[0]
    data = np.ones(num_vec)
    row = np.arange(num_vec)
    col = np.arange(num_vec)

    exp_minCost = sparse.csr_matrix((data, (row, col)))

    temp_exp_minCost = exp_minCost.copy()

    while np.max(temp_exp_minCost) > 0.0:

        temp_exp_minCost = temp_exp_minCost @ weight_mat
        exp_minCost += temp_exp_minCost

    return exp_minCost

# 期待最小費用からリンクの条件付き選択確率を計算する関数
def calc_choPer(weight_mat, exp_minCost, orig_node_id):


    temp_minCost_vec = exp_minCost[orig_node_id, :]
    data = temp_minCost_vec.data
    row = temp_minCost_vec.indices
    col = temp_minCost_vec.indices

    temp_mat = sparse.csr_matrix((data, (row, col)))
    per_nume = temp_mat @ weight_mat

    
    # ここはもう少し効率化できそう
    per_mat = np.divide(per_nume.toarray(), temp_minCost_vec.toarray(), out=np.zeros_like(per_nume.toarray()), where=temp_minCost_vec.toarray() != 0)
    per_mat = sparse.csr_matrix(per_mat)

    return per_mat

# ノードフローを計算する関数(demandをスパース行列にすると早そう)
def calc_nodeFlow(per_mat, demand):

    num_vec = per_mat.shape[0]
    data = np.ones(num_vec)
    row_col = np.arange(num_vec)

    node_per_mat = sparse.csr_matrix((data, (row_col, row_col)))
    temp_per_mat = sparse.csr_matrix((data, (row_col, row_col)))

    while np.max(temp_per_mat) > 0.0:

        temp_per_mat = temp_per_mat @ per_mat
        node_per_mat += temp_per_mat

    nodeFlow = (node_per_mat @ demand.T).T

    return nodeFlow

# リンクフローを計算する関数
def calc_linkFlow(per_mat, nodeFlow):
    
    # temp_nodeFlow = np.reshape(nodeFlow.toarray(), (1, per_mat.shape[1]))
    # linkFlow = np.multiply(temp_nodeFlow, per_mat)

    # ここはもう少し速くできそう
    linkFlow = np.multiply(nodeFlow.toarray(), per_mat.toarray())
    linkFlow = sparse.csr_matrix(linkFlow)

    return linkFlow

# ロジット配分を計算する関数
def LOGIT(cost_vec, tripsMat, init_incMat, term_incMat, theta):

    link_weight = make_link_weight(cost_vec, theta)
    weight_mat = trans_linkVec_to_linkMat(link_weight, init_incMat, term_incMat)
    exp_minCost = calc_expected_minCost_mat(weight_mat)

    # total_link_flow = np.zeros(shape = (init_incMat.shape[0], init_incMat.shape[0]))
    total_link_flow = sparse.csr_matrix(([], ([], [])), shape=(init_incMat.shape[0], init_incMat.shape[0]))

    for orig_node_id in range(tripsMat.shape[0]):

        per_mat = calc_choPer(weight_mat, exp_minCost, orig_node_id)
        node_flow = calc_nodeFlow(per_mat, tripsMat[orig_node_id])
        link_flow = calc_linkFlow(per_mat, node_flow)
        total_link_flow += link_flow

    link_flow_vec = trans_linkMat_to_linkVec(total_link_flow, init_incMat, term_incMat)

    return link_flow_vec


# 起点別ロジット配分を計算する関数(起点別リンクフローを並べたベクトルを返す)
def LOGIT_perOrig(cost_vec, tripsMat, init_incMat, term_incMat, theta):

    link_weight = make_link_weight(cost_vec, theta)
    weight_mat = trans_linkVec_to_linkMat(link_weight, init_incMat, term_incMat)
    exp_minCost = calc_expected_minCost_mat(weight_mat)

    # total_link_flow = np.zeros(shape = (init_incMat.shape[0], init_incMat.shape[0]))
    link_flow = np.array([])
    
    for orig_node_id in range(tripsMat.shape[0]):

        per_mat = calc_choPer(weight_mat, exp_minCost, orig_node_id)
        node_flow = calc_nodeFlow(per_mat, tripsMat[orig_node_id])
        temp_link_flow = calc_linkFlow(per_mat, node_flow)
        temp_link_flow_vec = trans_linkMat_to_linkVec(temp_link_flow, init_incMat, term_incMat)
        link_flow = np.hstack([link_flow, temp_link_flow_vec])

    return link_flow

# ロジット配分の総期待最小費用を計算する関数
def LOGIT_cost(cost_vec, tripsMat, init_incMat, term_incMat, theta):
    
    link_weight = make_link_weight(cost_vec, theta)
    weight_mat = trans_linkVec_to_linkMat(link_weight, init_incMat, term_incMat)
    exp_minCost = calc_expected_minCost_mat(weight_mat)

    temp_exp_minCost = - np.log(exp_minCost.toarray(), out=np.zeros_like(exp_minCost.toarray()), where=exp_minCost.toarray() != 0) / theta

    num_origin_node = tripsMat.shape[0]

    total_cost = np.sum(np.diag(temp_exp_minCost[:num_origin_node] @ tripsMat.T))

    return total_cost


if __name__ == '__main__':

    import os

    net_name = 'GridNet_36'
    scenarios = ['Scenario_0']

    for scene in scenarios:

        root = os.path.dirname(os.path.abspath('.'))
        veh_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'user')
        user_files = os.listdir(user_root)


        # 時空間ネットワークを読み込む
        TS_links = rn.read_net(os.path.join(root, '..', '_sampleData', net_name, scene, 'TS_net', 'netname_ts_net.tntp'.replace('netname', net_name)))
        # print(TS_links)


        # -----------------車両側の仮想ネットワーク情報を追加-------------------------------------------------------------------
        veh_links = {}
        veh_num_zones = {}
        veh_num_nodes = {}
        veh_trips = {}
        veh_tripsMat = {}
        veh_init_incMat = {}
        veh_term_incMat = {}
        veh_costVec = {}
        for file in veh_files:
            veh_links[int(file)] = rn.read_net(veh_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            veh_trips[int(file)] = rn.read_trips(veh_root + '\\' + file + '\\netname_vir_trips.tntp'.replace('netname', net_name))
            veh_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            veh_num_nodes[int(file)] = rn.read_num_nodes(veh_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))

            # # links の要らない情報を削除
            # keys = veh_links[int(file)].columns
            # for key in keys:
            #     if key == 'init_node' or key == 'term_node' or key == 'free_flow_time':
            #         continue
            #     else:
            #         veh_links[int(file)].drop(key, axis=1, inplace=True)
            # リンクの接続情報を行列形式で取得
            veh_init_incMat[int(file)] = make_init_incMat(veh_links[int(file)], veh_num_nodes[int(file)])
            veh_term_incMat[int(file)] = make_term_incMat(veh_links[int(file)], veh_num_nodes[int(file)])
            # リンクコストをベクトル形式で取得
            veh_costVec[int(file)] = np.array(veh_links[int(file)]['free_flow_time'])
            
            
            # tripsを行列形式に変換
            veh_tripsMat[int(file)] = make_tripsMat(veh_trips[int(file)], int(veh_num_zones[int(file)]/2), int(veh_num_nodes[int(file)]))
            veh_tripsMat[int(file)] = sparse.csr_matrix(veh_tripsMat[int(file)])
            # for orig_node in veh_trips[int(file)].keys():
            #     for dest_node in veh_trips[int(file)][orig_node].keys():
            #         veh_tripsMat[int(file)][orig_node-1, dest_node-1] = veh_trips[int(file)][orig_node][dest_node]

        del veh_links
        del veh_trips
        del veh_num_zones
        del veh_num_nodes

        # -----------------利用者側の仮想ネットワーク情報を追加-------------------------------------------------------------------
        user_links = {}
        user_num_nodes = {}
        user_num_zones = {}
        user_trips = {}
        user_tripsMat = {}
        user_init_incMat = {}
        user_term_incMat = {}
        user_costVec = {}
        for file in user_files:
            user_links[int(file)] = rn.read_net(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            user_trips[int(file)] = rn.read_trips(user_root + '\\' + file + '\\netname_vir_trips.tntp'.replace('netname', net_name))
            user_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            user_num_nodes[int(file)] = rn.read_num_nodes(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))

            # # links の要らない情報を削除
            # keys = user_links[int(file)].columns
            # for key in keys:
            #     if key == 'init_node' or key == 'term_node' or key == 'free_flow_time':
            #         continue
            #     else:
            #         user_links[int(file)].drop(key, axis=1, inplace=True)
            # リンクの接続情報を行列形式で取得
            user_init_incMat[int(file)] = make_init_incMat(user_links[int(file)], user_num_nodes[int(file)])
            user_term_incMat[int(file)] = make_term_incMat(user_links[int(file)], user_num_nodes[int(file)])
            # リンクコストをベクトル形式で取得
            user_costVec[int(file)] = np.array(user_links[int(file)]['free_flow_time'])

            # tripsを行列形式に変換
            user_tripsMat[int(file)] = make_tripsMat(user_trips[int(file)], int(user_num_zones[int(file)]/2), int(user_num_nodes[int(file)]))
            user_tripsMat[int(file)] = sparse.csr_matrix(user_tripsMat[int(file)])

        del user_links
        del user_trips
        del user_num_nodes
        del user_num_zones


        # LOGIT配分を計算してみよう！
        for veh_num in veh_tripsMat.keys():

            print('vehicle', str(veh_num), ':')

            # LOGIT配分を計算
            start_time = time.process_time()
            cost_vec = veh_costVec[veh_num]
            ligit_flow = LOGIT(cost_vec, veh_tripsMat[veh_num], veh_init_incMat[veh_num], veh_term_incMat[veh_num], theta = 1.0)
            end_time = time.process_time()
            print('LOGIT_time = ', end_time - start_time)

            # LOGIT配分の総期待最小費用を計算
            start_time = time.process_time()
            logit_cost = LOGIT_cost(cost_vec, veh_tripsMat[veh_num], veh_init_incMat[veh_num], veh_term_incMat[veh_num], theta=1.0)
            end_time = time.process_time()

            print('LOGIT_cost = ', logit_cost)
            print('LOGIT_obj_time = ', end_time - start_time)

        for user_num in user_tripsMat.keys():

            print('user', str(user_num), ':')
            
            # LOGIT配分を計算
            start_time = time.process_time()
            cost_vec = user_costVec[user_num]
            ligit_flow = LOGIT(cost_vec, user_tripsMat[veh_num], user_init_incMat[veh_num], user_term_incMat[veh_num], theta = 1.0)
            end_time = time.process_time()
            print('LOGIT_time = ', end_time - start_time)

            # LOGIT配分の総期待最小費用を計算
            start_time = time.process_time()
            logit_cost = LOGIT_cost(cost_vec, user_tripsMat[veh_num], user_init_incMat[veh_num], user_term_incMat[veh_num], theta=1.0)
            end_time = time.process_time()

            print('LOGIT_cost = ', logit_cost)
            print('LOGIT_obj_time = ', end_time - start_time)

