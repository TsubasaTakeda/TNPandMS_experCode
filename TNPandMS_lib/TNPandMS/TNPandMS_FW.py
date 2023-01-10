import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
sys.path.append('../NGEV/')
import readSparseMat as rsm
import FrankWolf as fw
import readNetwork as rn
import LOGIT as logit
import time
import numpy as np
import pandas as pd
from scipy import sparse


class VEH_INFO:

    def __init__(self, veh_costVec, veh_tripsMat, veh_init_incMat, veh_term_incMat, theta, TNP_constMat, MSV_constMat):
        self.veh_costVec = veh_costVec
        self.veh_nowCostVec = veh_costVec
        self.veh_tripsMat = veh_tripsMat
        self.veh_init_incMat = veh_init_incMat
        self.veh_term_incMat = veh_term_incMat
        self.theta = theta
        self.TNP_constMat = TNP_constMat
        self.MSV_constMat = MSV_constMat


class USER_INFO:

    def __init__(self, user_costVec, user_tripsMat, user_init_incMat, user_term_incMat, theta, MSU_constMat):
        self.user_costVec = user_costVec
        self.user_tripsMat = user_tripsMat
        self.user_init_incMat = user_init_incMat
        self.user_term_incMat = user_term_incMat
        self.theta = theta
        self.MSU_constMat = MSU_constMat



def LOGIT_TNPandMS_FW(veh_info, user_info, TNP_capa, output_root):

    
    def make_B_eq():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        B_eq = None

        # 車両側のフロー保存則の行列を作成
        for veh_num in veh_info.keys():
            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):
                veh_incMat = veh_info[veh_num].veh_term_incMat - veh_info[veh_num].veh_init_incMat
                if B_eq is None:
                    B_eq = veh_incMat
                else:
                    dammy_mat = np.zeros(shape=(veh_incMat.shape[0], B_eq.shape[1]))
                    temp_B_eq = sparse.hstack([dammy_mat, veh_incMat])
                    dammy_mat = np.zeros(shape=(B_eq.shape[0], veh_incMat.shape[1]))
                    B_eq = sparse.hstack([B_eq, dammy_mat])
                    B_eq = sparse.vstack([B_eq, temp_B_eq])


        # 利用者側のフロー保存則の行列を作成
        for user_num in user_info.keys():

            for orig_node_id in range(user_info[user_num].user_tripsMat.shape[0]):
                user_incMat = user_info[user_num].user_term_incMat - user_info[user_num].user_init_incMat
                if B_eq is None:
                    B_eq = user_incMat
                else:
                    dammy_mat = np.zeros(shape=(user_incMat.shape[0], B_eq.shape[1]))
                    temp_B_eq = sparse.hstack([dammy_mat, user_incMat])
                    dammy_mat = np.zeros(shape=(B_eq.shape[0], user_incMat.shape[1]))
                    B_eq = sparse.hstack([B_eq, dammy_mat])
                    B_eq = sparse.vstack([B_eq, temp_B_eq])

        return B_eq, para_time, total_time

    def make_b_eq():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        b_eq = np.array([])
        
        # 車両側のOD需要を追加
        for veh_num in veh_info.keys():
            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):
                temp_b_eq = veh_info[veh_num].veh_tripsMat[orig_node_id].copy()
                temp_b_eq[orig_node_id] = - np.sum(temp_b_eq)
                b_eq = np.hstack([b_eq, temp_b_eq])


        # 利用者側のOD需要を追加
        for user_num in user_info.keys():
            for orig_node_id in range(user_info[user_num].user_tripsMat.shape[0]):
                temp_b_eq = user_info[user_num].user_tripsMat[orig_node_id]
                temp_b_eq[orig_node_id] = - np.sum(temp_b_eq)
                b_eq = np.hstack([b_eq, temp_b_eq])

        return b_eq, para_time, total_time


        
    def make_B():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        TNP_B = None
        MS_B = None

        # 車両側の制約条件の保存則の行列を作成
        for veh_num in veh_info.keys():

            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):
                if TNP_B is None:
                    TNP_B = veh_info[veh_num].TNP_constMat.copy()
                else:
                    temp_TNP_B = veh_info[veh_num].TNP_constMat.copy()
                    TNP_B = np.hstack([TNP_B, temp_TNP_B])

                if MS_B is None:
                    MS_B = -veh_info[veh_num].MSV_constMat.copy()
                else:
                    temp_MS_B = -veh_info[veh_num].MSV_constMat.copy()
                    MS_B = np.hstack([MS_B, temp_MS_B])

        # 利用者側の制約条件の行列を作成
        for user_num in user_info.keys():

            for orig_node_id in range(user_info[user_num].user_tripsMat.shape[0]):
                temp_TNP_B = np.zeros(shape=(TNP_B.shape[0], user_info[user_num].MSU_constMat.shape[1]))
                TNP_B = np.hstack([TNP_B, temp_TNP_B])

                if MS_B is None:
                    MS_B = user_info[user_num].MSU_constMat.copy()
                else:
                    temp_MS_B = user_info[user_num].MSU_constMat.copy()
                    MS_B = np.hstack([MS_B, temp_MS_B])

        # nonNeg_B = -np.eye(TNP_B.shape[1])

        B = np.vstack([TNP_B, MS_B])

        return B, para_time, total_time

    def make_b():

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        # TNP制約
        TNP_b = TNP_capa.copy()   
        # MS制約   
        MS_b = np.zeros(veh_info[list(veh_info.keys())[0]].MSV_constMat.shape[0])
        # 非負制約
        num_var = 0
        for veh_num in veh_info.keys():
            num_var += veh_info[veh_num].veh_tripsMat.shape[0] * veh_info[veh_num].veh_term_incMat.shape[1]
        for user_num in user_info.keys():
            num_var += user_info[user_num].user_tripsMat.shape[0] * user_info[user_num].user_term_incMat.shape[1]
            
        # nonNeg_b = np.zeros(num_var)

        b = np.hstack([TNP_b, MS_b])

        return b, para_time, total_time



    # 勾配関数
    def nbl_func(now_flow):

        # print('before: ', veh_info[0].veh_costVec)

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        nbl_vec = np.array([])

        # ---------- 車両側の勾配を計算--------------------
        for veh_num in veh_info.keys():
            
            num_link = veh_info[veh_num].MSV_constMat.shape[1]

            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):

                start_time = time.process_time()

                # 起点別リンクフローを設定
                veh_linkFlow = now_flow[start_index:start_index+num_link]
                start_index += num_link

                # 起点別ノードフローを計算
                veh_nodeFlow = (veh_info[veh_num].veh_term_incMat @ veh_linkFlow)

                # 線形項の勾配
                temp_nbl = veh_info[veh_num].veh_costVec.copy()

                # エントロピー項の勾配
                veh_link_logTerm = np.log(veh_linkFlow, out=np.zeros_like(veh_linkFlow), where=veh_linkFlow != 0.0)
                veh_node_logTerm = np.log(veh_nodeFlow, out=np.zeros_like(veh_nodeFlow), where=veh_nodeFlow != 0.0)
                temp_nbl += veh_link_logTerm 
                temp_nbl -= veh_node_logTerm @ veh_info[veh_num].veh_term_incMat

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                nbl_vec = np.hstack([nbl_vec, temp_nbl])
    
        # ---------- 利用者側の勾配を計算--------------------
        for user_num in user_info.keys():
            
            num_link = user_info[user_num].MSU_constMat.shape[1]

            for orig_node_id in range(user_info[user_num].user_tripsMat.shape[0]):

                start_time = time.process_time()

                # 起点別リンクフローを設定
                user_linkFlow = now_flow[start_index:start_index+num_link]
                start_index += num_link

                # 起点別ノードフローを計算
                user_nodeFlow = (user_info[user_num].user_term_incMat @ user_linkFlow)

                # 線形項の勾配
                temp_nbl = user_info[user_num].user_costVec.copy()

                # エントロピー項の勾配
                user_link_logTerm = np.log(user_linkFlow, out=np.zeros_like(user_linkFlow), where=user_linkFlow != 0.0)
                user_node_logTerm = np.log(user_nodeFlow, out=np.zeros_like(user_nodeFlow), where=user_nodeFlow != 0.0)
                temp_nbl += user_link_logTerm 
                temp_nbl -= user_node_logTerm @ user_info[user_num].user_term_incMat

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

                nbl_vec = np.hstack([nbl_vec, temp_nbl])

        para_time += max(temp_para_time)

        # print('after: ', veh_info[0].veh_costVec)

        return nbl_vec, para_time, total_time

        
    # 目的関数
    def obj_func(now_flow):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        obj = 0.0

        # ---------- 車両側の目的関数値を計算--------------------
        for veh_num in veh_info.keys():

            # print(veh_info[veh_num].veh_costVec)
            
            num_link = veh_info[veh_num].MSV_constMat.shape[1]

            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):

                start_time = time.process_time()

                # 起点別リンクフローを設定
                veh_linkFlow = now_flow[start_index:start_index+num_link]
                start_index += num_link

                # 起点別ノードフローを計算
                veh_nodeFlow = (veh_info[veh_num].veh_term_incMat @ veh_linkFlow)

                # 線形項を計算
                obj += veh_info[veh_num].veh_costVec @ veh_linkFlow
                # print(veh_info[veh_num].veh_costVec)
                # print('linear term: ', veh_info[veh_num].veh_costVec @ veh_linkFlow)

                # エントロピー項を計算
                veh_link_logTerm = np.log(veh_linkFlow, out=np.zeros_like(veh_linkFlow), where=veh_linkFlow != 0.0)
                veh_node_logTerm = np.log(veh_nodeFlow, out=np.zeros_like(veh_nodeFlow), where=veh_nodeFlow != 0.0)
                obj += (veh_linkFlow @ veh_link_logTerm - veh_nodeFlow @ veh_node_logTerm) / veh_info[veh_num].theta

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

        para_time += max(temp_para_time)

    
        # ---------- 利用者側の目的関数値を計算--------------------
        for user_num in user_info.keys():
            
            num_link = user_info[user_num].MSU_constMat.shape[1]

            for orig_node_id in range(user_info[user_num].user_tripsMat.shape[0]):

                start_time = time.process_time()

                # 起点別リンクフローを設定
                user_linkFlow = now_flow[start_index:start_index+num_link]
                start_index += num_link

                # 起点別ノードフローを計算
                user_nodeFlow = (user_info[user_num].user_term_incMat @ user_linkFlow)

                # 線形項を計算
                obj += user_info[user_num].user_costVec @ user_linkFlow

                # エントロピー項を計算
                user_link_logTerm = np.log(user_linkFlow, out=np.zeros_like(user_linkFlow), where=user_linkFlow != 0.0)
                user_node_logTerm = np.log(user_nodeFlow, out=np.zeros_like(user_nodeFlow), where=user_nodeFlow != 0.0)
                obj += (user_linkFlow @ user_link_logTerm - user_nodeFlow @ user_node_logTerm) / user_info[user_num].theta

                # print(user_linkFlow @ user_link_logTerm - user_nodeFlow @ user_node_logTerm)
                # print(user_linkFlow @ user_link_logTerm)
                # print(user_nodeFlow @ user_node_logTerm)

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

        para_time += max(temp_para_time)

        return obj, para_time, total_time



    # 初期解を作成する関数
    def make_init_sol():

        init_flow = np.array([])

        for veh_num in veh_info.keys():

            for orig_node_index in range(veh_info[veh_num].veh_tripsMat.shape[0]):

                term_incMat = veh_info[veh_num].veh_term_incMat

                tripsMat = np.reshape(veh_info[veh_num].veh_tripsMat[orig_node_index], (1, term_incMat.shape[0]))
                init_incMat = np.reshape(veh_info[veh_num].veh_init_incMat[orig_node_index], (1, term_incMat.shape[1]))

                temp_veh_linkFlow = logit.trans_linkMat_to_linkVec(tripsMat, init_incMat, term_incMat)

                init_flow = np.hstack([init_flow, temp_veh_linkFlow])

        
        for user_num in user_info.keys():

            for orig_node_index in range(user_info[veh_num].user_tripsMat.shape[0]):

                term_incMat = user_info[user_num].user_term_incMat

                tripsMat = np.reshape(user_info[user_num].user_tripsMat[orig_node_index], (1, term_incMat.shape[0]))
                init_incMat = np.reshape(user_info[user_num].user_init_incMat[orig_node_index], (1, term_incMat.shape[1]))

                temp_user_linkFlow = logit.trans_linkMat_to_linkVec(tripsMat, init_incMat, term_incMat)

                init_flow = np.hstack([init_flow, temp_user_linkFlow])

        return init_flow    

    init_sol = make_init_sol()

    [B_eq, temp_para_time, temp_total_time] = make_B_eq()
    [B, temp_para_time, temp_total_time] = make_B()
    [b_eq, temp_para_time, temp_total_time] = make_b_eq()
    [b, temp_para_time, temp_total_time] = make_b()


    frankw = fw.FrankWolf()
    frankw.set_x_init(init_sol)
    frankw.set_obj_func(obj_func)
    frankw.set_nbl_func(nbl_func)
    frankw.set_B(B)
    frankw.set_b(b)
    frankw.set_B_eq(B_eq)
    frankw.set_b_eq(b_eq)
    frankw.set_lb(0.0)
    frankw.set_conv_judge(0.1)
    frankw.set_output_iter(1)
    frankw.set_output_root(output_root)
    frankw.exect_FW()


    # print('\n\n')

    # print('sol: ', fista.sol)
    print('sol_obj: ', frankw.sol_obj)
    print('iteration: ', frankw.iter)
    print('pararel_time: ', frankw.para_time)
    print('total_time: ', frankw.total_time)
    print('num_call_obj: ', frankw.num_call_obj)
    print('num_call_nbl: ', frankw.num_call_nbl)
    # print('output_data: ')
    # print(fista.output_data)

    return frankw







if __name__ == '__main__':

    import os

    net_name = 'Sample'
    # scenarios = ['Scenario_0', 'Scenario_1', 'Scenario_2', 'Scenario_3']
    scenarios = ['Scenario_2']

    for scene in scenarios:

        root = os.path.dirname(os.path.abspath('.'))
        veh_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'virtual_net', 'user')
        user_files = os.listdir(user_root)


        # 時空間ネットワークを読み込む
        TS_links = rn.read_net(os.path.join(root, '..', '_sampleData', net_name, scene, 'TS_net', 'Sample_ts_net.tntp'))
        TNP_capa = np.array(TS_links['capacity'])
        del TS_links

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
            veh_links[int(file)] = rn.read_net(veh_root + '\\' + file + '\Sample_vir_net.tntp')
            veh_trips[int(file)] = rn.read_trips(veh_root + '\\' + file + '\Sample_vir_trips.tntp')
            veh_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\Sample_vir_net.tntp')
            veh_num_nodes[int(file)] = rn.read_num_nodes(veh_root + '\\' + file + '\Sample_vir_net.tntp')

            # リンクの接続情報を行列形式で取得
            veh_init_incMat[int(file)] = logit.make_init_incMat(veh_links[int(file)], veh_num_nodes[int(file)])
            veh_term_incMat[int(file)] = logit.make_term_incMat(veh_links[int(file)], veh_num_nodes[int(file)])
            # リンクコストをベクトル形式で取得
            veh_costVec[int(file)] = np.array(veh_links[int(file)]['free_flow_time'])
            # print(veh_costVec[int(file)])
            
            # tripsを行列形式に変換
            veh_tripsMat[int(file)] = logit.make_tripsMat(veh_trips[int(file)], int(veh_num_zones[int(file)]/2), int(veh_num_nodes[int(file)]))

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
            user_links[int(file)] = rn.read_net(user_root + '\\' + file + '\Sample_vir_net.tntp')
            user_trips[int(file)] = rn.read_trips(user_root + '\\' + file + '\Sample_vir_trips.tntp')
            user_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\Sample_vir_net.tntp')
            user_num_nodes[int(file)] = rn.read_num_nodes(user_root + '\\' + file + '\Sample_vir_net.tntp')

            # リンクの接続情報を行列形式で取得
            user_init_incMat[int(file)] = logit.make_init_incMat(user_links[int(file)], user_num_nodes[int(file)])
            user_term_incMat[int(file)] = logit.make_term_incMat(user_links[int(file)], user_num_nodes[int(file)])
            # リンクコストをベクトル形式で取得
            user_costVec[int(file)] = np.array(user_links[int(file)]['free_flow_time'])

            # tripsを行列形式に変換
            user_tripsMat[int(file)] = logit.make_tripsMat(user_trips[int(file)], int(user_num_zones[int(file)]/2), int(user_num_nodes[int(file)]))

        del user_links
        del user_trips
        del user_num_nodes
        del user_num_zones

        # -----------------制約条件の係数行列を取得-------------------------------------------------------------------

        veh_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'constMat', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'constMat', 'user')
        user_files = os.listdir(user_root)

        # 車両側の行列を取得
        TNP_constMat = {}
        MSV_constMat = {}
        for file in veh_files:
            TNP_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\TNP_constMat').toarray()
            MSV_constMat[int(file)] = rsm.read_sparse_mat(veh_root + '\\' + file + '\MSV_constMat').toarray()
            # print(TNP_constMat[int(file)][0, 190])
            # print(rsm.read_sparse_mat(veh_root + '\\' + file + '\TNP_constMat'))
            # print(type(TNP_constMat[int(file)]))

        # 利用者側の行列を取得
        MSU_constMat = {}
        for file in user_files:
            MSU_constMat[int(file)] = rsm.read_sparse_mat(user_root + '\\' + file + '\MSU_constMat').toarray()
            
        # 車両側の情報をクラスとして格納
        veh_info = {}
        for veh_num in veh_tripsMat.keys():
            veh_info[veh_num] = VEH_INFO(veh_costVec[veh_num], veh_tripsMat[veh_num], veh_init_incMat[veh_num], veh_term_incMat[veh_num], 1.0, TNP_constMat[veh_num], MSV_constMat[veh_num])

        # 利用者側の情報をクラスとして格納
        user_info = {}
        for user_num in user_tripsMat.keys():
            user_info[user_num] = USER_INFO(user_costVec[user_num], user_tripsMat[user_num], user_init_incMat[user_num], user_term_incMat[user_num], 1.0, MSU_constMat[user_num])
        
        output_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'result', 'FW_LOGIT')
        os.makedirs(output_root, exist_ok=True)
        LOGIT_TNPandMS_FW(veh_info, user_info, TNP_capa, output_root)
