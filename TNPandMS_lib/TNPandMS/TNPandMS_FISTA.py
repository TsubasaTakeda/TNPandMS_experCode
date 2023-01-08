import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
sys.path.append('../NGEV/')
import readSparseMat as rsm
import accelGradient as ag
import readNetwork as rn
import LOGIT as logit
import time
import numpy as np
import pandas as pd


class VEH_INFO:

    def __init__(self, veh_costVec, veh_tripsMat, veh_init_incMat, veh_term_incMat, theta, TNP_constMat, MSV_constMat):
        self.veh_costVec = veh_costVec
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




def LOGIT_TNPandMS_FISTA(veh_info, user_info, TNP_capa, output_root):

    # 勾配関数
    def nbl_func(now_sol):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        num_TNPconst = veh_info[list(veh_info.keys())[0]].TNP_constMat.shape[0]
        num_MSconst = veh_info[list(veh_info.keys())[0]].MSV_constMat.shape[0]
        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]

        nbl_TNP = -TNP_capa.copy()
        nbl_MS = np.zeros(num_MSconst)

        # 車両側のフローを計算
        for veh_num in veh_info.keys():
            
            # LOGIT配分を計算
            start_time = time.process_time()
            
            # 現在の各リンクコストを計算し，costとしてlinksに代入
            costVec = veh_info[veh_num].veh_costVec + now_sol_TNP @ veh_info[veh_num].TNP_constMat - now_sol_MS @ veh_info[veh_num].MSV_constMat

            # ロジット配分を計算
            logit_flow = logit.LOGIT(costVec, veh_info[veh_num].veh_tripsMat, veh_info[veh_num].veh_init_incMat, veh_info[veh_num].veh_term_incMat, veh_info[veh_num].theta)
            logit_flow = np.reshape(logit_flow, (logit_flow.shape[0], 1))
            nbl_TNP += (veh_info[veh_num].TNP_constMat @ logit_flow).T[0]
            nbl_MS -= (veh_info[veh_num].MSV_constMat @ logit_flow).T[0]
            end_time = time.process_time()

            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)


        # 利用者側のフローを計算
        for user_num in user_info.keys():

            
            # LOGIT配分を計算
            start_time = time.process_time()
            costVec = user_info[user_num].user_costVec + now_sol_MS @ user_info[user_num].MSU_constMat
            logit_flow = logit.LOGIT(costVec, user_info[user_num].user_tripsMat, user_info[user_num].user_init_incMat, user_info[user_num].user_term_incMat, user_info[user_num].theta)
            logit_flow = np.reshape(logit_flow, (logit_flow.shape[0], 1))
            nbl_MS += (user_info[user_num].MSU_constMat @ logit_flow).T[0]
            end_time = time.process_time()

            # print(user_info[user_num].MSU_constMat[19])
            # print((user_info[user_num].MSU_constMat @ logit_flow).T[0])

            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

            # print(user_info[user_num].user_tripsMat[3])

        para_time += max(temp_para_time)

        # print(nbl_MS)

        nbl = np.concatenate([nbl_TNP, nbl_MS])
        # min に合わせるために符号を逆に
        nbl = -nbl

        return nbl, para_time, total_time


    # 目的関数を計算
    def obj_func(now_sol):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        num_TNPconst = veh_info[list(veh_info.keys())[0]].TNP_constMat.shape[0]

        now_sol_TNP = now_sol[:num_TNPconst]
        now_sol_MS = now_sol[num_TNPconst:]

        obj = 0.0

        for veh_num in veh_info.keys():

            # LOGIT配分を計算
            start_time = time.process_time()
            
            # 現在の各リンクコストを計算し，costとしてlinksに代入
            costVec = veh_info[veh_num].veh_costVec + now_sol_TNP @ veh_info[veh_num].TNP_constMat - now_sol_MS @ veh_info[veh_num].MSV_constMat

            # 総期待最小費用を計算
            logit_cost = logit.LOGIT_cost(costVec, veh_info[veh_num].veh_tripsMat, veh_info[veh_num].veh_init_incMat, veh_info[veh_num].veh_term_incMat, veh_info[veh_num].theta)
            # print('veh_cost = ', logit_cost)
            obj += logit_cost
            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        for user_num in user_info.keys():
            
            # LOGIT配分を計算
            start_time = time.process_time()
            
            # 現在の各リンクコストを計算し，costとしてlinksに代入
            costVec = user_info[user_num].user_costVec + now_sol_MS @ user_info[user_num].MSU_constMat

            # 総期待最小費用を計算
            logit_cost = logit.LOGIT_cost(costVec, user_info[user_num].user_tripsMat, user_info[user_num].user_init_incMat, user_info[user_num].user_term_incMat, user_info[user_num].theta)
            # print('user_cost = ', logit_cost)
            obj += logit_cost
            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        # 目的関数を計算
        obj -= now_sol_TNP @ TNP_capa
        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    
    def proj_func(now_sol):

        start_time = time.process_time()

        now_sol[now_sol < 0.0] = 0.0

        end_time = time.process_time()

        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):

        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if np.min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -np.min(now_nbl)
        # conv = - (now_sol @ now_nbl)
        end_time = time.process_time()

        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)

    # 初期解の設定
    num_TNPconst = veh_info[list(veh_info.keys())[0]].TNP_constMat.shape[0]
    num_MSconst = veh_info[list(veh_info.keys())[0]].MSV_constMat.shape[0]
    # num_TNPconst = TNP_constMat[list(veh_nodes.keys())[0]].shape[0]
    # num_MSconst = MSV_constMat[list(veh_nodes.keys())[0]].shape[0]
    init_sol = np.zeros(num_TNPconst + num_MSconst)

    # [nbl, temp_para_time, temp_total_time] = nbl_func(init_sol)

    # print(temp_para_time)
    # print(temp_total_time)

    # [obj, temp_para_time, temp_total_time] = obj_func(init_sol)

    # print(temp_para_time)
    # print(temp_total_time)

    total_flow = np.sum(list(veh_info[list(veh_info.keys())[0]].veh_tripsMat))
    max_cost = np.max(list(veh_info[list(veh_info.keys())[0]].veh_costVec))

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(init_sol)
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
    scenarios = ['Scenario_0', 'Scenario_1', 'Scenario_2', 'Scenario_3']

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
            
            # tripsを行列形式に変換
            veh_tripsMat[int(file)] = np.zeros(shape=(int(veh_num_zones[int(file)]/2), veh_num_nodes[int(file)]))
            for orig_node in veh_trips[int(file)].keys():
                for dest_node in veh_trips[int(file)][orig_node].keys():
                    veh_tripsMat[int(file)][orig_node-1, dest_node-1] = veh_trips[int(file)][orig_node][dest_node]

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
            user_tripsMat[int(file)] = np.zeros(shape=(int(user_num_zones[int(file)]/2), user_num_nodes[int(file)]))
            for orig_node in user_trips[int(file)].keys():
                for dest_node in user_trips[int(file)][orig_node].keys():
                    user_tripsMat[int(file)][orig_node-1, dest_node-1] = user_trips[int(file)][orig_node][dest_node]

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

        
        output_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'result', 'FISTA_D_LOGIT')
        os.makedirs(output_root, exist_ok=True)
        LOGIT_TNPandMS_FISTA(veh_info, user_info, TNP_capa, output_root)
