import sys
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
sys.path.append('../NGEV/')
import readSparseMat as rsm
import accelGradient as ag
import MSA as msa
import readNetwork as rn
import LOGIT as logit
import time
import numpy as np
import pandas as pd


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

class TEMP_INFO:
    def __init__(self):
        self.MS_fista = None



# 利用者側 NGEV-CC配分アルゴリズム (no cycle)
def NGEV_CC_MS(user_info, MS_capa):

    # 勾配関数
    def nbl_func(now_sol_MS):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        nbl_MS = -MS_capa

        for user_num in user_info.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算
            costVec = user_info[user_num].user_costVec + now_sol_MS @ user_info[user_num].MSU_constMat

            # ロジット配分を計算
            logit_flow = logit.LOGIT(costVec, user_info[user_num].user_tripsMat, user_info[user_num].user_init_incMat, user_info[user_num].user_term_incMat, user_info[user_num].theta)
            logit_flow = np.reshape(logit_flow, (logit_flow.shape[0], 1))
            nbl_MS += (user_info[user_num].MSU_constMat @ logit_flow).T[0]

            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        # min に合わせるために符号を逆に
        nbl_MS = -nbl_MS

        para_time += max(temp_para_time)

        return nbl_MS, para_time, total_time
        

    def obj_func(now_sol_MS):
        
        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        # 目的関数を計算
        obj = - now_sol_MS @ MS_capa

        for user_num in user_info.keys():
            
            start_time = time.process_time()
            
            # 現在の各リンクコストを計算し，costとしてlinksに代入
            costVec = user_info[user_num].user_costVec + now_sol_MS @ user_info[user_num].MSU_constMat

            # 総期待最小費用を計算
            logit_cost = logit.LOGIT_cost(costVec, user_info[user_num].user_tripsMat, user_info[user_num].user_init_incMat, user_info[user_num].user_term_incMat, user_info[user_num].theta)
            obj += logit_cost
            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    
    def proj_func(now_sol):
        start_time = time.process_time()
        # now_sol[now_sol < 0.0] = 0.0
        now_sol = now_sol.clip(0.0)
        end_time = time.process_time()
        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):
        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if np.min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -np.min(now_nbl)
        end_time = time.process_time()
        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)


    print('\n\nstart NGEV_CC_MS')

    # 初期解の設定
    num_MSconst = user_info[list(user_info.keys())[0]].MSU_constMat.shape[0]
    sol_init = np.zeros(num_MSconst)
    
    total_flow = np.sum(list(user_info[list(user_info.keys())[0]].user_tripsMat))
    max_cost = np.max(list(user_info[list(user_info.keys())[0]].user_costVec))

    fista = ag.FISTA_PROJ_BACK()
    fista.set_x_init(sol_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_proj_func(proj_func)
    fista.set_conv_func(conv_func)
    fista.set_lips_init(total_flow/num_MSconst*max_cost)
    fista.set_back_para(1.1)
    fista.set_conv_judge(0.1)
    fista.set_output_iter(100)
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


# 車両側 NGEV-CC配分アルゴリズム (no cycle)
def NGEV_CC_TNP(veh_info, TNP_capa):

    # 勾配関数
    def nbl_func(now_sol_TNP):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        nbl_TNP = -TNP_capa

        for veh_num in veh_info.keys():

            start_time = time.process_time()

            # 現在の各リンクコストを計算
            costVec = veh_info[veh_num].veh_nowCostVec + now_sol_TNP @ veh_info[veh_num].TNP_constMat

            # ロジット配分を計算
            logit_flow = logit.LOGIT(costVec, veh_info[veh_num].veh_tripsMat, veh_info[veh_num].veh_init_incMat, veh_info[veh_num].veh_term_incMat, veh_info[veh_num].theta)
            logit_flow = np.reshape(logit_flow, (logit_flow.shape[0], 1))
            nbl_TNP += (veh_info[veh_num].TNP_constMat @ logit_flow).T[0]

            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        # min に合わせるために符号を逆に
        nbl_TNP = -nbl_TNP

        para_time += max(temp_para_time)

        return nbl_TNP, para_time, total_time
        

    def obj_func(now_sol_TNP):
        
        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        # 現在のMS価格を代入
        MS_priceVec = temp_info.MS_fista.sol

        # 目的関数を計算
        obj = - now_sol_TNP @ TNP_capa

        for veh_num in veh_info.keys():
            
            start_time = time.process_time()
            
            # 現在の各リンクコストを計算
            costVec = veh_info[veh_num].veh_costVec - MS_priceVec @ veh_info[veh_num].MSV_constMat + now_sol_TNP @ veh_info[veh_num].TNP_constMat

            # 総期待最小費用を計算
            logit_cost = logit.LOGIT_cost(costVec, veh_info[veh_num].veh_tripsMat, veh_info[veh_num].veh_init_incMat, veh_info[veh_num].veh_term_incMat, veh_info[veh_num].theta)
            obj += logit_cost
            end_time = time.process_time()
            total_time += end_time - start_time
            temp_para_time.append(end_time - start_time)

        # minに合わせるために符号を逆に
        obj = -obj

        para_time += max(temp_para_time)
 
        return obj, para_time, total_time

    
    def proj_func(now_sol):
        start_time = time.process_time()
        # now_sol[now_sol < 0.0] = 0.0
        now_sol = now_sol.clip(0.0)
        end_time = time.process_time()
        return now_sol, end_time-start_time, end_time-start_time

    def conv_func(now_sol):
        [now_nbl, para_time, total_time] = nbl_func(now_sol)
        start_time = time.process_time()
        if np.min(now_nbl) > 0:
            conv = 0.0
        else:
            conv = -np.min(now_nbl)
        end_time = time.process_time()
        return conv, para_time + (end_time - start_time), total_time + (end_time-start_time)


    print('\n\nstart NGEV_CC_TNP')

    # 初期解の設定
    num_TNPconst = veh_info[list(veh_info.keys())[0]].TNP_constMat.shape[0]
    sol_init = np.zeros(num_TNPconst)
    
    total_flow = np.sum(list(veh_info[list(veh_info.keys())[0]].veh_tripsMat))
    max_cost = np.max(list(veh_info[list(veh_info.keys())[0]].veh_costVec))

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
    print('pararel_time: ', fista.para_time)
    print('total_time: ', fista.total_time)
    # print('num_call_nabla: ', fista.num_call_nbl)
    # print('num_call_obj: ', fista.num_call_obj)
    # print('num_call_proj: ', fista.num_call_proj)
    # print('num_call_conv: ', fista.num_call_conv)
    # print('output_data: ')
    # print(fista.output_data)

    return fista












def LOGIT_TNPandMS_MSA(veh_info, user_info, TNP_capa, temp_info, output_root):

    def MS_price_to_vehCost(MS_price):

        for veh_num in veh_info.keys():
        
            veh_info[veh_num].veh_nowCostVec = veh_info[veh_num].veh_costVec - MS_price @ veh_info[veh_num].MSV_constMat

        return 0

    def TNP_price_to_vehFlow(TNP_price):

        vehFlow = np.array([])
        
        for veh_num in veh_info.keys():

            costVec = veh_info[veh_num].veh_nowCostVec + TNP_price @ veh_info[veh_num].TNP_constMat
            logit_flow = logit.LOGIT_perOrig(costVec, veh_info[veh_num].veh_tripsMat, veh_info[veh_num].veh_init_incMat, veh_info[veh_num].veh_term_incMat, veh_info[veh_num].theta)
            vehFlow = np.hstack([vehFlow, logit_flow])

        return vehFlow

    def vehFlow_to_MScapa(vehFlow):

        num_MSconst = veh_info[list(veh_info.keys())[0]].MSV_constMat.shape[0]
        MScapa = np.zeros(num_MSconst)

        start_index = 0

        for veh_num in veh_info.keys():

            num_link = veh_info[veh_num].MSV_constMat.shape[1]

            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):
                MScapa += veh_info[veh_num].MSV_constMat @ vehFlow[start_index:start_index+num_link]
                start_index += num_link

        return MScapa



    # 降下方向関数
    def dir_func(now_vehFlow):

        para_time = 0.0
        total_time = 0.0

        MS_price = temp_info.MS_fista.sol

        # MS価格から車両コストを修正
        MS_price_to_vehCost(MS_price)

        # 修正された車両コストでNGEV_CC_TNPを計算
        TNP_fista = NGEV_CC_TNP(veh_info, TNP_capa)
        TNP_price = TNP_fista.sol
        temp_vehFlow = TNP_price_to_vehFlow(TNP_price)
        para_time += TNP_fista.para_time
        total_time += TNP_fista.total_time

        dir_vec = temp_vehFlow - now_vehFlow

        return dir_vec, para_time, total_time

        


    # 目的関数
    def obj_func(now_vehFlow):

        para_time = 0.0
        total_time = 0.0
        temp_para_time = []

        start_index = 0

        obj = 0.0

        # ---------- 車両側の目的関数値を計算--------------------
        for veh_num in veh_info.keys():
            
            num_link = veh_info[veh_num].MSV_constMat.shape[1]

            for orig_node_id in range(veh_info[veh_num].veh_tripsMat.shape[0]):

                start_time = time.process_time()

                # 起点別リンクフローを設定
                veh_linkFlow = now_vehFlow[start_index:start_index+num_link]
                start_index += num_link

                # 起点別ノードフローを計算
                veh_nodeFlow = (veh_info[veh_num].veh_term_incMat @ veh_linkFlow)

                # 線形項を計算
                obj += veh_info[veh_num].veh_costVec @ veh_linkFlow

                # エントロピー項を計算
                veh_link_logTerm = np.log(veh_linkFlow, out=np.zeros_like(veh_linkFlow), where=veh_linkFlow != 0.0)
                veh_node_logTerm = np.log(veh_nodeFlow, out=np.zeros_like(veh_nodeFlow), where=veh_nodeFlow != 0.0)
                obj += (veh_linkFlow @ veh_link_logTerm - veh_nodeFlow @ veh_node_logTerm) / veh_info[veh_num].theta

                end_time = time.process_time()
                total_time += end_time - start_time
                temp_para_time.append(end_time - start_time)

        para_time += max(temp_para_time)

        # ---------- 利用者側の目的関数値を計算--------------------

        # 車両フローをMS容量に変換
        MScapa = vehFlow_to_MScapa(now_vehFlow)
        # MS容量を基に NGEV_CC_MS を計算
        temp_info.MS_fista = NGEV_CC_MS(user_info, MScapa)
        obj -= temp_info.MS_fista.sol_obj
        para_time += temp_info.MS_fista.para_time
        total_time += temp_info.MS_fista.total_time

        return obj, para_time, total_time


    # 初期解を作成する関数
    def make_init_sol():

        init_vehFlow = np.array([])

        for veh_num in veh_info.keys():

            for orig_node_index in range(veh_info[veh_num].veh_tripsMat.shape[0]):

                term_incMat = veh_info[veh_num].veh_term_incMat

                tripsMat = np.reshape(veh_info[veh_num].veh_tripsMat[orig_node_index], (1, term_incMat.shape[0]))
                init_incMat = np.reshape(veh_info[veh_num].veh_init_incMat[orig_node_index], (1, term_incMat.shape[1]))

                temp_veh_linkFlow = logit.trans_linkMat_to_linkVec(tripsMat, init_incMat, term_incMat)

                # print(temp_veh_linkFlow)

                init_vehFlow = np.hstack([init_vehFlow, temp_veh_linkFlow])

        return init_vehFlow    

    init_sol = make_init_sol()
    # print(init_sol)

    # print(obj_func(init_sol))
    # print(dir_func(init_sol))

    veh_msa = msa.MSA()
    veh_msa.set_x_init(init_sol)
    veh_msa.set_obj_func(obj_func)
    veh_msa.set_dir_func(dir_func)
    veh_msa.set_conv_judge(0.001)
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

        temp_info = TEMP_INFO()
        
        output_root = os.path.join(root, '..', '_sampleData', net_name, scene, 'result', 'MSA_LOGIT')
        os.makedirs(output_root, exist_ok=True)
        LOGIT_TNPandMS_MSA(veh_info, user_info, TNP_capa, temp_info, output_root)
