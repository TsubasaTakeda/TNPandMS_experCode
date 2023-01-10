import os
import sys
import numpy as np
sys.path.append('../Network/')
sys.path.append('../optimizationProgram/')
sys.path.append('../Matrix/')
sys.path.append('../NGEV/')
import readSparseMat as rsm
import readNetwork as rn
import LOGIT as logit
import TNPandMS_FISTA as fista
import TNPandMS_MSA as msa
import TNPandMS_PL as pl
import TNPandMS_FW as fw


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



dir_name = '_sampleData'
# networks = ['GridNet_4', 'GridNet_9', 'GridNet_16', 'GridNet_25']
networks = ['GridNet_25']
scenarios = ['Scenario_0']
# algorithms = ['TNPandMS_FISTA', 'TNPandMS_MSA', 'TNPandMS_PL', 'TNPandMS_FW']
algorithms = ['TNPandMS_FW']


for net_name in networks:

    for scene in scenarios:

        root = os.path.dirname(os.path.abspath('.'))
        veh_root = os.path.join(root, '..', dir_name, net_name, scene, 'virtual_net', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', dir_name, net_name, scene, 'virtual_net', 'user')
        user_files = os.listdir(user_root)


        # 時空間ネットワークを読み込む
        TS_links = rn.read_net(os.path.join(root, '..', dir_name, net_name, scene, 'TS_net', 'netname_ts_net.tntp'.replace('netname', net_name)))
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
            veh_links[int(file)] = rn.read_net(veh_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            veh_trips[int(file)] = rn.read_trips(veh_root + '\\' + file + '\\netname_vir_trips.tntp'.replace('netname', net_name))
            veh_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            veh_num_nodes[int(file)] = rn.read_num_nodes(veh_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))

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
            user_links[int(file)] = rn.read_net(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            user_trips[int(file)] = rn.read_trips(user_root + '\\' + file + '\\netname_vir_trips.tntp'.replace('netname', net_name))
            user_num_zones[int(file)] = rn.read_num_zones(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))
            user_num_nodes[int(file)] = rn.read_num_nodes(user_root + '\\' + file + '\\netname_vir_net.tntp'.replace('netname', net_name))

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

        veh_root = os.path.join(root, '..', dir_name, net_name, scene, 'constMat', 'vehicle')
        veh_files = os.listdir(veh_root)
        user_root = os.path.join(root, '..', dir_name, net_name, scene, 'constMat', 'user')
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
        
        for algo in algorithms:

            if algo == 'TNPandMS_FISTA':
                
                output_root = os.path.join(root, '..', dir_name, net_name, scene, 'result', 'FISTA_LOGIT')
                os.makedirs(output_root, exist_ok=True)
                fista.LOGIT_TNPandMS_FISTA(veh_info, user_info, TNP_capa, output_root)

            if algo == 'TNPandMS_MSA':

                output_root = os.path.join(root, '..', dir_name, net_name, scene, 'result', 'MSA_LOGIT')
                os.makedirs(output_root, exist_ok=True)
                msa.LOGIT_TNPandMS_MSA(veh_info, user_info, TNP_capa, temp_info, output_root)

            if algo == 'TNPandMS_PL':

                output_root = os.path.join(root, '..', dir_name, net_name, scene, 'result', 'PL_LOGIT')
                os.makedirs(output_root, exist_ok=True)
                pl.LOGIT_TNPandMS_PL(veh_info, user_info, TNP_capa, temp_info, output_root)

            if algo == 'TNPandMS_FW':

                output_root = os.path.join(root, '..', dir_name, net_name, scene, 'result', 'FW_LOGIT')
                os.makedirs(output_root, exist_ok=True)
                fw.LOGIT_TNPandMS_FW(veh_info, user_info, TNP_capa, output_root)