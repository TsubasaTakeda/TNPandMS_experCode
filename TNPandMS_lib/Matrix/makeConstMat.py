from os import makedirs
import pandas as pd
import math
import sys
sys.path.append('../Network/')
import readNetwork as rn
from scipy import sparse



def make_TNP_constMat(vehicle_links, num_const):

    const_mat = sparse.lil_matrix((num_const, len(vehicle_links)))

    for link_index in range(num_const):

        link_set = vehicle_links[vehicle_links['original_ts_link'] == link_index]
        for vir_link_index in link_set.index:
            const_mat[link_index, vir_link_index] = 1.0

    return const_mat



def make_MSV_constMat(vehicle_links, vehicle_nodes, vehicle_info, num_price_index, num_ts_links, num_in_links):

    num_const = num_price_index * (num_ts_links + num_in_links*2)
    # print(num_const)

    const_mat = sparse.lil_matrix((num_const, len(vehicle_links)))

    # ---------時空間ネットワークの係数行列------------------------------------------------------------------
    for vehicle_state in vehicle_info.index:

        for i in range(len(vehicle_info['state'][vehicle_state])):

            if vehicle_info['price_index'][vehicle_state][i] == 0:
                continue

            price_index = vehicle_info['price_index'][vehicle_state][i]
            num_coeffi = vehicle_info['state'][vehicle_state][i]
            # print('price_index = ', price_index)
            # print('num_coeffi = ', num_coeffi)

            start_row_num = (price_index-1) * (num_ts_links + num_in_links*2)

            link_set = vehicle_links[(vehicle_links['vehicle_state'] == vehicle_state) & (vehicle_links['link_type'] == 1)]
            for vir_link_index, link in link_set.iterrows():
                const_mat[start_row_num + link['original_ts_link'], vir_link_index] = num_coeffi


    # ---------inリンクの係数行列------------------------------------------------------------------
    for price_index in [i + 1 for i in range(num_price_index)]:
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + num_ts_links

        link_set = vehicle_links[(vehicle_links['price_index'] == price_index) & (vehicle_links['link_type'] == 2)]
        for vir_link_index, link in link_set.iterrows():
            original_ts_node = vehicle_nodes['original_TS_node'][link['init_node']]
            const_mat[start_row_num + original_ts_node - 1, vir_link_index] = num_coeffi

    # ---------outリンクの係数行列------------------------------------------------------------------
    for price_index in [i + 1 for i in range(num_price_index)]:
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + (num_ts_links + num_in_links)

        link_set = vehicle_links[(vehicle_links['price_index'] == price_index) & (vehicle_links['link_type'] == 3)]
        for vir_link_index, link in link_set.iterrows():
            original_ts_node = vehicle_nodes['original_TS_node'][link['init_node']]
            const_mat[start_row_num + original_ts_node - 1, vir_link_index] = num_coeffi

    return const_mat


def make_MSU_constMat(user_links, user_nodes, num_price_index, num_ts_links, num_original_links, num_in_links):

    num_const = num_price_index * (num_ts_links + num_in_links*2)
    # print(num_const)

    const_mat = sparse.lil_matrix((num_const, len(user_links)))
    # print(const_mat.shape)

    for price_index in [i + 1 for i in range(num_price_index)]:

        # ---------時空間ネットワークの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2)
        link_set = user_links[(user_links['original_link'] != -1) & (user_links['price_index'] == price_index) & (user_links['link_type'] == 1)]
        for vir_link_index, link in link_set.iterrows():
            # print('time = ', link['time'], '; original_link = ', link['original_link'], '; num_original_links = ', num_original_links)
            # print(start_row_num + link['time'] * num_original_links + link['original_link'])
            const_mat[start_row_num + link['time'] * num_original_links + link['original_link'], vir_link_index] = num_coeffi

        # ---------inリンクの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + num_ts_links

        link_set = user_links[(user_links['original_link'] != -1) & (user_links['price_index'] == price_index) & (user_links['link_type'] == 2)]
        for vir_link_index, link in link_set.iterrows():
            original_ts_node = user_nodes['original_TS_node'][link['init_node']]
            const_mat[start_row_num + original_ts_node - 1, vir_link_index] = num_coeffi

        # ---------outリンクの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + (num_ts_links + num_in_links)

        link_set = user_links[(user_links['original_link'] != -1) & (user_links['price_index'] == price_index) & (user_links['link_type'] == 3)]
        for vir_link_index, link in link_set.iterrows():
            original_ts_node = user_nodes['original_TS_node'][link['init_node']]
            const_mat[start_row_num + original_ts_node - 1, vir_link_index] = num_coeffi

    # print(const_mat)

    return const_mat
            

def make_incMat(links, nodes):

    num_const = len(nodes)
    # print(num_const)

    const_mat = sparse.lil_matrix((num_const, len(links)))
    # print(const_mat.shape)

    for i in nodes.index:

        # ---------流出リンク------------------------------------------------------------------
        num_coeffi = -1.0
        link_set = links[links['init_node'] == i]
        for link_index, link in link_set.iterrows():
            const_mat[i-1, link_index] = num_coeffi

        # ---------流入リンク------------------------------------------------------------------
        num_coeffi = 1.0
        link_set = links[links['term_node'] == i]
        for link_index, link in link_set.iterrows():
            const_mat[i-1, link_index] = num_coeffi

    return const_mat




if __name__ == '__main__':

    import os
    # import matplotlib.pyplot as plt
    import readSparseMat as rsm

    # def plot_coo_matrix(m):
    #     if not isinstance(m, sparse.coo_matrix):
    #         m = sparse.coo_matrix(m)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, facecolor='black')
    #     ax.plot(m.col, m.row, 's', color='white', ms=1)
    #     ax.set_xlim(0, m.shape[1])
    #     ax.set_ylim(0, m.shape[0])
    #     ax.set_aspect('equal')
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)
    #     ax.invert_yaxis()
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     return ax

    dir_name = '_sampleData'
    net_name = 'SiouxFalls_24'
    scene_name = 'scenario_1'


    root = os.path.dirname(os.path.abspath('.'))
    vehicle_root_0 = os.path.join(root, '..', dir_name, net_name, scene_name, 'virtual_net', 'vehicle', '0')
    user_root_0 = os.path.join(root, '..', dir_name, net_name, scene_name, 'virtual_net', 'user', '0')
    user_root_1 = os.path.join(root, '..', dir_name, net_name, scene_name, 'virtual_net', 'user', '1')

    # ノード情報を追加
    vehicle_nodes_0 = rn.read_node(vehicle_root_0 + '\\netname_vir_node.tntp'.replace('netname', net_name))
    user_nodes_0 = rn.read_node(user_root_0 + '\\netname_vir_node.tntp'.replace('netname', net_name))
    user_nodes_1 = rn.read_node(user_root_1 + '\\netname_vir_node.tntp'.replace('netname', net_name))
    # print(user_nodes)

    # リンク情報を追加
    vehicle_links_0 = rn.read_net(vehicle_root_0 + '\\netname_vir_net.tntp'.replace('netname', net_name))
    user_links_0 = rn.read_net(user_root_0 + '\\netname_vir_net.tntp'.replace('netname', net_name))
    user_links_1 = rn.read_net(user_root_1 + '\\netname_vir_net.tntp'.replace('netname', net_name))
    original_links = rn.read_net(os.path.join(root, '..', dir_name, net_name, 'netname_net.tntp'.replace('netname', net_name)))
    # print(user_links)

    root3 = os.path.join(root, '..', dir_name, net_name, scene_name, 'netname_vu.tntp'.replace('netname', net_name))
    [vehicle_info, user_info] = rn.read_vu(root3)
    capa_scale = rn.read_capa_scale(root3)
    # print(user_info)

    num_TNP_const = max(vehicle_links_0['original_ts_link']) + 1
    num_price_index = max(vehicle_links_0['price_index'])
    num_ts_nodes = max(vehicle_nodes_0['original_TS_node'])
    num_o_nodes = max(vehicle_nodes_0['original_node'])
    # print(num_ts_nodes, num_o_nodes)
    num_in_links = num_ts_nodes - num_o_nodes
    # print(num_price_index)
    # print(num_in_links)

    V_incMat_0 = make_incMat(vehicle_links_0, vehicle_nodes_0)
    U_incMat_0 = make_incMat(user_links_0, user_nodes_0)
    U_incMat_1 = make_incMat(user_links_1, user_nodes_1)

    # ax = plot_coo_matrix(V_incMat_0)
    # ax.figure.show()
    # print(V_incMat)
    # print(U_incMat)

    TNP_constMat_0 = make_TNP_constMat(vehicle_links_0, num_TNP_const)
    # print(TNP_constMat)

    MSV_constMat_0 = make_MSV_constMat(vehicle_links_0, vehicle_nodes_0, vehicle_info[0], num_price_index, num_TNP_const, num_in_links)
    # print(MSV_constMat)

    MSU_constMat_0 = make_MSU_constMat(user_links_0, user_nodes_0, num_price_index, num_TNP_const, len(original_links), num_in_links)
    MSU_constMat_1 = make_MSU_constMat(user_links_1, user_nodes_1, num_price_index, num_TNP_const, len(original_links), num_in_links)
    # print(MSU_constMat)


    root_write = os.path.join(root, '..', dir_name, net_name, 'constMat', 'vehicle', '0')
    makedirs(root_write, exist_ok=True)
    temp_root = os.path.join(root_write, 'incidenceMat')
    rsm.write_sparse_mat(temp_root, V_incMat_0)
    temp_root = os.path.join(root_write, 'TNP_constMat')
    rsm.write_sparse_mat(temp_root, TNP_constMat_0)
    temp_root = os.path.join(root_write, 'MSV_constMat')
    rsm.write_sparse_mat(temp_root, MSV_constMat_0)

    
    root_write = os.path.join(root, '..', dir_name, net_name, 'constMat', 'user', '0')
    makedirs(root_write, exist_ok=True)
    temp_root = os.path.join(root_write, 'incidenceMat')
    rsm.write_sparse_mat(temp_root, U_incMat_0)
    temp_root = os.path.join(root_write, 'MSU_constMat')
    rsm.write_sparse_mat(temp_root, MSU_constMat_0)

    root_write = os.path.join(root, '..', dir_name, net_name, 'constMat', 'user', '1')
    makedirs(root_write, exist_ok=True)
    temp_root = os.path.join(root_write, 'incidenceMat')
    rsm.write_sparse_mat(temp_root, U_incMat_1)
    temp_root = os.path.join(root_write, 'MSU_constMat')
    rsm.write_sparse_mat(temp_root, MSU_constMat_1)


