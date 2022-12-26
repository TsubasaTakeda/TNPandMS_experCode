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

        link_set = vehicle_links[vehicle_links['original_TS_link'] == link_index]
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
                const_mat[start_row_num + link['original_TS_link'], vir_link_index] = num_coeffi


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


def make_MSU_constMat(user_links, user_nodes, num_price_index, num_ts_links, num_in_links):

    num_const = num_price_index * (num_ts_links + num_in_links*2)
    # print(num_const)

    const_mat = sparse.lil_matrix((num_const, len(user_links)))
    # print(const_mat.shape)

    for price_index in [i + 1 for i in range(num_price_index)]:

        # ---------時空間ネットワークの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2)

        link_set = user_links[(user_links['price_index'] == price_index) & (user_links['link_type'] == 1)]
        for vir_link_index, link in link_set.iterrows():
            const_mat[start_row_num + link['original_TS_link'], vir_link_index] = num_coeffi

        # ---------inリンクの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + num_ts_links

        link_set = user_links[(user_links['price_index'] == price_index) & (user_links['link_type'] == 2)]
        for vir_link_index, link in link_set.iterrows():
            original_ts_node = user_nodes['original_TS_node'][link['init_node']]
            const_mat[start_row_num + original_ts_node - 1, vir_link_index] = num_coeffi

        # ---------outリンクの係数行列------------------------------------------------------------------
        num_coeffi = 1.0
        start_row_num = (price_index-1) * (num_ts_links + num_in_links*2) + (num_ts_links + num_in_links)

        link_set = user_links[(user_links['price_index'] == price_index) & (user_links['link_type'] == 3)]
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
    import readSparseMat as rsm

    root = os.path.dirname(os.path.abspath('.'))
    vehicle_root = os.path.join(root, '..', '_sampleData', 'Sample', 'virtual_net', 'vehicle', '0')
    user_root = os.path.join(root, '..', '_sampleData', 'Sample', 'virtual_net', 'user', '0')

    # ノード情報を追加
    vehicle_nodes = rn.read_node(vehicle_root + '\Sample_vir_node.tntp')
    user_nodes = rn.read_node(user_root + '\Sample_vir_node.tntp')
    # print(user_nodes)

    # リンク情報を追加
    vehicle_links = rn.read_net(vehicle_root + '\Sample_vir_net.tntp')
    user_links = rn.read_net(user_root + '\Sample_vir_net.tntp')
    # print(user_links)

    root3 = os.path.join(root, '..', '_sampleData', 'Sample', 'Sample1_vu.tntp')
    [vehicle_info, user_info] = rn.read_vu(root3)
    capa_scale = rn.read_capa_scale(root3)
    # print(user_info)

    num_TNP_const = max(vehicle_links['original_ts_link']) + 1
    num_price_index = max(vehicle_links['price_index'])
    num_ts_nodes = max(vehicle_nodes['original_TS_node'])
    num_o_nodes = max(vehicle_nodes['original_node'])
    # print(num_ts_nodes, num_o_nodes)
    num_in_links = num_ts_nodes - num_o_nodes
    # print(num_price_index)
    # print(num_in_links)

    V_incMat = make_incMat(vehicle_links, vehicle_nodes)
    U_incMat = make_incMat(user_links, user_nodes)
    # print(V_incMat)
    # print(U_incMat)

    TNP_constMat = make_TNP_constMat(vehicle_links, num_TNP_const)
    # print(TNP_constMat)

    MSV_constMat = make_MSV_constMat(vehicle_links, vehicle_nodes, vehicle_info[0], num_price_index, num_TNP_const, num_in_links)
    # print(MSV_constMat)

    MSU_constMat = make_MSU_constMat(user_links, user_nodes, num_price_index, num_TNP_const, num_in_links)
    # print(MSU_constMat)


    root_write = os.path.join(root, '..','_sampleData', 'Sample', 'constMat', 'user', '0')
    makedirs(root_write, exist_ok=True)
    temp_root = os.path.join(root_write, 'incidence_matrix')

    rsm.write_sparse_mat(temp_root, U_incMat)
    U_incMat = rsm.read_sparse_mat(temp_root)
    # print(U_incMat)
    # print(type(U_incMat))


