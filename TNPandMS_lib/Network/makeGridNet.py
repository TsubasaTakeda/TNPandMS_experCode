from sqlalchemy import null
import pandas as pd
import random

# Grid_networkを作成する関数
# num_node: ノード数(整数の平方数になるようにすること)
# capa_scale = [lower, upper]: capacityの下限と上限値を設定(その間でランダムに設定されます)
# fft_scale = [lower, upper]: free_flow_timeの下限と上限を設定(指定しない場合，5.0で固定になります)
def make_grid_network(num_node, capa_scale, fft_scale = null):

    def make_grid_nodes():

        per_grid = int(num_node**(1/2))

        index_list = list(range(1, num_node+1))
        X = []
        Y = []

        for tate_index in range(per_grid):
            
            temp_X = list(range(0, per_grid))
            temp_Y = [-tate_index for i in range(per_grid)]

            X += temp_X
            Y += temp_Y
        

        nodes = pd.DataFrame(
            {
                'Node': index_list,
                'X': X,
                'Y': Y,
            }
        )

        nodes.set_index('Node', inplace=True)

        return nodes

    def make_grid_links():

        per_grid = int(num_node**(1/2))

        index_list = list(range((per_grid-1) * (per_grid) * 2))

        init_node = []
        term_node = []
        capacity = []
        fft = []
        

        for tate_index in range(per_grid-1):

            for yoko_index in range(per_grid-1):

                node_num = tate_index*per_grid + yoko_index + 1
                init_node += [node_num, node_num]
                term_node += [node_num+1, node_num+per_grid]

            yoko_index = per_grid-1
            node_num = tate_index*per_grid + yoko_index+1
            init_node += [node_num]
            term_node += [node_num + per_grid]

        tate_index = per_grid-1
        for yoko_index in range(per_grid-1):
            node_num = tate_index*per_grid + yoko_index + 1
            init_node += [node_num]
            term_node += [node_num+1]

        # free_flow_time を作成
        if fft_scale == null:
            fft = [5.0 for i in range(len(index_list))]
        else:
            fft = [random.uniform(fft_scale[0], fft_scale[1]) for i in range(len(index_list))]

        # capacity を作成
        capacity = [float(int(random.uniform(capa_scale[0], capa_scale[1]))) for i in range(len(index_list))]

        links = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                'capacity': capacity,
                'free_flow_time': fft,
            }, index=index_list
        )

        return links

    grid_nodes = make_grid_nodes()
    grid_links = make_grid_links()

    return grid_nodes, grid_links

if __name__ == '__main__':

    import os
    import readNetwork as rn
    import makeODdemand as mod

    num_node = 9
    num_zones = 9
    capa_scale = [1000, 5000]
    fft_scale = [5.0, 5.0]
    total_flow = 10000

    dir_name = '_sampleData'
    net_name = 'GridNet_numnode'.replace('numnode', str(num_node))
    







    # ディレクトリの場所を取得
    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', dir_name, net_name)
    os.makedirs(root, exist_ok=True)

    # grid_netを作成
    [grid_nodes, grid_links] = make_grid_network(num_node, capa_scale, fft_scale)
    # OD需要を作成
    demand = mod.make_int_trips_random(num_zones, total_flow)

    node_root = os.path.join(root, 'netname_node.tntp'.replace('netname', net_name))
    rn.write_node(node_root, grid_nodes)

    link_root = os.path.join(root, 'netname_net.tntp'.replace('netname', net_name))
    rn.write_net(link_root, grid_links, num_zones, num_node, 1, len(grid_links))

    trips_root = os.path.join(root, 'netname_trips.tntp'.replace('netname', net_name))
    rn.write_trips(trips_root, demand, num_zones, total_flow)

    