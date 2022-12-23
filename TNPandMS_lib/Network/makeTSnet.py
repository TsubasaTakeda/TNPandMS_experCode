import readNetwork as rn
import pandas as pd




def make_ts_net_nodes(original_nodes, num_times):
    for time in range(num_times):

        index_list = original_nodes.index + \
            [len(original_nodes)*time for i in range(len(original_nodes))]
        index_list = list(index_list)
        # print(index_list)
        X = list(original_nodes['X'])
        Y = list(original_nodes['Y'])
        o_node = list(original_nodes.index)
        time_list = [time for i in range(len(original_nodes))]

        add_df = pd.DataFrame(
            {
                'Node': index_list,
                'X': X,
                'Y': Y,
                'original_node': o_node,
                'time': time_list
            }
        )

        add_df.set_index('Node', inplace=True)

        if time == 0:
            TS_nodes = add_df
        else:
            TS_nodes = TS_nodes.append(add_df)

    return TS_nodes

# 時空間ネットワークのリンクを作成する関数
def make_ts_net_links(original_links, original_nodes, num_times, capacity_scale):
    for time in range(num_times - 1):

        index_list = original_links.index + [len(original_links)*time for i in range(len(original_links))]
        index_list = list(index_list)
        # print(index_list)
        init_node = original_links['init_node'] + [len(original_nodes)*time for i in range(len(original_links))]
        init_node = list(init_node)
        # print(init_node)
        term_node = original_links['term_node'] + [len(original_nodes)*(time + 1) for i in range(len(original_links))]
        term_node = list(term_node)
        # print(term_node)
        capacity = original_links['capacity'] * capacity_scale
        capacity = list(capacity)
        # print(capacity)
        free_flow_time = list(original_links['free_flow_time'])
        # print(free_flow_time)
        o_link = list(original_links.index)
        time_list = [time for i in range(len(original_links))]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                'capacity': capacity,
                'free_flow_time': free_flow_time,
                'original_link': o_link,
                'time': time_list
            }, index=index_list
        )

        # print(add_df)

        if time == 0:
            TS_links = add_df
        else:
            TS_links = TS_links.append(add_df)

    return TS_links


# 特殊な時空間ネットワーク(同親ノードを繋ぐリンクのみが存在する)のリンクを作成する関数
def make_ts_net_links_off(original_nodes, num_times, capacity_scale):
    
    for time in range(num_times - 1):

        index_list = [i for i in range(len(original_nodes))] 
        for i in range(len(index_list)):
            index_list[i] += len(original_nodes)*time 
        # print(index_list)
        init_node = index_list
        term_node = [i for i in range(len(original_nodes))]
        for i in range(len(term_node)):
            term_node[i] += len(original_nodes)*(time + 1)
        free_flow_time = [1.0 for i in range(len(original_nodes))]
        time_list = [time for i in range(len(original_nodes))]
        # print(time_list)

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                'free_flow_time': free_flow_time,
                'time': time_list
            }, index=index_list
        )

        # print(add_df)

        if time == 0:
            TS_links = add_df
        else:
            TS_links = TS_links.append(add_df)

    # print(TS_links)

    return TS_links

# 時空間ネットワークを作成するプログラム
def make_TS_net(original_links, original_nodes, num_times, capacity_scale):

    # ノード作成
    TS_nodes = make_ts_net_nodes(original_nodes, num_times)

    # リンク作成
    TS_links = make_ts_net_links(original_links, original_nodes, num_times, capacity_scale)

    return TS_links, TS_nodes


# 特殊な時空間ネットワーク(リンクが同親のノード同士を繋ぐネットワーク)を作成するプログラム
def make_TS_net_off(original_nodes, num_times, capacity_scale):

    # ノード作成
    TS_nodes = make_ts_net_nodes(original_nodes, num_times)

    # リンク作成
    TS_links = make_ts_net_links_off(original_nodes, num_times, capacity_scale)

    return TS_links, TS_nodes


def make_vehicle_net(original_links, original_nodes, num_zones, num_times, capa_scale, vehicle_info):

    [ts_net_links, ts_net_nodes] = make_TS_net(original_links, original_nodes, num_times, capa_scale)
    print(ts_net_links)
    print(ts_net_nodes)

    print(vehicle_info)


if __name__ == "__main__":

    import os

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample')

    root1 = os.path.join(root, 'Sample_net.tntp')
    links = rn.read_net(root1)
    num_zones = rn.read_num_zones(root1)
    num_nodes = rn.read_num_nodes(root1)
    # print(links)
    # print('\n\n')

    root2 = os.path.join(root, 'Sample_node.tntp')
    nodes = rn.read_node(root2)
    # print(nodes)
    # print('\n\n')

    root3 = os.path.join(root, 'Sample1_vu.tntp')
    [vehicle_info, user_info] = rn.read_vu(root3)
    num_time = rn.read_num_times(root3)
    capa_scale = rn.read_capa_scale(root3)
    # print(capa_scale)
    # print(vehicle_info)
    # print(user_info)
    # print(type(num_time))
    # print('\n\n')

    # [ts_links, ts_nodes] = make_TS_net(links, nodes, num_time, capa_scale)
    # print(ts_links)
    # print(ts_nodes)
    # [ts_links, ts_nodes] = make_TS_net_off(nodes, num_time, capa_scale)
    # print(ts_links)
    # print(ts_nodes)

    for vehicle in vehicle_info.keys():
        make_vehicle_net(links, nodes, num_zones, num_time, capa_scale, vehicle_info[vehicle])
