from os import makedirs
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
def make_ts_net_links_off(original_nodes, num_times):
    
    for time in range(num_times - 1):

        index_list = [i + len(original_nodes)*time for i in range(len(original_nodes))] 
        # print(index_list)
        init_node = [i+1 + len(original_nodes)*time for i in range(len(original_nodes))]
        term_node = [i+1 + len(original_nodes)*(time + 1) for i in range(len(original_nodes))]
        free_flow_time = [1.0 for i in range(len(original_nodes))]
        original_link = [-1 for i in range(len(original_nodes))]
        time_list = [time for i in range(len(original_nodes))]
        # print(time_list)

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                'free_flow_time': free_flow_time,
                'original_link': original_link,
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
def make_TS_net_off(original_nodes, num_times):

    # ノード作成
    TS_nodes = make_ts_net_nodes(original_nodes, num_times)

    # リンク作成
    TS_links = make_ts_net_links_off(original_nodes, num_times)

    return TS_links, TS_nodes


def make_vehicle_net(original_links, original_nodes, num_zones, num_times, vehicle_info, vehicle_scost):


    # -------------------------------------------起点ノードを作成-------------------------------------------------------
    index_list = [i+1 for i in range(num_zones)]
    # print(index_list)
    X = list(original_nodes['X'])[:num_zones]
    Y = list(original_nodes['Y'])[:num_zones]
    # print(X)
    # print(Y)
    o_node = list(original_nodes.index)[:num_zones]
    # print(o_node)
    time_list = [-1 for i in range(num_zones)]
    # print(time_list)

    VTS_nodes = pd.DataFrame(
            {
                'Node': index_list,
                'X': X,
                'Y': Y,
                'original_node': o_node,
                'time': time_list,
                'original_TS_node': [-1 for i in range(num_zones)],
                'vehicle_state': [-2 for i in range(num_zones)]
            }
        )

    VTS_nodes.set_index('Node', inplace=True)
    # print(VTS_nodes)

    # -------------------------------------------終点ノードを作成-------------------------------------------------------
    index_list = [i+1 + num_zones for i in range(num_zones)]
    # print(index_list)
    X = list(original_nodes['X'])[:num_zones]
    Y = list(original_nodes['Y'])[:num_zones]
    # print(X)
    # print(Y)
    o_node = list(original_nodes.index)[:num_zones]
    # print(o_node)
    time_list = [-2 for i in range(num_zones)]
    # print(time_list)

    add_df = pd.DataFrame(
        {
            'Node': index_list,
            'X': X,
            'Y': Y,
            'original_node': o_node,
            'time': time_list,
            'original_TS_node': [-1 for i in range(num_zones)],
            'vehicle_state': [-1 for i in range(num_zones)]
        }
    )

    add_df.set_index('Node', inplace=True)
    VTS_nodes = VTS_nodes.append(add_df)
    # print(VTS_nodes)

    # -------------------------------------------各状態のノードを作成-------------------------------------------------------
    [ts_net_links, ts_net_nodes] = make_TS_net(original_links, original_nodes, num_times, 1.0)
    # print(ts_net_links)
    # print(ts_net_nodes)

    for i in vehicle_info.index:
        
        index_list = [num_zones*2 + len(ts_net_nodes)*i + j for j in ts_net_nodes.index]

        add_df = pd.DataFrame(
            {
                'Node': index_list,
                'X': list(ts_net_nodes['X']),
                'Y': list(ts_net_nodes['Y']),
                'original_node': list(ts_net_nodes['original_node']),
                'time': list(ts_net_nodes['time']), 
                'original_TS_node': list(ts_net_nodes.index),
                'vehicle_state': [i for j in range(len(ts_net_nodes))]
            }
        )

        add_df.set_index('Node', inplace=True)
        # print(add_df)

        VTS_nodes = VTS_nodes.append(add_df)

    # print(VTS_nodes)

    # -------------------------------------------各状態のリンクを作成-------------------------------------------------------
    for i in vehicle_info.index:

        index_list = list(ts_net_links.index + len(ts_net_links)*i)
        # print(index_list)

        init_node = list(ts_net_links['init_node'] + [num_zones * 2 + len(ts_net_nodes) * i for j in range(len(ts_net_links))])
        term_node = list(ts_net_links['term_node'] + [num_zones * 2 + len(ts_net_nodes) * i for j in range(len(ts_net_links))])

        # capacity = list(ts_net_links['capacity'])
        fft = list(ts_net_links['free_flow_time'] + vehicle_info['add_cost'][i])
        o_link = list(ts_net_links['original_link'])
        time = list(ts_net_links['time'])
        o_ts_link = list(ts_net_links.index)
        v_state = [i for j in range(len(ts_net_links))]
        link_type = [1 for j in range(len(ts_net_links))]
        price_index = [vehicle_info['price_index'][i][0] for j in range(len(ts_net_links))]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                # 'capacity': capacity,
                'free_flow_time': fft,
                'original_link': o_link,
                'time': time,
                'original_TS_link': o_ts_link,
                'vehicle_state': v_state,
                'link_type': link_type, 
                'price_index': price_index
            }, index = index_list
        )

        if i == 0:
            VTS_links = add_df
        else:
            VTS_links = VTS_links.append(add_df)


    # print(VTS_links)

    # -------------------------------------------各状態を行き来するリンクを作成-------------------------------------------------------
    # -------------------------------------------起点からstate=0に向けたリンクを作成-------------------------------------------------------
    
    for time_index in range(num_times):

        now_num_links = len(VTS_links)
        
        index_list = [now_num_links + i for i in range(num_zones)]
        # print(index_list)
        init_node = list(original_nodes.index)[:num_zones]
        term_node = list(original_nodes.index + num_zones*2 + len(original_nodes)*time_index)[:num_zones]

        fft = [1.0 for i in range(num_zones)]
        # print(fft)
        o_link = [-1 for i in range(num_zones)]
        time = [time_index for i in range(num_zones)]
        o_ts_link = [-1 for i in range(num_zones)]
        v_state = [0 for i in range(num_zones)]
        link_type = [-1 for i in range(num_zones)]
        price_index = [0 for i in range(num_zones)]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                # 'capacity': capacity,
                'free_flow_time': fft,
                'original_link': o_link,
                'time': time,
                'original_TS_link': o_ts_link,
                'vehicle_state': v_state,
                'link_type': link_type,
                'price_index': price_index
            }, index=index_list
        )

        VTS_links = VTS_links.append(add_df)


    # print(VTS_links.loc[35:])

    # now_num_links = len(VTS_links)

    # -------------------------------------------state=0から終点に向けたリンクを作成-------------------------------------------------------

    for time_index in range(num_times):

        now_num_links = len(VTS_links)
        index_list = [now_num_links + i for i in range(num_zones)]
        # print(index_list)
        init_node = list(original_nodes.index + num_zones*2 + len(original_nodes)*time_index)[:num_zones]
        term_node = list(original_nodes.index + num_zones)[:num_zones]
        # print(init_node)
        # print(term_node)

        fft = [vehicle_scost[time_index] for i in range(num_zones)]
        # # print(fft)
        o_link = [-1 for i in range(num_zones)]
        time = [time_index for i in range(num_zones)]
        o_ts_link = [-1 for i in range(num_zones)]
        v_state = [0 for i in range(num_zones)]
        link_type = [-1 for i in range(num_zones)]
        price_index = [0 for i in range(num_zones)]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                # 'capacity': capacity,
                'free_flow_time': fft,
                'original_link': o_link,
                'time': time,
                'original_TS_link': o_ts_link,
                'vehicle_state': v_state,
                'link_type': link_type,
                'price_index': price_index
            }, index=index_list
        )

        VTS_links = VTS_links.append(add_df)

    # -------------------------------------------各stateから別のstateに向けたリンクを作成-------------------------------------------------------
    for from_state in vehicle_info.index:
        for i in range(len(vehicle_info['to_state'][from_state])):

            now_num_links = len(VTS_links)

            index_list = [now_num_links + i for i in range(len(ts_net_nodes) - len(ts_net_nodes[ts_net_nodes['time'] == num_times-1]))]
            # print(index_list)

            init_node = list(VTS_nodes[(VTS_nodes['vehicle_state'] == from_state) & ( VTS_nodes['time'] != num_times-1 )].index)
            # print(init_node)

            to_state = vehicle_info['to_state'][from_state][i]
            term_node = list(VTS_nodes[(VTS_nodes['vehicle_state'] == to_state) & (VTS_nodes['time'] != 0)].index)
            # print(term_node)

            fft = [vehicle_info['to_cost'][from_state][i] for j in range(len(index_list))]
            # print(fft)

            o_link = [-1 for j in range(len(index_list))]
            time = list(VTS_nodes[(VTS_nodes['vehicle_state'] == from_state) & (VTS_nodes['time'] != num_times-1)]['time'])
            # print(time)

            o_ts_link = [-1 for j in range(len(index_list))]
            v_state = [-1 for j in range(len(index_list))]


            if vehicle_info['in_or_out'][from_state][i] == 'in':
                link_type = 2
            elif vehicle_info['in_or_out'][from_state][i] == 'out':
                link_type = 3
            else:
                link_type = 0
            link_type = [link_type for j in range(len(index_list))]

            price_index = vehicle_info['to_price_index'][from_state][i]
            price_index = [price_index for j in range(len(index_list))]

            add_df = pd.DataFrame(
                {
                    'init_node': init_node,
                    'term_node': term_node,
                    # 'capacity': capacity,
                    'free_flow_time': fft,
                    'original_link': o_link,
                    'time': time,
                    'original_TS_link': o_ts_link,
                    'vehicle_state': v_state,
                    'link_type': link_type,
                    'price_index': price_index
                }, index=index_list
            )
            # print(add_df)
            VTS_links = VTS_links.append(add_df)

    # print(VTS_links)
    # print(vehicle_info)

    return VTS_links, VTS_nodes


def make_user_net(original_links, original_nodes, num_zones, num_times, user_info, user_scost):

    # -------------------------------------------起点ノードを作成-------------------------------------------------------
    index_list = [i+1 for i in range(num_zones)]
    # print(index_list)
    X = list(original_nodes['X'])[:num_zones]
    Y = list(original_nodes['Y'])[:num_zones]
    # print(X)
    # print(Y)
    o_node = list(original_nodes.index)[:num_zones]
    # print(o_node)
    time_list = [-1 for i in range(num_zones)]
    # print(time_list)

    UTS_nodes = pd.DataFrame(
        {
            'Node': index_list,
            'X': X,
            'Y': Y,
            'original_node': o_node,
            'time': time_list,
            'original_TS_node': [-1 for i in range(num_zones)],
            'user_state': [-2 for i in range(num_zones)]
        }
    )

    UTS_nodes.set_index('Node', inplace=True)
    # print(VTS_nodes)

    # -------------------------------------------終点ノードを作成-------------------------------------------------------
    index_list = [i+1 + num_zones for i in range(num_zones)]
    # print(index_list)
    X = list(original_nodes['X'])[:num_zones]
    Y = list(original_nodes['Y'])[:num_zones]
    # print(X)
    # print(Y)
    o_node = list(original_nodes.index)[:num_zones]
    # print(o_node)
    time_list = [-2 for i in range(num_zones)]
    # print(time_list)

    add_df = pd.DataFrame(
        {
            'Node': index_list,
            'X': X,
            'Y': Y,
            'original_node': o_node,
            'time': time_list,
            'original_TS_node': [-1 for i in range(num_zones)],
            'user_state': [-1 for i in range(num_zones)]
        }
    )

    add_df.set_index('Node', inplace=True)
    UTS_nodes = UTS_nodes.append(add_df)

    # print(UTS_nodes)


    # -------------------------------------------各状態のノードを作成-------------------------------------------------------
    [ts_net_links, ts_net_nodes] = make_TS_net(original_links, original_nodes, num_times, capa_scale)
    # print(ts_net_links)
    # print(ts_net_nodes)

    [ts_net_links_off, ts_net_nodes_off] = make_TS_net_off(original_nodes, num_times)
    # print(ts_net_links_off)
    # print(ts_net_nodes_off)

    for i in user_info.index:

        # print(user_info['link_on'][i])

        if user_info['link_on'][i] == 'off':

            index_list = [num_zones*2 + len(ts_net_nodes_off)*i + j for j in ts_net_nodes_off.index]

            add_df = pd.DataFrame(
                {
                    'Node': index_list,
                    'X': list(ts_net_nodes_off['X']),
                    'Y': list(ts_net_nodes_off['Y']),
                    'original_node': list(ts_net_nodes_off['original_node']),
                    'time': list(ts_net_nodes_off['time']),
                    'original_TS_node': list(ts_net_nodes_off.index),
                    'user_state': [i for j in range(len(ts_net_nodes_off))]
                }
            )

        else:

            index_list = [num_zones*2 + len(ts_net_nodes)*i + j for j in ts_net_nodes.index]

            add_df = pd.DataFrame(
                {
                    'Node': index_list,
                    'X': list(ts_net_nodes['X']),
                    'Y': list(ts_net_nodes['Y']),
                    'original_node': list(ts_net_nodes['original_node']),
                    'time': list(ts_net_nodes_off['time']),
                    'original_TS_node': list(ts_net_nodes.index),
                    'user_state': [i for j in range(len(ts_net_nodes))]
                }
            )


        add_df.set_index('Node', inplace=True)
        # print(add_df)

        UTS_nodes = UTS_nodes.append(add_df)

    # print(UTS_nodes)


    # # -------------------------------------------各状態のリンクを作成-------------------------------------------------------
    

    for i in user_info.index:

        if i == 0:
            now_num_links = 0
        else:
            now_num_links = len(UTS_links)

        if user_info['link_on'][i] == 'off':
            
            index_list = list(ts_net_links_off.index + now_num_links)
            # print(index_list)

            init_node = list(ts_net_links_off['init_node'] + [num_zones * 2 + len(ts_net_nodes_off) * i for j in range(len(ts_net_links_off))])
            term_node = list(ts_net_links_off['term_node'] + [num_zones * 2 + len(ts_net_nodes_off) * i for j in range(len(ts_net_links_off))])

            # capacity = list(ts_net_links['capacity'])
            fft = list(ts_net_links_off['free_flow_time'])
            o_link = list(ts_net_links_off['original_link'])
            time = list(ts_net_links_off['time'])
            o_ts_link = list(ts_net_links_off.index)
            u_state = [i for j in range(len(ts_net_links_off))]
            link_type = [1 for j in range(len(ts_net_links_off))]
            price_index = [user_info['price_index'][i][0] for j in range(len(ts_net_links_off))]

            add_df = pd.DataFrame(
                {
                    'init_node': init_node,
                    'term_node': term_node,
                    # 'capacity': capacity,
                    'free_flow_time': fft,
                    'original_link': o_link,
                    'time': time,
                    'original_TS_link': o_ts_link,
                    'user_state': u_state,
                    'link_type': link_type,
                    'price_index': price_index
                }, index=index_list
            )

        else:

            index_list = list(ts_net_links.index + now_num_links)
            # print(index_list)

            init_node = list(ts_net_links['init_node'] + [num_zones * 2 + len(ts_net_nodes) * i for j in range(len(ts_net_links))])
            term_node = list(ts_net_links['term_node'] + [num_zones * 2 + len(ts_net_nodes) * i for j in range(len(ts_net_links))])

            # capacity = list(ts_net_links['capacity'])
            fft = list(ts_net_links['free_flow_time'] + user_info['add_cost'][i])
            o_link = list(ts_net_links['original_link'])
            time = list(ts_net_links['time'])
            o_ts_link = list(ts_net_links.index)
            u_state = [i for j in range(len(ts_net_links))]
            link_type = [1 for j in range(len(ts_net_links))]
            price_index = [user_info['price_index'][i][0] for j in range(len(ts_net_links))]

            add_df = pd.DataFrame(
                {
                    'init_node': init_node,
                    'term_node': term_node,
                    # 'capacity': capacity,
                    'free_flow_time': fft,
                    'original_link': o_link,
                    'time': time,
                    'original_TS_link': o_ts_link,
                    'user_state': u_state,
                    'link_type': link_type,
                    'price_index': price_index
                }, index=index_list
            )

        if i == 0:
            UTS_links = add_df
        else:
            UTS_links = UTS_links.append(add_df)

    # print(UTS_links)

    # -------------------------------------------各状態を行き来するリンクを作成-------------------------------------------------------
    # -------------------------------------------起点からstate=0に向けたリンクを作成-------------------------------------------------------

    for time_index in range(num_times):

        now_num_links = len(UTS_links)

        # print(UTS_links)

        index_list = [now_num_links + i for i in range(num_zones)]
        # print(index_list)
        init_node = list(original_nodes.index)[:num_zones]
        term_node = list(original_nodes.index + num_zones*2 + len(original_nodes)*time_index)[:num_zones]

        fft = [1.0 for i in range(num_zones)]
        # print(fft)
        o_link = [-1 for i in range(num_zones)]
        time = [time_index for i in range(num_zones)]
        o_ts_link = [-1 for i in range(num_zones)]
        u_state = [0 for i in range(num_zones)]
        link_type = [-1 for i in range(num_zones)]
        price_index = [0 for i in range(num_zones)]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                # 'capacity': capacity,
                'free_flow_time': fft,
                'original_link': o_link,
                'time': time,
                'original_TS_link': o_ts_link,
                'user_state': u_state,
                'link_type': link_type,
                'price_index': price_index
            }, index=index_list
        )

        UTS_links = UTS_links.append(add_df)

    # print(UTS_links)

    # -------------------------------------------state=0から終点に向けたリンクを作成-------------------------------------------------------

    for time_index in range(num_times):

        now_num_links = len(UTS_links)
        index_list = [now_num_links + i for i in range(num_zones)]
        # print(index_list)
        init_node = list(original_nodes.index + num_zones*2 + len(original_nodes)*time_index)[:num_zones]
        term_node = list(original_nodes.index + num_zones)[:num_zones]
        # print(init_node)
        # print(term_node)

        fft = [user_scost[time_index] for i in range(num_zones)]
        # # print(fft)
        o_link = [-1 for i in range(num_zones)]
        time = [time_index for i in range(num_zones)]
        o_ts_link = [-1 for i in range(num_zones)]
        u_state = [0 for i in range(num_zones)]
        link_type = [-1 for i in range(num_zones)]
        price_index = [0 for i in range(num_zones)]

        add_df = pd.DataFrame(
            {
                'init_node': init_node,
                'term_node': term_node,
                # 'capacity': capacity,
                'free_flow_time': fft,
                'original_link': o_link,
                'time': time,
                'original_TS_link': o_ts_link,
                'user_state': u_state,
                'link_type': link_type,
                'price_index': price_index
            }, index=index_list
        )

        UTS_links = UTS_links.append(add_df)

    # print(UTS_links[40:])

    # -------------------------------------------各stateから別のstateに向けたリンクを作成-------------------------------------------------------
    for from_state in user_info.index:
        for i in range(len(user_info['to_state'][from_state])):

            now_num_links = len(UTS_links)

            index_list = [now_num_links + i for i in range(len(ts_net_nodes) - len(ts_net_nodes[ts_net_nodes['time'] == num_times-1]))]
            # print(index_list)

            init_node = list(UTS_nodes[(UTS_nodes['user_state'] == from_state) & (UTS_nodes['time'] != num_times-1)].index)
            # print(init_node)

            to_state = user_info['to_state'][from_state][i]
            term_node = list(UTS_nodes[(UTS_nodes['user_state'] == to_state) & (UTS_nodes['time'] != 0)].index)
            # print(term_node)

            fft = [user_info['to_cost'][from_state][i] for j in range(len(index_list))]
            # print(fft)

            o_link = [-1 for j in range(len(index_list))]
            time = list(UTS_nodes[(UTS_nodes['user_state'] == from_state) & (UTS_nodes['time'] != num_times-1)]['time'])
            # print(time)

            o_ts_link = [-1 for j in range(len(index_list))]
            u_state = [-1 for j in range(len(index_list))]

            if user_info['in_or_out'][from_state][i] == 'in':
                link_type = 2
            elif user_info['in_or_out'][from_state][i] == 'out':
                link_type = 3
            else:
                link_type = 0
            link_type = [link_type for j in range(len(index_list))]

            price_index = user_info['to_price_index'][from_state][i]
            price_index = [price_index for j in range(len(index_list))]

            add_df = pd.DataFrame(
                {
                    'init_node': init_node,
                    'term_node': term_node,
                    # 'capacity': capacity,
                    'free_flow_time': fft,
                    'original_link': o_link,
                    'time': time,
                    'original_TS_link': o_ts_link,
                    'user_state': u_state,
                    'link_type': link_type,
                    'price_index': price_index
                }, index=index_list
            )
            # print(add_df)
            UTS_links = UTS_links.append(add_df)

    print(UTS_links.loc[35:70])
    # print(user_info)

    return UTS_links, UTS_nodes














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

    root4 = os.path.join(root, 'Sample_scost.tntp')
    [vehicle_scost, user_scost] = rn.read_scost(root4)
    # print(vehicle_scost)
    # print(user_scost)

    # [ts_links, ts_nodes] = make_TS_net(links, nodes, num_time, capa_scale)
    # print(ts_links)
    # print(ts_nodes)
    # [ts_links, ts_nodes] = make_TS_net_off(nodes, num_time, capa_scale)
    # print(ts_links)
    # print(ts_nodes)

    root5 = os.path.join(root, 'virtual_net')
    root5_vehicle = os.path.join(root5, 'vehicle')
    root5_user = os.path.join(root5, 'user')
    makedirs(root5_vehicle, exist_ok=True)
    makedirs(root5_user, exist_ok=True)

    for vehicle in vehicle_info.keys():
        temp_root = os.path.join(root5_vehicle, str(vehicle))
        makedirs(temp_root, exist_ok=True)
        [VTS_links, VTS_nodes] = make_vehicle_net(links, nodes, num_zones, num_time, vehicle_info[vehicle], vehicle_scost[vehicle])
        # print(VTS_links)
        # print(VTS_links[VTS_links['price_index'] == 1])
        # print(VTS_links[VTS_links['time'] == 3])
        rn.write_net(os.path.join(temp_root, 'Sample_vir_net.tntp'), VTS_links, num_zones*2, len(VTS_nodes), 1, len(VTS_links))
        rn.write_node(os.path.join(temp_root, 'Sample_vir_node.tntp'), VTS_nodes)

    for user in user_info.keys():
        temp_root = os.path.join(root5_user, str(user))
        makedirs(temp_root, exist_ok=True)
        [UTS_links, UTS_nodes] = make_user_net(links, nodes, num_zones, num_time, user_info[user], user_scost[user])
        # print(UTS_links[UTS_links['price_index'] == 1])
        # print(UTS_nodes)
        rn.write_net(os.path.join(temp_root, 'Sample_vir_net.tntp'), UTS_links, num_zones*2, len(UTS_nodes), 1, len(UTS_links))
        rn.write_node(os.path.join(temp_root, 'Sample_vir_node.tntp'), UTS_nodes)
