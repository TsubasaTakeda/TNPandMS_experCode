import readNetwork as rn
import pandas as pd



def make_vehicle_net(original_links, original_nodes, num_zones, num_times, vehicle_info):

    # print(original_links)
    print('a')



def make_TS_net(original_links, original_nodes, num_times):

    print(original_nodes)


    TS_nodes = pd.DataFrame({
        'Node': [], 
        'X': [], 
        'Y': [], 
        'original_node': [], 
        'time': [],
    })

    TS_nodes.set_index('Node', inplace=True)

    data = pd.Series([0.00, 0.00, 1, 0], index=TS_nodes.columns, name = 1)

    TS_nodes = TS_nodes.append(data)

    print(TS_nodes)

    # print(original_links)

    return 0, 1


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
    vu = rn.read_vn(root3)
    num_time = rn.read_num_times(root3)
    # print(vu)
    # print(type(num_time))
    # print('\n\n')

    [ts_links, ts_nodes] = make_TS_net(links, nodes, num_time)
    print(ts_links)
    print(ts_nodes)

    make_vehicle_net(links, nodes, num_zones, num_time, vu[1])
