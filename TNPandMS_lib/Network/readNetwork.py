# import os
# import sys
from nbformat import write
import numpy as np
import pandas as pd
import openmatrix as omx


def read_net(netfile):

    net = pd.read_csv(netfile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop(['~', ';'], axis=1, inplace=True)

    return net


def read_node(nodefile):
    
    node = pd.read_csv(nodefile, sep='\t', index_col=0)

    node.drop([';'], axis=1, inplace=True)
    
    return node

def read_trips(tripsfile):

    f = open(tripsfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig = int(orig[0])

        d = [eval('{'+a.replace(';', ',').replace(' ', '') + '}')
             for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())
    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            # We map values to a index i-1, as Numpy is base 0
            mat[i, j] = matrix.get(i+1, {}).get(j+1, 0)

    index = np.arange(zones) + 1

    myfile = omx.open_file('demand.omx', 'w')
    myfile['matrix'] = mat
    myfile.create_mapping('taz', index)
    myfile.close()

    return matrix

# function to read <NUMBER OF ZONES>
def read_num_zones(netfile):

    file_data = open(netfile)

    num_zones = 0

    for i in range(8):
        line = file_data.readline()
        if "<NUMBER OF ZONES>" in line:
            num_zones = int(line.split('\t')[0][17:])
            # print(num_zones)

    return num_zones

# function to read <NUMBER OF NODES>
def read_num_nodes(netfile):

    file_data = open(netfile)

    num_nodes = 0

    for i in range(8):
        line = file_data.readline()
        if "<NUMBER OF NODES>" in line:
            num_nodes = int(line.split('\t')[0][17:])
            # print(num_nodes)

    return num_nodes

# function to read <FIRST THRU NODE>
def read_ftn(netfile):

    file_data = open(netfile)

    ftn = 0

    for i in range(8):
        line = file_data.readline()
        if "<FIRST THRU NODE>" in line:
            ftn = int(line.split('\t')[0][17:])
            # print(num_nodes)

    return ftn


# function to read <NUMBER OF LINKS>
def read_num_links(netfile):

    file_data = open(netfile)

    num_links = 0

    for i in range(8):
        line = file_data.readline()
        if "<NUMBER OF LINKS>" in line:
            num_links = int(line.split('\t')[0][17:])
            # print(num_nodes)

    return num_links


# function to read <TOTAL OD FLOW>
def read_total_flow(tripsfile):

    file_data = open(tripsfile)

    total_flow = 0

    for i in range(3):
        line = file_data.readline()
        if "<TOTAL OD FLOW>" in line:
            total_flow = int(line.split('\t')[0][17:])
            # print(num_nodes)

    return total_flow


# netfile の書き込み関数
def write_net(netfile, links, num_zones, num_nodes, ftn, num_links):

    links.sort_index()

    print('start write net')


    f = open(netfile, mode='w')
    f.write('<NUMBER OF ZONES> ' + str(num_zones))
    f.write('\n<NUMBER OF NODES> ' + str(num_nodes))
    f.write('\n<FIRST THRU NODE> ' + str(ftn))
    f.write('\n<NUMBER OF LINKS> ' + str(num_links))
    f.write('\n<ORIGINAL HEADER> ')
    f.write('\n<END OF METADATA>\n\n\n')
    f.write('~')
    for key in links.columns:
        f.write('\t' + key)
    f.write('\t;')
    for index in links.index:
        f.write('\n')
        for key in links.columns:
            f.write('\t' + str(links[key][index]))
        f.write('\t;')
    f.close()



    print('end write net')


# nodefile の書き込み関数
def write_node(netfile, nodes):

    nodes.sort_index()

    print('start write nodes')

    f = open(netfile, mode='w')
    f.write('Node')
    for key in nodes.columns:
        f.write('\t' + key)
    f.write('\t;')
    for index in nodes.index:
        f.write('\n' + str(index))
        for key in nodes.columns:
            f.write('\t' + str(nodes[key][index]))
        f.write('\t;')
    f.close()

    print('end write nodes\n')


# tripsfile の書き込み関数
def write_trips(netfile, trips, num_zones, total_flow):

    # trips をソート
    # print(trips)
    trips = dict(sorted(trips.items()))
    for key in trips.keys():
        trips[key] = dict(sorted(trips[key].items()))

    print('start write trips')

    f = open(netfile, mode='w')
    f.write('<NUMBER OF ZONES> ' + str(num_zones))
    f.write('\n<TOTAL OD FLOW> ' + str(total_flow))
    f.write('\n<END OF METADATA>\n\n')
    for key in trips.keys():
        f.write('\n\nOrigin  ' + str(key) + '\n')
        k = 0
        for key2 in trips[key].keys():
            f.write('\t' + str(key2) + ' :\t' + str(trips[key][key2]) + ';')
            k += 1
            if k % 5 == 0:
                f.write('\n')
    f.close()

    print('end write trips\n')


if __name__ == "__main__":

    import os

    name = 'SiouxFalls'

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', name)
    
    root1 = os.path.join(root, 'str1_net.tntp'.replace('str1', name))
    # print(root1)
    net = read_net(root1)
    # print(net)
    # print('\n\n')

    num_zones = read_num_zones(root1)
    num_nodes = read_num_nodes(root1)
    ftn = read_ftn(root1)
    num_links = read_num_links(root1)

    root2 = os.path.join(root, 'str1_node.tntp'.replace('str1', name))
    node = read_node(root2)
    total_flow = read_total_flow(root2)
    # print(node)
    # print('\n\n')

    root3 = os.path.join(root, 'str1_trips.tntp'.replace('str1', name))
    # root3 = os.path.join(root, '..', 'SiouxFalls', 'SiouxFalls_trips.tntp')
    trips = read_trips(root3)
    print(trips)

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', name, 'TSnet')
    os.makedirs(root, exist_ok=True)
    root = os.path.join(root, 'str1TS_net.tntp'.replace('str1', name))

    write_net(root, net, num_zones, num_nodes, ftn, num_links)
    print('\n\n')
    net = read_net(root)
    print(net)

    root = os.path.join(root, '..', 'str1TS_node.tntp'.replace('str1', name))
    write_node(root, node)
    node = read_node(root)
    print(node)

    root = os.path.join(root, '..', 'str1TS_trips.tntp'.replace('str1', name))
    write_trips(root, trips, num_zones, total_flow)
    trips = read_trips(root)
    print(trips)