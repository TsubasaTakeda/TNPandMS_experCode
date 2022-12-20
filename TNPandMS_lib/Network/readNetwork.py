# import os
# import sys
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


if __name__ == "__main__":

    import os

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample')
    
    root1 = os.path.join(root, 'Sample_net.tntp')
    print(root1)
    net = read_net(root1)
    print(net)
    print('\n\n')

    root2 = os.path.join(root, 'Sample_node.tntp')
    node = read_node(root2)
    print(node)
    print('\n\n')

    root3 = os.path.join(root, 'Sample_trips.tntp')
    # root3 = os.path.join(root, '..', 'SiouxFalls', 'SiouxFalls_trips.tntp')
    trips = read_trips(root3)
    print(trips[1])



