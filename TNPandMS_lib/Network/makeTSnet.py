import readNetwork as rn


if __name__ == "__main__":

    import os

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample')

    root1 = os.path.join(root, 'Sample_net.tntp')
    links = rn.read_net(root1)
    print(links)
    print('\n\n')

    root2 = os.path.join(root, 'Sample_node.tntp')
    nodes = rn.read_node(root2)
    print(nodes)
    print('\n\n')

    root3 = os.path.join(root, 'Sample1_vu.tntp')
    vu = rn.read_vn(root3)
    num_time = rn.read_num_times(root3)
    print(vu)
    print(type(num_time))
    print('\n\n')
