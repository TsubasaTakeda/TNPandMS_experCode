import readNetwork as rn


if __name__ == "__main__":

    import os

    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', '_sampleData', 'Sample')

    root1 = os.path.join(root, 'Sample_net.tntp')
    print(root1)
    net = rn.read_net(root1)
    print(net)
    print('\n\n')
