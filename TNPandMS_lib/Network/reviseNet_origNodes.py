
def make_orig_incDict(nodes, orig_nodes):

    incDict = {}
    start_id = 1
    for i in range(1, len(nodes)+1):
        if i in orig_nodes:
            incDict[start_id] = i
            start_id += 1

    for i in range(1, len(nodes)+1):
        if i not in list(incDict.values()):
            incDict[start_id] = i
            start_id += 1

    return incDict

def revise_nodes(nodes, incDict):

    nodes['Node'] = range(1, len(nodes)+1)

    for i in range(1, len(nodes)+1):
        new_i = incDict[i]
        nodes.loc[new_i, 'Node'] = i

    # print(nodes)

    nodes.set_index('Node', inplace=True)
    nodes.sort_index(inplace=True)
        
    return nodes

def revise_links(links, incDict):

    links['prev_init_node'] = list(links['init_node'])
    links['prev_term_node'] = list(links['term_node'])

    for i in incDict.keys():

        if incDict[i] == i:
            continue

        prev_node = incDict[i]
        new_node = i

        link_set_init_prev = links[links['prev_init_node'] == prev_node]
        link_set_term_prev = links[links['prev_term_node'] == prev_node]

        for link_id, link in link_set_init_prev.iterrows():
            links.loc[link_id, 'init_node'] = new_node
        for link_id, link in link_set_term_prev.iterrows():
            links.loc[link_id, 'term_node'] = new_node

    links.sort_values('init_node', inplace=True)
    links.reset_index(drop=True, inplace=True)

    links.drop('prev_init_node', axis=1)
    links.drop('prev_term_node', axis=1)

    return links




if __name__ == '__main__':

    import os
    import readNetwork as rn

    dir_name = '_sampleData'
    net_name = 'SiouxFalls_24'
    orig_nodes = [i for i in range(1, 25)]



    # ディレクトリの場所を取得
    root = os.path.dirname(os.path.abspath('.'))
    root = os.path.join(root, '..', dir_name, net_name)
    os.makedirs(root, exist_ok=True)


    node_root = os.path.join(root, 'netname_node.tntp'.replace('netname', net_name))
    nodes = rn.read_node(node_root)

    link_root = os.path.join(root, 'netname_net.tntp'.replace('netname', net_name))
    links = rn.read_net(link_root)

    trips_root = os.path.join(root, 'netname_trips.tntp'.replace('netname', net_name))
    trips = rn.read_trips(trips_root)

    incDict = make_orig_incDict(nodes, orig_nodes)

    revised_nodes = revise_nodes(nodes, incDict)
    revised_links = revise_links(links, incDict)
    

    # print(revised_nodes)
    # print(revised_links)
    
    node_root = os.path.join(root, 'netname_node.tntp'.replace('netname', net_name))
    rn.write_node(node_root, revised_nodes)

    link_root = os.path.join(root, 'netname_net.tntp'.replace('netname', net_name))
    rn.write_net(link_root, revised_links, len(orig_nodes), len(revised_nodes), 1, len(revised_links))