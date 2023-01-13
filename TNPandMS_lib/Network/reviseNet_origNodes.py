
def revise_nodes(nodes, orig_nodes):

    nodes['Node'] = range(1, len(nodes)+1)

    start_index = 1
    end_index = len(nodes)
    for i in range(1, len(nodes)+1):
        # nodes.loc[i, 'Node'] = orig_nodes[i-1]
        if i in orig_nodes:
            nodes.loc[i, 'Node'] = start_index
            start_index += 1
        else:
            nodes.loc[i, 'Node'] = end_index
            end_index -= 1 

    # print(nodes)

    nodes.set_index('Node', inplace=True)
    nodes.sort_index(inplace=True)
        
    return nodes

def revise_links(links, orig_nodes):

    for i in range(1, len(orig_nodes)+1):

        link_set_init_i = links[links['init_node'] == i]
        link_set_term_i = links[links['term_node'] == i]

        orig_node_id = orig_nodes[i-1]
        link_set_init_orig = links[links['init_node'] == orig_node_id]
        link_set_term_orig = links[links['term_node'] == orig_node_id]

        for link_id, link in link_set_init_i.iterrows():
            links.loc[link_id, 'init_node'] = orig_node_id
        for link_id, link in link_set_term_i.iterrows():
            links.loc[link_id, 'term_node'] = orig_node_id
        for link_id, link in link_set_init_orig.iterrows():
            links.loc[link_id, 'init_node'] = i
        for link_id, link in link_set_term_orig.iterrows():
            links.loc[link_id, 'term_node'] = i

    links.sort_values('init_node', inplace=True)
    links.reset_index(drop=True, inplace=True)

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

    revised_nodes = revise_nodes(nodes, orig_nodes)
    revised_links = revise_links(links, orig_nodes)
    

    # print(revised_nodes)
    # print(revised_links)
    
    node_root = os.path.join(root, 'netname_node.tntp'.replace('netname', net_name))
    rn.write_node(node_root, revised_nodes)

    link_root = os.path.join(root, 'netname_net.tntp'.replace('netname', net_name))
    rn.write_net(link_root, revised_links, len(orig_nodes), len(revised_nodes), 1, len(revised_links))