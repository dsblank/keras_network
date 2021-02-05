def minimum(seq):
    """
    Find the minimum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The minimum value in list or matrix.

    >>> minimum([5, 2, 3, 1])
    1
    >>> minimum([[5, 2], [3, 1]])
    1
    >>> minimum([[[5], [2]], [[3], [1]]])
    1
    """
    try:
        seq[0][0]
        return min([minimum(v) for v in seq])
    except:
        return np.array(seq).min()

def maximum(seq):
    """
    Find the maximum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The maximum value in list or matrix.

    >>> maximum([0.5, 0.2, 0.3, 0.1])
    0.5
    >>> maximum([[0.5, 0.2], [0.3, 0.1]])
    0.5
    >>> maximum([[[0.5], [0.2]], [[0.3], [0.1]]])
    0.5
    """
    try:
        seq[0][0]
        return max([maximum(v) for v in seq])
    except:
        return np.array(seq).max()

def gather_nodes(layers):
    nodes = []
    for layer in layers:
        for node in layer.inbound_nodes:
            if node not in nodes:
                nodes.append(node)

        for node in layer.outbound_nodes:
            if node not in nodes:
                nodes.append(node)
    return nodes

def topological_sort(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    nodes = topological_sort_nodes(layers)
    layer_list = []
    for node in nodes:
        if hasattr(node.inbound_layers, "__iter__"):
            for layer in node.inbound_layers:
                if layer not in layer_list:
                    layer_list.append(layers)
        else:
            if node.inbound_layers not in layer_list:
                layer_list.append(node.inbound_layers)

        if node.outbound_layer not in layer_list:
            layer_list.append(node.outbound_layer)

    return layer_list

def topological_sort_nodes(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    ## Initilize all:
    nodes = gather_nodes(layers)
    for node in nodes:
        node.visited = False
    stack = []
    for node in nodes:
        if not node.visited:
            visit_node(node, stack)
    stack.reverse()
    return stack

def visit_node(node, stack):
    """
    Utility function for topological_sort.
    """
    node.visited = True
    if node.outbound_layer:
        for subnode in node.outbound_layer.outbound_nodes:
            if not subnode.visited:
                visit_node(subnode, stack)
    stack.append(node)
