# -*- coding: utf-8 -*-
# ******************************************************
# keras_network: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/dsblank/keras_network
#
# ******************************************************

import io

import numpy as np


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
    except Exception:
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
    except Exception:
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
    # Initilize all:
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


def scale_output_for_image(vector, minmax, truncate=False):
    """
    Given an activation name (or something else) and an output
    vector, scale the vector.
    """
    return rescale_numpy_array(vector, minmax, (0, 255), "uint8", truncate=truncate,)


def rescale_numpy_array(a, old_range, new_range, new_dtype, truncate=False):
    """
    Given a numpy array, old min/max, a new min/max and a numpy type,
    create a new numpy array that scales the old values into the new_range.

    >>> import numpy as np
    >>> new_array = rescale_numpy_array(np.array([0.1, 0.2, 0.3]), (0, 1), (0.5, 1.), float)
    >>> ", ".join(["%.2f" % v for v in new_array])
    '0.55, 0.60, 0.65'
    """
    assert isinstance(old_range, (tuple, list)) and isinstance(new_range, (tuple, list))
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        if truncate:
            a = np.clip(a, old_min, old_max)
        else:
            raise Exception("array values are outside range %s" % (old_range,))
    new_min, new_max = new_range
    old_delta = float(old_max - old_min)
    new_delta = float(new_max - new_min)
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max) / 2).astype(new_dtype)
    else:
        return (new_min + (a - old_min) * new_delta / old_delta).astype(new_dtype)


def svg_to_image(svg, background=(255, 255, 255, 255)):
    import cairosvg
    from PIL import Image

    if isinstance(svg, bytes):
        pass
    elif isinstance(svg, str):
        svg = svg.encode("utf-8")
    else:
        raise Exception("svg_to_image takes a str, rather than %s" % type(svg))

    image_bytes = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image_bytes))
    if background is not None:
        # create a blank image, with background:
        canvas = Image.new("RGBA", image.size, background)
        try:
            canvas.paste(image, mask=image)
        except Exception:
            canvas = None  # fails on images that don't need backgrounds
        if canvas:
            return canvas
        else:
            return image
    else:
        return image
